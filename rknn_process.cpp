#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "rknn_api.h"
#include "postproc_yolov5_seg_rknn2.h"
#include "postproc_yolov6_rknn3.h"
#include "yolov5_seg.h"
#include "yolov6_rknn3.h"
#include "rknn_process.h"
#include "super_enc_common.h"
#include "mpp_log.h"
#include "mpp_common.h"
#include "assert.h"
#include "dma_alloc.hpp"

#define SEG_OUT_CHN_NUM        (7)  /* rknn yolov5 seg output channel number */
#define SEG_OUT_BUF_SIZE       (1632000)  /* rknn yolov5 seg output size */

static MPP_RET dump_detect_rectangle(FILE *fp, ObjDetRetLst *result, int frm_cnt)
{
    for (int k = 0; k < result->count; k++) {
        int class_id = result->results[k].cls_id;
        if (class_id == LABEL_PERSON || class_id == LABEL_BICYCLE || class_id == LABEL_CAR ||
            class_id == LABEL_MOTORCYCLE || class_id == LABEL_BUS) {
            fprintf(fp, "frame %d obj_num %d x %d y %d w %d h %d\n",
                    frm_cnt, result->count, result->results[k].box.left,
                    result->results[k].box.top,
                    result->results[k].box.right - result->results[k].box.left,
                    result->results[k].box.bottom - result->results[k].box.top);
        }
    }

    return MPP_OK;
}

/* calculate average MV of CUs inside previous-frame predicted boxes from CME data */
static MPP_RET cme_calc_avg_mv_per_object(SuperEncCtx *sec)
{
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    CmeFrameData *cfd = &sec->cme_cur_frame;
    ObjDetRetLst *od_results = &sec->od_results;
    int ctu_total = cfd->ctu_total;
    int interval = sec->args->cme_pred_interval;
    int k, i;

    if (interval == 0 || (interval > 0 && (sec->frame_count % interval == 0))) {
        nn_ctx->show_pred_info = 0;
    } else {
        nn_ctx->show_pred_info = 1;
    }

    /* no CME data or no previous prediction available */
    if (!sec->fp_cme_out || nn_ctx->cme_prev_count == 0) {
        nn_ctx->cme_mv_count = 0;
        return MPP_OK;
    }

    int prev_count = nn_ctx->cme_prev_count;

    for (k = 0; k < prev_count; k++) {
        int sum_mv_x = 0, sum_mv_y = 0;
        int cu_count = 0;

        int box_left   = nn_ctx->cme_prev_boxes[k].left;
        int box_top    = nn_ctx->cme_prev_boxes[k].top;
        int box_right  = nn_ctx->cme_prev_boxes[k].right;
        int box_bottom = nn_ctx->cme_prev_boxes[k].bottom;

        for (i = 0; i < ctu_total; i++) {
            CmeCuInfo *cu = &cfd->cu_data[i];
            int cu_cx = cu->cu_x * cu->cu_size;
            int cu_cy = cu->cu_y * cu->cu_size;

            /* CU center falls inside previous predicted box */
            if (cu_cx >= box_left && cu_cx < box_right &&
                cu_cy >= box_top  && cu_cy < box_bottom) {
                /* skip zero-MV CUs (static background blocks) */
                if (cu->mv_x == 0 && cu->mv_y == 0)
                    continue;

                sum_mv_x += cu->mv_x;
                sum_mv_y += cu->mv_y;
                cu_count++;
            }
        }

        nn_ctx->cme_mv_x[k] = (cu_count > 0) ? (sum_mv_x / cu_count) : 0;
        nn_ctx->cme_mv_y[k] = (cu_count > 0) ? (sum_mv_y / cu_count) : 0;

        if (nn_ctx->show_pred_info) {
            /* debug: compare with detection box */
            int class_id = (k < od_results->count) ? od_results->results[k].cls_id : -1;
            mpp_log("[CME MV] obj[%d] class=%d prev_box=(%d,%d)-(%d,%d) cu=%d avg_mv=(%d,%d)\n",
                    k, class_id,
                    box_left, box_top, box_right, box_bottom,
                    cu_count, nn_ctx->cme_mv_x[k], nn_ctx->cme_mv_y[k]);
        }
    }

    nn_ctx->cme_mv_count = prev_count;

    return MPP_OK;
}

/* predict current-frame boxes from previous-frame prediction + CME MV
 * first frame: seed prev_boxes from detection results.
 * every N frames (interval): reset prev_boxes with detection results.
 *   if detection count==0, keep existing prev_boxes to avoid chain break.
 * other frames: pred = prev + mv, save pred as prev for next frame.
 */
static MPP_RET cme_predict_next_frame_boxes(SuperEncCtx *sec)
{
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    ObjDetRetLst *od_results = &sec->od_results;
    int width  = nn_ctx->input_image_width;
    int height = nn_ctx->input_image_height;
    int align = 16;
    int width_aligned  = MPP_ALIGN(width, align);
    int height_aligned = MPP_ALIGN(height, align);
    int interval = sec->args->cme_pred_interval;
    int k;

    /* interval=0: prediction disabled, use detection as prediction */
    if (interval == 0) {
        for (k = 0; k < od_results->count; k++) {
            nn_ctx->cme_pred_boxes[k] = od_results->results[k].box;
        }
        nn_ctx->cme_pred_count = od_results->count;
        for (k = 0; k < od_results->count; k++) {
            nn_ctx->cme_prev_boxes[k] = od_results->results[k].box;
        }
        nn_ctx->cme_prev_count = od_results->count;
        return MPP_OK;
    }

    /* no CME data, nothing to predict */
    if (!sec->fp_cme_out) {
        nn_ctx->cme_pred_count = 0;
        return MPP_OK;
    }

    /* prev_count==0: only use detection to seed at reset frame boundaries */
    if (nn_ctx->cme_prev_count == 0) {
        if (sec->frame_count % interval == 0 && od_results->count > 0) {
            mpp_log("[CME PRED] seed at frame %d (detect=%d)\n",
                    sec->frame_count, od_results->count);
            for (k = 0; k < od_results->count; k++) {
                nn_ctx->cme_prev_boxes[k] = od_results->results[k].box;
            }
            nn_ctx->cme_prev_count = od_results->count;
            nn_ctx->cme_reset_cnt = 0;
        }
        nn_ctx->cme_pred_count = 0;
        return MPP_OK;
    }

    /* reset frame: update prev_boxes with detection results */
    if ((nn_ctx->cme_reset_cnt + 1) % interval == 0) {
        if (od_results->count > 0) {
            mpp_log("[CME PRED] reset at frame %d (interval=%d, detect=%d)\n",
                    sec->frame_count, interval, od_results->count);
            for (k = 0; k < od_results->count; k++) {
                nn_ctx->cme_prev_boxes[k] = od_results->results[k].box;
            }
            nn_ctx->cme_prev_count = od_results->count;
        } else {
            mpp_log("[CME PRED] reset at frame %d (interval=%d) but detect=0, clear prev\n",
                    sec->frame_count, interval);
            memset(nn_ctx->cme_prev_boxes, 0, sizeof(nn_ctx->cme_prev_boxes));
            nn_ctx->cme_prev_count = 0;
        }
        nn_ctx->cme_pred_count = 0;
        nn_ctx->cme_reset_cnt = 0;
        return MPP_OK;
    }

    int prev_count = nn_ctx->cme_prev_count;

    /* predict current frame: prev_prediction + current MV */
    for (k = 0; k < prev_count; k++) {
        image_rect_t *prev = &nn_ctx->cme_prev_boxes[k];
        image_rect_t *pred = &nn_ctx->cme_pred_boxes[k];
        int mv_x = nn_ctx->cme_mv_x[k];
        int mv_y = nn_ctx->cme_mv_y[k];

        pred->left   = prev->left   - mv_x;
        pred->top    = prev->top    - mv_y;
        pred->right  = prev->right  - mv_x;
        pred->bottom = prev->bottom - mv_y;

        if (pred->left   < 0)      pred->left   = 0;
        if (pred->top    < 0)      pred->top    = 0;
        if (pred->right  > width)  pred->right  = width;
        if (pred->bottom > height) pred->bottom = height;

        /* align to 16, same as adjust_detect_rectangle_coordinate */
        pred->left   = pred->left   / align * align;
        pred->top    = pred->top    / align * align;
        pred->right  = MPP_MIN(MPP_ALIGN(pred->right,  align), width_aligned  - 1);
        pred->bottom = MPP_MIN(MPP_ALIGN(pred->bottom, align), height_aligned - 1);

        if (nn_ctx->show_pred_info) { // debug
            mpp_log("[CME PRED] obj[%d] prev=(%d,%d)-(%d,%d) mv=(%d,%d) pred=(%d,%d)-(%d,%d)\n",
                    k,
                    prev->left, prev->top, prev->right, prev->bottom,
                    mv_x, mv_y,
                    pred->left, pred->top, pred->right, pred->bottom);
        }

        /* debug: compare prediction with actual detection */
        if (k < od_results->count && nn_ctx->show_pred_info) {
            ObjDetRet *obj = &od_results->results[k];
            mpp_log("[CME COMP] obj[%d] detect=(%d,%d)-(%d,%d) pred=(%d,%d)-(%d,%d)\n",
                    k,
                    obj->box.left, obj->box.top, obj->box.right, obj->box.bottom,
                    pred->left, pred->top, pred->right, pred->bottom);
        }
    }

    nn_ctx->cme_pred_count = prev_count;

    /* save prediction result as previous for next frame */
    for (k = 0; k < prev_count; k++) {
        nn_ctx->cme_prev_boxes[k] = nn_ctx->cme_pred_boxes[k];
    }
    nn_ctx->cme_prev_count = prev_count;

    nn_ctx->cme_reset_cnt++;
    if (nn_ctx->cme_reset_cnt >= interval) {
        nn_ctx->cme_reset_cnt = 0;
    }

    return MPP_OK;
}

static MPP_RET cme_print_pred_info(SuperEncCtx *sec)
{
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    ObjDetRetLst *od_results = &sec->od_results;
    RK_S32 interval = sec->args->cme_pred_interval;
    RK_S32 obj_num = 0;
    int k;

    if (interval == 0) {
        obj_num = sec->od_results.count;
    } else {
        obj_num = (sec->frame_count % interval == 0) ? sec->od_results.count : nn_ctx->cme_pred_count;
    }

    mpp_log("[CME PRED INFO] frame=%d interval=%d detect=%d pred=%d prev=%d reset_cnt=%d\n",
            sec->frame_count, sec->args->cme_pred_interval,
            od_results->count, nn_ctx->cme_pred_count,
            nn_ctx->cme_prev_count, nn_ctx->cme_reset_cnt);

    for (k = 0; k < obj_num; k++) {
        image_rect_t *pred = &nn_ctx->cme_pred_boxes[k];
        mpp_log("[CME PRED INFO] pred[%d]=(%d,%d)-(%d,%d) %dx%d\n",
                k, pred->left, pred->top, pred->right, pred->bottom,
                pred->right - pred->left, pred->bottom - pred->top);
    }

    return MPP_OK;
}

static MPP_RET adjust_detect_rectangle_coordinate(SuperEncCtx *sec, ObjDetRetLst *result)
{
    int align = 16;
    int width = MPP_ALIGN(sec->args->width, align);
    int height = MPP_ALIGN(sec->args->height, align);

    for (int k = 0; k < result->count; k++) {
        result->results[k].box.left = result->results[k].box.left / align * align;
        result->results[k].box.top = result->results[k].box.top / align * align;
        result->results[k].box.right = MPP_MIN(MPP_ALIGN(result->results[k].box.right, align), width - 1);
        result->results[k].box.bottom = MPP_MIN(MPP_ALIGN(result->results[k].box.bottom, align), height - 1);
    }

    return MPP_OK;
}

/* return 1 if detect object, 0 if no detect object
 * interval==0: always use detection results (pure detect mode)
 * interval!=0 && frame%interval==0 (reset frame): use detection results
 * interval!=0 && frame%interval!=0: use prediction results
 */
static int get_dectect_flag(RknnCtx *nn_ctx, ObjDetRetLst *result,
                            int pos_x, int pos_y, int cme_pred_interval, int frame_count)
{
    int x, y, w, h;
    int flag = 0;
    int use_pred = (cme_pred_interval != 0 && frame_count % cme_pred_interval != 0);

    if (use_pred) {
        /* prediction mode: use pred_boxes for range check, no class_id filter */
        for (int k = 0; k < nn_ctx->cme_pred_count; k++) {
            x = nn_ctx->cme_pred_boxes[k].left;
            y = nn_ctx->cme_pred_boxes[k].top;
            w = nn_ctx->cme_pred_boxes[k].right  - nn_ctx->cme_pred_boxes[k].left;
            h = nn_ctx->cme_pred_boxes[k].bottom - nn_ctx->cme_pred_boxes[k].top;

            if (pos_x >= x && pos_x < x + w && pos_y >= y && pos_y < y + h) {
                flag = 1;
                break;
            }
        }
    } else {
        /* detection mode: use result for class_id filter and range check */
        for (int k = 0; k < result->count; k++) {
            int class_id = result->results[k].cls_id;
            if (class_id == LABEL_PERSON || class_id == LABEL_BICYCLE || class_id == LABEL_CAR ||
                class_id == LABEL_MOTORCYCLE || class_id == LABEL_BUS) {
                x = result->results[k].box.left;
                y = result->results[k].box.top;
                w = result->results[k].box.right  - result->results[k].box.left;
                h = result->results[k].box.bottom - result->results[k].box.top;

                if (pos_x >= x && pos_x < x + w && pos_y >= y && pos_y < y + h) {
                    flag = 1;
                    break;
                }
            }
        }
    }

    return flag;
}

static MPP_RET trans_rectangle_to_segmap(SuperEncCtx *sec, RknnCtx *nn_ctx, ObjDetRetLst *od_results,
                                         ObjMapRetLst *object_results, uint8_t ctu_size,
                                         int frame_count, int cme_pred_interval)
{
    int i, j;
    int h, w, block_num, pos_idx;
    int blk_pos_x, blk_pos_y, blk_end_x, blk_end_y;
    int pic_width = sec->args->width;
    int pic_height = sec->args->height;
    uint8_t *object_map = object_results->object_seg_map;
    uint8_t *seg_mask = od_results->results_seg[0].seg_mask;
    int b16_num = MPP_ALIGN(pic_width, ctu_size) / 16 * MPP_ALIGN(pic_height, ctu_size) / 16;
    int fg_b16_num = 0;
    int bg_fg_flag = 0; /* 0 - background, 1 - foreground */

    block_num = 0;
    if (od_results->count >= 1 || (nn_ctx->cme_pred_count > 0)) {
        object_results->found_objects = 1;

        for (h = 0; h < pic_height; h += ctu_size) {
            for (w = 0; w < pic_width; w += ctu_size) {
                for (i = 0; i < ctu_size / 16; i++) {
                    for (j = 0; j < ctu_size / 16; j++) {
                        blk_pos_x = w + j * 16;
                        blk_pos_y = h + i * 16;

                        bg_fg_flag = get_dectect_flag(nn_ctx, od_results, blk_pos_x, blk_pos_y,
                                                      cme_pred_interval, frame_count);
                        object_map[block_num] = (blk_pos_x > pic_width || blk_pos_y > pic_height) ? 0 : bg_fg_flag;
                        fg_b16_num += (object_map[block_num] >= 1);
                        if (bg_fg_flag || (block_num == b16_num - 1))
                            FPRINT(sec->fp_nn_out, "frame %d blk_idx %d (%d, %d) object_map %d\n",
                                   frame_count, block_num, blk_pos_x, blk_pos_y, bg_fg_flag);
                        block_num++;
                    }
                }
            }
        }
    } else if (sec->fp_nn_out) {
        /* for nn result debug */
        FPRINT(sec->fp_nn_out, "frame %d blk_idx %d (0, 0) object_map 0\n", frame_count, b16_num - 1);
    }

    object_results->fg_area = fg_b16_num * 100 / b16_num;

    return MPP_OK;
}

static MPP_RET fscanf_nn_result(RknnCtx *nn_ctx, ObjMapRetLst *result,
                                int ctu_size, int cur_frm_num)
{
    FILE *fp_in = nn_ctx->fp_segmap_external;
    uint8_t *object_map = result->object_seg_map;
    int w_aligned = MPP_ALIGN(nn_ctx->input_image_width, ctu_size);
    int h_aligned = MPP_ALIGN(nn_ctx->input_image_height, ctu_size);
    int b16_num = w_aligned / 16 * h_aligned / 16;
    int frm_num, blk_idx, pos_x, pos_y, object_map_val;
    int b16_cnt = 0;

    if (fp_in == NULL) {
        mpp_err("fp segmap external is NULL\n");
        return MPP_NOK;
    }

    memset(object_map, 0, b16_num);

    for (b16_cnt = 0; b16_cnt < b16_num; b16_cnt++) {
        /* input is raster scan b16x16 */
        if (fscanf(fp_in, "frame %d blk_idx %d (%d, %d) object_map %d\n",
                    &frm_num, &blk_idx, &pos_x, &pos_y, &object_map_val) == 5) {
            if (blk_idx < b16_num) {
                int zz_idx = blk_idx;;
                if (ctu_size == 32) {
                    int b32_x = pos_x / 32;
                    int b32_y = pos_y / 32;
                    int b32_idx = b32_y * (w_aligned / 32) + b32_x;
                    zz_idx = b32_idx * 4 + (pos_y % 32) / 16 * 2 + (pos_x % 32) / 16;
                } else if (ctu_size == 64) {
                    //TODO: support 64x64(2026.02.05)
                }

                /* 5 == vehicle; 8 == person */
                object_map[zz_idx] = (object_map_val == 5 || object_map_val == 8) ? 1 : 0;
            } else {
                mpp_err("frame %d blk_idx %d out of range\n", frm_num, blk_idx);
                // return MPP_NOK;
            }
        }

        if ((pos_x + 16 >= w_aligned) && (pos_y + 16 >= h_aligned)) {
            mpp_log("frame %d fscanf %d blk16\n", cur_frm_num, b16_cnt + 1);
            break;
        }
    }

    return MPP_OK;
}

#ifdef RKNN_V3
MPP_RET super_enc_rknn_init(SuperEncCtx *sec)
{
    MPP_RET ret = MPP_OK;
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    Rknn3Ctx *nn3_ctx = &sec->rknn3_ctx;
    ObjMapRetLst *om_results = &sec->om_results;
    image_buffer_t *image = &sec->src_image;

    if (sec->args->rknn_ver == 3 || sec->args->soc_id == SOC_RK3572) {
        nn3_ctx->sec = (void *)sec;
        nn3_ctx->scene_mode = sec->args->yolo_scene_mode;
    } else {
        nn_ctx->run_type = sec->args->run_type;
        nn_ctx->scene_mode = sec->args->yolo_scene_mode;
        nn_ctx->fp_segmap_external = sec->fp_nn_in;
        nn_ctx->fp_segmap = sec->fp_nn_out;
        nn_ctx->segmap_calc_en = !sec->args->rect_to_segmap_en;
        nn_ctx->show_time_lvl = sec->args->show_time;
    }

    if (sec->args->run_type != RUN_YUV_EXTERNAL_NPU_MPP) {
        if (sec->args->model_path && sec->args->weight) {
            if (init_yolov6_model_rknn3(&sec->rknn3_ctx) != MPP_OK) {
                mpp_err_f("init yolov6 seg model failed\n");
                return MPP_NOK;
            }
        } else {
            mpp_err_f("rknn3 model and weight is required!\n");
            return MPP_NOK;
        }
    }

    if (sec->args->run_type != RUN_JPEG_RKNN &&
        sec->args->run_type != RUN_JPEG_RKNN_MPP) {
        /* if input file is jpeg, we can not know width and height, so we set it to 0 */
        nn_ctx->pre_alloc_mask = 1;
        nn_ctx->input_image_width = sec->args->width;
        nn_ctx->input_image_height = sec->args->height;
    }

    om_results->found_objects = 0;
    if (sec->args->run_type != RUN_JPEG_RKNN && sec->args->run_type != RUN_JPEG_RKNN_MPP) {
        RK_S32 w = MPP_ALIGN(sec->args->width, 64);
        RK_S32 h = MPP_ALIGN(sec->args->height, 64);
        om_results->object_seg_map = (uint8_t *)calloc(1, w * h); //TODO: one byte for blk16(2025.02.26)
        if (!om_results->object_seg_map) {
            mpp_err_f("malloc object_seg_map(%dx%d) failed\n", w, h);
            return MPP_NOK;
        }
    }

    ret = (MPP_RET)init_postproc_yolov6_rknn3(nn3_ctx);
    if (ret != MPP_OK) {
        mpp_err_f("init post process failed\n");
        return ret;
    }

    return ret;
}

MPP_RET super_enc_rknn_release(SuperEncCtx *sec)
{
    rknn_output *outputs = sec->outputs;
    ObjMapRetLst *om_results = &sec->om_results;

    for (int i = 0; i < SEG_OUT_CHN_NUM; i++)
        SE_FREE(outputs[i].buf);

    SE_FREE(om_results->object_seg_map);

    if (sec->args->run_type == RUN_JPEG_RKNN || sec->args->run_type == RUN_JPEG_RKNN_MPP)
        SE_FREE(sec->src_image.virt_addr);

    release_yolov6_model_rknn3(&sec->rknn3_ctx);

    deinit_postproc_yolov6_rknn3(&sec->rknn3_ctx);

    return MPP_OK;
}

MPP_RET super_enc_rknn_process(SuperEncCtx *sec)
{
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    Rknn3Ctx *nn3_ctx = &sec->rknn3_ctx;
    rknn_output *outputs = sec->outputs;
    image_buffer_t *image = &sec->src_image;
    RunType run_type = sec->args->run_type;
    MPP_RET ret = MPP_OK;
    RK_S32 ctu_size = 16;

    memset(image, 0, sizeof(image_buffer_t));

    if (run_type == RUN_YUV_EXTERNAL_NPU_MPP) {
        mpp_log("frame %d get detect result from external npu\n", sec->frame_count);
    } else if (run_type == RUN_JPEG_RKNN || run_type == RUN_JPEG_RKNN_MPP) {
        ret = (MPP_RET)read_image(sec->args->file_input, image);
        if (ret != MPP_OK) {
            mpp_err_f("read image failed\n");
            return ret;
        }

        sec->om_results.object_seg_map = (uint8_t *)calloc(1, image->height * image->width);
        if (!sec->om_results.object_seg_map) { //TODO: one byte for blk16(2025.02.26)
            mpp_err_f("malloc object_seg_map(%dx%d) failed\n", image->width, image->height);
            return MPP_NOK;
        }
    } else {
        /* input yuv data */
        image->width = sec->args->width;
        image->height = sec->args->height;
        image->format = (sec->args->format == MPP_FMT_YUV420P) ? IMAGE_FORMAT_YUV420P :
                        IMAGE_FORMAT_YUV420SP_NV12; //TODO: support other format(2025.02.24)
        image->virt_addr = sec->src_buf;
        image->size = sec->args->width * sec->args->height * 3 / 2;
    }

    ctu_size = (sec->args->type == MPP_VIDEO_CodingAVC) ? 16 : 32;
    if (run_type == RUN_YUV_EXTERNAL_NPU_MPP) {
        ret = fscanf_nn_result(nn_ctx, &sec->om_results, ctu_size, sec->frame_count);
        if (ret != MPP_OK) {
            mpp_err_f("fscanf nn result failed\n");
            return ret;
        }
    } else {
        ret = (MPP_RET)inference_yolov6_model_rknn3(nn3_ctx, image, &sec->od_results);
        if (ret != MPP_OK) {
            mpp_err_f("inference yolov5 seg model failed\n");
            return ret;
        }

        // ret = (MPP_RET)post_process_image(nn_ctx, &sec->od_results, outputs);
        // if (ret != MPP_OK) {
        //     mpp_err_f("post process image failed\n");
        //     return ret;
        // }
    }

    if (sec->args->adjust_rect_coord)
        adjust_detect_rectangle_coordinate(sec, &sec->od_results);

    if (sec->fp_nn_dect_rect)
        dump_detect_rectangle(sec->fp_nn_dect_rect, &sec->od_results, sec->frame_count);

    /* compute CME MV averages for each detection box from CME data */
    // cme_calc_avg_mv_per_object(sec);

    /* predict current-frame boxes from previous-frame boxes + CME MV */
    // cme_predict_next_frame_boxes(sec);

    /* print prediction info for debugging */
    if (nn_ctx->show_pred_info)
        cme_print_pred_info(sec);

    ret = trans_rectangle_to_segmap(sec, nn_ctx, &sec->od_results,
                        &sec->om_results, ctu_size, sec->frame_count, sec->args->cme_pred_interval);
    if (ret != MPP_OK) {
        mpp_err_f("seg mask to class map failed\n");
        return ret;
    }

    if (run_type == RUN_YUV_EXTERNAL_NPU_MPP)
        mpp_log("frame %d get fg_area %d from external npu\n",
                sec->frame_count, sec->om_results.fg_area);
    else {
        RK_S32 interval = sec->args->cme_pred_interval;
        RK_S32 obj_num = 0;

        if (interval == 0) {
            obj_num = sec->od_results.count;
        } else {
            obj_num = (sec->frame_count % interval == 0) ? sec->od_results.count : nn_ctx->cme_pred_count;
        }

        mpp_log("frame %d rknn found %d objects fg_area %d\n",
            sec->frame_count, obj_num, sec->om_results.fg_area);
    }

    return ret;
}

#else /* RKNN_V2: RK3588, RK3576, RV1126B */

MPP_RET super_enc_rknn_init(SuperEncCtx *sec)
{
    MPP_RET ret = MPP_OK;
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    ObjMapRetLst *om_results = &sec->om_results;
    image_buffer_t *image = &sec->src_image;

    nn_ctx->sec = (void *)sec;
    nn_ctx->run_type = sec->args->run_type;
    nn_ctx->scene_mode = sec->args->yolo_scene_mode;
    nn_ctx->fp_segmap_external = sec->fp_nn_in;
    nn_ctx->fp_segmap = sec->fp_nn_out;
    nn_ctx->segmap_calc_en = !sec->args->rect_to_segmap_en;
    nn_ctx->show_time_lvl = sec->args->show_time;

    if (sec->args->run_type != RUN_YUV_EXTERNAL_NPU_MPP && sec->args->model_path) {
        ret = (MPP_RET)init_yolov5_seg_model(sec->args->model_path, nn_ctx);
        if (ret != MPP_OK) {
            mpp_err_f("init yolov5 seg model failed\n");
            return ret;
        }
    }

    if (sec->args->run_type != RUN_JPEG_RKNN &&
        sec->args->run_type != RUN_JPEG_RKNN_MPP) {
        /* if input file is jpeg, we can not know width and height, so we set it to 0 */
        nn_ctx->pre_alloc_mask = 1;
        nn_ctx->input_image_width = sec->args->width;
        nn_ctx->input_image_height = sec->args->height;
    }

    if (sec->args->run_type != RUN_YUV_EXTERNAL_NPU_MPP) {
        assert(nn_ctx->io_num.n_output == SEG_OUT_CHN_NUM);
        for (int i = 0; i < SEG_OUT_CHN_NUM; i++) {
            sec->outputs[i].index = i;
            sec->outputs[i].want_float = (!nn_ctx->is_quant);
            sec->outputs[i].size = SEG_OUT_BUF_SIZE;
            sec->outputs[i].is_prealloc = 1;

            if (sec->outputs[i].is_prealloc) {
                sec->outputs[i].buf = calloc(1, sec->outputs[i].size);
                if (!sec->outputs[i].buf) {
                    mpp_err_f("malloc output buf failed\n");
                    return MPP_NOK;
                }
            }
        }
    }

    om_results->found_objects = 0;
    if (sec->args->run_type != RUN_JPEG_RKNN && sec->args->run_type != RUN_JPEG_RKNN_MPP) {
        RK_S32 w = MPP_ALIGN(sec->args->width, 64);
        RK_S32 h = MPP_ALIGN(sec->args->height, 64);
        om_results->object_seg_map = (uint8_t *)calloc(1, w * h); //TODO: one byte for blk16(2025.02.26)
        if (!om_results->object_seg_map) {
            mpp_err_f("malloc object_seg_map(%dx%d) failed\n", w, h);
            return MPP_NOK;
        }
    }

    if (sec->soc_name == SOC_RK3588) {
        image_buffer_t *src = &sec->src_image;
        image_buffer_t *dst = &sec->dst_image;

        src->width = sec->args->width;
        src->height = sec->args->height;
        src->format = (sec->args->format == MPP_FMT_YUV420P) ? IMAGE_FORMAT_YUV420P :
                        IMAGE_FORMAT_YUV420SP_NV12; //TODO: support other format(2025.02.24)
        src->size = src->width * src->height * get_bpp_from_format(src->format);
        src->use_dma32_buf = 1;

        dst->width = nn_ctx->model_width;
        dst->height = nn_ctx->model_height;
        dst->format = IMAGE_FORMAT_RGB888;
        dst->size = dst->width * dst->height * get_bpp_from_format(dst->format);
        dst->use_dma32_buf = 1;

        /*
         * Allocate dma_buf within 4G from dma32_heap,
         * return dma_fd and virtual address.
         */
        if (dma_buf_alloc(DMA_HEAP_DMA32_UNCACHE_PATCH, src->size, &src->fd, (void **)&src->virt_addr)) {
            mpp_err_f("dma_buf_alloc src failed\n");
            return MPP_NOK;
        }
        if (dma_buf_alloc(DMA_HEAP_DMA32_UNCACHE_PATCH, dst->size, &dst->fd, (void **)&dst->virt_addr)) {
            mpp_err_f("dma_buf_alloc dst failed\n");
            return MPP_NOK;
        }
        mpp_log("virt addr %p %p\n", src->virt_addr, dst->virt_addr);
        nn_ctx->dst_img = dst;
    }

    ret = (MPP_RET)init_post_process(nn_ctx);
    if (ret != MPP_OK) {
        mpp_err_f("init post process failed\n");
        return ret;
    }

    return ret;
}

MPP_RET super_enc_rknn_release(SuperEncCtx *sec)
{
    rknn_output *outputs = sec->outputs;
    ObjMapRetLst *om_results = &sec->om_results;

    for (int i = 0; i < SEG_OUT_CHN_NUM; i++)
        SE_FREE(outputs[i].buf);

    SE_FREE(om_results->object_seg_map);

    if (sec->args->run_type == RUN_JPEG_RKNN || sec->args->run_type == RUN_JPEG_RKNN_MPP)
        SE_FREE(sec->src_image.virt_addr);

    release_yolov5_seg_model(&sec->rknn_ctx);

    if (sec->soc_name == SOC_RK3588) {
        image_buffer_t *src = &sec->src_image;
        image_buffer_t *dst = &sec->dst_image;
        dma_buf_free(src->size, &src->fd, src->virt_addr);
        dma_buf_free(dst->size, &dst->fd, dst->virt_addr);
    }

    deinit_post_process(&sec->rknn_ctx);

    return MPP_OK;
}

MPP_RET super_enc_rknn_process(SuperEncCtx *sec)
{
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    rknn_output *outputs = sec->outputs;
    image_buffer_t *image = &sec->src_image;
    RunType run_type = sec->args->run_type;
    MPP_RET ret = MPP_OK;
    RK_S32 ctu_size = 16;

    if (sec->soc_name != SOC_RK3588)
        memset(image, 0, sizeof(image_buffer_t));

    if (run_type == RUN_YUV_EXTERNAL_NPU_MPP) {
        mpp_log("frame %d get detect result from external npu\n", sec->frame_count);
    } else if (run_type == RUN_JPEG_RKNN || run_type == RUN_JPEG_RKNN_MPP) {
        ret = (MPP_RET)read_image(sec->args->file_input, image);
        if (ret != MPP_OK) {
            mpp_err_f("read image failed\n");
            return ret;
        }

        sec->om_results.object_seg_map = (uint8_t *)calloc(1, image->height * image->width);
        if (!sec->om_results.object_seg_map) { //TODO: one byte for blk16(2025.02.26)
            mpp_err_f("malloc object_seg_map(%dx%d) failed\n", image->width, image->height);
            return MPP_NOK;
        }
    } else {
        /* input yuv data */
        if (sec->soc_name == SOC_RK3588) {
            //TODO: copy size, support other format(20250409)
            memcpy(image->virt_addr, sec->src_buf, sec->args->width * sec->args->height * 3 / 2);
        } else {
            image->width = sec->args->width;
            image->height = sec->args->height;
            image->format = (sec->args->format == MPP_FMT_YUV420P) ? IMAGE_FORMAT_YUV420P :
                            IMAGE_FORMAT_YUV420SP_NV12; //TODO: support other format(2025.02.24)
            image->virt_addr = sec->src_buf;
            image->size = sec->args->width * sec->args->height * 3 / 2;
        }
    }

    ctu_size = (sec->args->type == MPP_VIDEO_CodingAVC) ? 16 :
               (sec->soc_name == SOC_RK3576) ? 32 : 64;
    if (run_type == RUN_YUV_EXTERNAL_NPU_MPP) {
        ret = fscanf_nn_result(nn_ctx, &sec->om_results, ctu_size, sec->frame_count);
        if (ret != MPP_OK) {
            mpp_err_f("fscanf nn result failed\n");
            return ret;
        }
    } else {
        ret = (MPP_RET)inference_yolov5_seg_model(nn_ctx, image, outputs);
        if (ret != MPP_OK) {
            mpp_err_f("inference yolov5 seg model failed\n");
            return ret;
        }

        ret = (MPP_RET)post_process_image(nn_ctx, &sec->od_results, outputs);
        if (ret != MPP_OK) {
            mpp_err_f("post process image failed\n");
            return ret;
        }
    }

    if (sec->args->adjust_rect_coord)
        adjust_detect_rectangle_coordinate(sec, &sec->od_results);

    if (sec->fp_nn_dect_rect)
        dump_detect_rectangle(sec->fp_nn_dect_rect, &sec->od_results, sec->frame_count);

    /* compute CME MV averages for each detection box from CME data */
    cme_calc_avg_mv_per_object(sec);

    /* predict current-frame boxes from previous-frame boxes + CME MV */
    cme_predict_next_frame_boxes(sec);

    /* print prediction info for debugging */
    if (nn_ctx->show_pred_info)
        cme_print_pred_info(sec);

    if (sec->args->rect_to_segmap_en)
        ret = trans_rectangle_to_segmap(sec, nn_ctx, &sec->od_results,
                        &sec->om_results, ctu_size, sec->frame_count, sec->args->cme_pred_interval);
    else
        ret = (MPP_RET)seg_mask_to_class_map(nn_ctx, &sec->od_results,
                        &sec->om_results, ctu_size, sec->frame_count);
    if (ret != MPP_OK) {
        mpp_err_f("seg mask to class map failed\n");
        return ret;
    }

    if (run_type == RUN_YUV_EXTERNAL_NPU_MPP)
        mpp_log("frame %d get fg_area %d from external npu\n",
                sec->frame_count, sec->om_results.fg_area);
    else {
        RK_S32 interval = sec->args->cme_pred_interval;
        RK_S32 obj_num = 0;

        if (interval == 0) {
            obj_num = sec->od_results.count;
        } else {
            obj_num = (sec->frame_count % interval == 0) ? sec->od_results.count : nn_ctx->cme_pred_count;
        }

        mpp_log("frame %d rknn found %d objects fg_area %d\n",
            sec->frame_count, obj_num, sec->om_results.fg_area);
    }

    return ret;
}
#endif

