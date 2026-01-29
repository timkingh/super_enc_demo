#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "rknn_api.h"
#include "postprocess.h"
#include "yolov5_seg.h"
#include "rknn_process.h"
#include "super_enc_common.h"
#include "mpp_log.h"
#include "mpp_common.h"
#include "assert.h"
#include "dma_alloc.hpp"

#define SEG_OUT_CHN_NUM        (7)  /* rknn yolov5 seg output channel number */
#define SEG_OUT_BUF_SIZE       (1632000)  /* rknn yolov5 seg output size */

static MPP_RET dump_detect_rectangle(RknnCtx *nn_ctx, object_detect_result_list *result, int frm_cnt)
{
    FILE *fp = nn_ctx->fp_rect;

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

static MPP_RET adjust_detect_rectangle_coordinate(RknnCtx *nn_ctx, object_detect_result_list *result)
{
    int align = 16;
    int width = MPP_ALIGN(nn_ctx->input_image_width, align);
    int height = MPP_ALIGN(nn_ctx->input_image_height, align);

    for (int k = 0; k < result->count; k++) {
        result->results[k].box.left = result->results[k].box.left / align * align;
        result->results[k].box.top = result->results[k].box.top / align * align;
        result->results[k].box.right = MPP_MIN(MPP_ALIGN(result->results[k].box.right, align), width - 1);
        result->results[k].box.bottom = MPP_MIN(MPP_ALIGN(result->results[k].box.bottom, align), height - 1);
    }

    return MPP_OK;
}

/* return 1 if detect object, 0 if no detect object */
static int get_dectect_flag(RknnCtx *nn_ctx, object_detect_result_list *result, int pos_x, int pos_y)
{
    int x, y, w, h;
    int flag = 0;

    for (int k = 0; k < result->count; k++) {
        int class_id = result->results[k].cls_id;
        if (class_id == LABEL_PERSON || class_id == LABEL_BICYCLE || class_id == LABEL_CAR ||
            class_id == LABEL_MOTORCYCLE || class_id == LABEL_BUS) {
            x = result->results[k].box.left;
            y = result->results[k].box.top;
            w = result->results[k].box.right - result->results[k].box.left;
            h = result->results[k].box.bottom - result->results[k].box.top;
            if (pos_x >= x && pos_x < x + w && pos_y >= y && pos_y < y + h) {
                flag = 1;
                break;
            }
        }
    }

    return flag;
}

static MPP_RET trans_rectangle_to_segmap(RknnCtx *nn_ctx, object_detect_result_list *od_results,
                                         object_map_result_list *object_results, uint8_t ctu_size, int frame_count)
{
    int i, j;
    int h, w, block_num, pos_idx;
    int blk_pos_x, blk_pos_y, blk_end_x, blk_end_y;
    int pic_width = nn_ctx->input_image_width;
    int pic_height = nn_ctx->input_image_height;
    uint8_t *object_map = object_results->object_seg_map;
    uint8_t *seg_mask = od_results->results_seg[0].seg_mask;
    int b16_num = MPP_ALIGN(pic_width, ctu_size) / 16 * MPP_ALIGN(pic_height, ctu_size) / 16;
    int fg_b16_num = 0;
    int bg_fg_flag = 0; /* 0 - background, 1 - foreground */

    block_num = 0;
    // if more than one object, we need to convert the object map
    if (od_results->count >= 1) {
        object_results->found_objects = 1;

        for (h = 0; h < pic_height; h += ctu_size) {
            for (w = 0; w < pic_width; w += ctu_size) {
                for (i = 0; i < ctu_size / 16; i++) {
                    for (j = 0; j < ctu_size / 16; j++) {
                        blk_pos_x = w + j * 16;
                        blk_pos_y = h + i * 16;

                        bg_fg_flag = get_dectect_flag(nn_ctx, od_results, blk_pos_x, blk_pos_y);
                        object_map[block_num] = (blk_pos_x > pic_width || blk_pos_y > pic_height) ? 0 : bg_fg_flag;
                        fg_b16_num += (object_map[block_num] >= 1);
                        if (bg_fg_flag || (block_num == b16_num - 1))
                            FPRINT(nn_ctx->fp_segmap, "frame %d blk_idx %d (%d, %d) object_map %d\n",
                                   frame_count, block_num, blk_pos_x, blk_pos_y, bg_fg_flag);
                        block_num++;
                    }
                }
            }
        }
    } else if (nn_ctx->fp_segmap) {
        /* for nn result debug */
        FPRINT(nn_ctx->fp_segmap, "frame %d blk_idx %d (0, 0) object_map 0\n", frame_count, b16_num - 1);
    }

    object_results->foreground_area = fg_b16_num * 100 / b16_num;

    return MPP_OK;
}

MPP_RET super_enc_rknn_init(SuperEncCtx *sec)
{
    MPP_RET ret = MPP_OK;
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    object_map_result_list *om_results = &sec->om_results;
    image_buffer_t *image = &sec->src_image;

    nn_ctx->run_type = sec->args->run_type;
    nn_ctx->scene_mode = sec->args->yolo_scene_mode;
    nn_ctx->fp_segmap = sec->fp_nn_out;
    nn_ctx->fp_rect = sec->fp_nn_dect_rect;
    nn_ctx->segmap_calc_en = !sec->args->rect_to_segmap_en;
    nn_ctx->show_time_lvl = sec->args->show_time;

    ret = (MPP_RET)init_yolov5_seg_model(sec->args->model_path, nn_ctx);
    if (ret != MPP_OK) {
        mpp_err_f("init yolov5 seg model failed\n");
        return ret;
    }

    if (sec->args->run_type != RUN_JPEG_RKNN &&
        sec->args->run_type != RUN_JPEG_RKNN_MPP) {
        /* if input file is jpeg, we can not know width and height, so we set it to 0 */
        nn_ctx->pre_alloc_mask = 1;
        nn_ctx->input_image_width = sec->args->width;
        nn_ctx->input_image_height = sec->args->height;
    }

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

MPP_RET super_enc_rknn_process(SuperEncCtx *sec)
{
    RknnCtx *nn_ctx = &sec->rknn_ctx;
    rknn_output *outputs = sec->outputs;
    image_buffer_t *image = &sec->src_image;
    MPP_RET ret = MPP_OK;
    RK_S32 ctu_size = 16;

    if (sec->soc_name != SOC_RK3588)
        memset(image, 0, sizeof(image_buffer_t));

    if (sec->args->run_type == RUN_JPEG_RKNN || sec->args->run_type == RUN_JPEG_RKNN_MPP) {
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

    if (sec->args->adjust_rect_coord)
        adjust_detect_rectangle_coordinate(nn_ctx, &sec->od_results);

    if (nn_ctx->fp_rect)
        dump_detect_rectangle(nn_ctx, &sec->od_results, sec->frame_count);

    ctu_size = (sec->args->type == MPP_VIDEO_CodingAVC) ? 16 :
               (sec->soc_name == SOC_RK3576) ? 32 : 64;

    if (sec->args->rect_to_segmap_en)
        ret = trans_rectangle_to_segmap(nn_ctx, &sec->od_results,
                        &sec->om_results, ctu_size, sec->frame_count);
    else
        ret = (MPP_RET)seg_mask_to_class_map(nn_ctx, &sec->od_results,
                        &sec->om_results, ctu_size, sec->frame_count);
    if (ret != MPP_OK) {
        mpp_err_f("seg mask to class map failed\n");
        return ret;
    }

    mpp_log("frame %d rknn found %d objects fg_area %d%\n",
            sec->frame_count, sec->od_results.count,
            sec->om_results.foreground_area);

    return ret;
}

MPP_RET super_enc_rknn_release(SuperEncCtx *sec)
{
    rknn_output *outputs = sec->outputs;
    object_map_result_list *om_results = &sec->om_results;

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