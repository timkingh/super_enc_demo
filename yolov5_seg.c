#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "yolov5_seg.h"
#include "postprocess.h"
#include "mpp_log.h"
#include "mpp_err.h"
#include "mpp_time.h"
#include "mpp_common.h"
#include "super_enc_common.h"

#define DUMP_SEGMENT_MAP             (1)

#define SEG_DBG_FUNCTION             (0x00000001)
#define SEG_DBG_MODEL                (0x00000002)
#define SEG_DBG_TIME                 (0x00000004)

#define seg_log(cond, fmt, ...)   do { if (cond) mpp_log(fmt, ## __VA_ARGS__); } while (0)
#define seg_dbg(flag, fmt, ...)   seg_log((seg_debug & flag), fmt, ## __VA_ARGS__)
#define seg_dbg_func(fmt, ...)    seg_dbg(SEG_DBG_FUNCTION, "%s "fmt, __FUNCTION__, ## __VA_ARGS__)
#define seg_dbg_model(fmt, ...)   seg_dbg(SEG_DBG_MODEL, "%s "fmt, __FUNCTION__, ## __VA_ARGS__)
#define seg_dbg_time(fmt, ...)    seg_dbg(SEG_DBG_TIME, fmt, ## __VA_ARGS__)

static RK_S32 seg_debug = 0;

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    seg_dbg_model("index %d name %s n_dims %d dims=[%d, %d, %d, %d] "
        "n_elems %d size %d fmt %s type %s qnt_type %s zp %d scale %f\n",
        attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
        attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

RKYOLORetCode init_yolov5_seg_model(const char *model_path, RknnCtx *nn_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    seg_dbg_func("enter\n");

    if (nn_ctx->show_time_lvl >= 1)
        seg_debug |= SEG_DBG_TIME;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        mpp_err("load_model fail!\n");
        return ROCKIVA_RET_FAIL;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        mpp_err("rknn_init fail! ret=%d\n", ret);
        return ROCKIVA_RET_FAIL;
    }

    {
        rknn_sdk_version version;
        if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version)) == RKNN_SUCC) {
            mpp_log("rknn sdk api version: %s\n", version.api_version);
            mpp_log("rknn driver version: %s\n", version.drv_version);
        } else
            mpp_err("rknn query sdk version fail! ret=%d\n", ret);
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        mpp_err("rknn_query fail! ret=%d\n", ret);
        return ROCKIVA_RET_FAIL;
    }
    seg_dbg_model("input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    rknn_tensor_attr *attr = NULL;
    nn_ctx->io_num = io_num;
    nn_ctx->input_attrs = (rknn_tensor_attr *)calloc(1, io_num.n_input * sizeof(rknn_tensor_attr));
    if (!nn_ctx->input_attrs) {
        mpp_err("malloc input attrs fail!\n");
        return ROCKIVA_RET_FAIL;
    }

    for (int i = 0; i < io_num.n_input; i++) {
        attr = nn_ctx->input_attrs + i;
        attr->index = i;

        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, attr, sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            mpp_err("rknn_query fail! ret=%d\n", ret);
            return ROCKIVA_RET_FAIL;
        }
        dump_tensor_attr(attr);
    }

    attr = nn_ctx->input_attrs;
    if (attr->fmt == RKNN_TENSOR_NCHW) {
        nn_ctx->model_channel = attr->dims[1];
        nn_ctx->model_height = attr->dims[2];
        nn_ctx->model_width = attr->dims[3];
    } else {
        nn_ctx->model_height = attr->dims[1];
        nn_ctx->model_width = attr->dims[2];
        nn_ctx->model_channel = attr->dims[3];
    }
    seg_dbg_model("input height %d width %d channel %d\n",
           nn_ctx->model_height, nn_ctx->model_width, nn_ctx->model_channel);

    // Get Model Output Info
    nn_ctx->output_attrs = (rknn_tensor_attr *)calloc(1, io_num.n_output * sizeof(rknn_tensor_attr));
    if (!nn_ctx->output_attrs) {
        mpp_err("malloc output attrs fail!\n");
        return ROCKIVA_RET_FAIL;
    }

    for (int i = 0; i < io_num.n_output; i++) {
        attr = nn_ctx->output_attrs + i;
        attr->index = i;

        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, attr, sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            mpp_err("rknn_query fail! ret=%d\n", ret);
            return ROCKIVA_RET_FAIL;
        }
        dump_tensor_attr(attr);
    }

    // Set to context
    nn_ctx->rknn_ctx = ctx;
    attr = nn_ctx->output_attrs;
    nn_ctx->is_quant = (attr->qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                         attr->type != RKNN_TENSOR_FLOAT16);

    seg_dbg_func("leave\n");

    return ROCKIVA_RET_SUCCESS;
}

RKYOLORetCode inference_yolov5_seg_model(RknnCtx *nn_ctx, image_buffer_t *img, rknn_output outputs[])
{
    int ret;
    image_buffer_t dst_img;
    rknn_input *input = NULL;
    int bg_color = 114; // pad color for letterbox
    RK_S64 time_start, time_end;
    int use_dma32_buf = nn_ctx->dst_img && nn_ctx->dst_img->use_dma32_buf;

    seg_dbg_func("enter\n");

    memset(&nn_ctx->letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));

    input = calloc(1, sizeof(rknn_input) * nn_ctx->io_num.n_input);
    if (!input) {
        mpp_err_f("malloc input fail!\n");
        return ROCKIVA_RET_FAIL;
    }

    // Pre Process
    nn_ctx->input_image_width = img->width;
    nn_ctx->input_image_height = img->height;

    if (use_dma32_buf) {
        dst_img = *nn_ctx->dst_img;
    } else {
        dst_img.width = nn_ctx->model_width;
        dst_img.height = nn_ctx->model_height;
        dst_img.format = IMAGE_FORMAT_RGB888;
        dst_img.size = get_image_size(&dst_img);
        dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
        if (dst_img.virt_addr == NULL) {
            mpp_err_f("malloc buffer size:%d fail!\n", dst_img.size);
            return ROCKIVA_RET_FAIL;
        }
    }

    // letterbox
    time_start = mpp_time();
    ret = convert_image_with_letterbox(img, &dst_img, &nn_ctx->letter_box, bg_color);
    if (ret < 0) {
        mpp_err_f("convert_image_with_letterbox fail! ret=%d\n", ret);
        return ROCKIVA_RET_FAIL;
    }
    time_end = mpp_time();
    seg_dbg_time("convert_image_with_letterbox(RGA) time: %0.2f ms\n", (float)(time_end - time_start) / 1000);

    // Set Input Data
    input->index = 0;
    input->type = RKNN_TENSOR_UINT8;
    input->fmt = RKNN_TENSOR_NHWC;
    input->size = nn_ctx->model_width * nn_ctx->model_height * nn_ctx->model_channel;
    input->buf = dst_img.virt_addr;

    time_start = mpp_time();
    ret = rknn_inputs_set(nn_ctx->rknn_ctx, nn_ctx->io_num.n_input, input);
    if (ret < 0) {
        mpp_err_f("rknn_input_set fail! ret=%d\n", ret);
        return ROCKIVA_RET_FAIL;
    }
    time_end = mpp_time();
    seg_dbg_time("rknn_inputs_set time: %0.2f ms\n", (float)(time_end - time_start) / 1000);

    time_start = mpp_time();
    ret = rknn_run(nn_ctx->rknn_ctx, NULL);
    time_end = mpp_time();
    seg_dbg_time("rknn_run time: %0.2f ms\n", (float)(time_end - time_start) / 1000);

    if (!use_dma32_buf && dst_img.virt_addr != NULL) {
        free(dst_img.virt_addr);
    }

    time_start = mpp_time();
    ret = rknn_outputs_get(nn_ctx->rknn_ctx, nn_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        mpp_err_f("rknn_outputs_get fail! ret=%d\n", ret);
        return ROCKIVA_RET_FAIL;
    }
    time_end = mpp_time();
    seg_dbg_time("rknn_outputs_get time: %0.2f ms\n", (float)(time_end - time_start) / 1000);

    SE_FREE(input);
    seg_dbg_func("leave\n");

    return ROCKIVA_RET_SUCCESS;
}

RKYOLORetCode post_process_image(RknnCtx *nn_ctx, object_detect_result_list *od_results,
                                 rknn_output outputs[])
{
    int ret = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    RK_S64 time_start, time_end;

    seg_dbg_func("enter\n");

#if 0
    post_process(nn_ctx, outputs, &nn_ctx->letter_box, box_conf_threshold, nms_threshold, od_results);
#else
    time_start = mpp_time();
    calc_instance_mask(nn_ctx, outputs, &nn_ctx->letter_box, box_conf_threshold, nms_threshold, od_results);
    time_end = mpp_time();
    seg_dbg_time("calc_instance_mask(postprocess) time: %0.2f ms\n", (float)(time_end - time_start) / 1000);

    if (nn_ctx->segmap_calc_en) {
        time_start = mpp_time();
        trans_detect_result(nn_ctx, &nn_ctx->letter_box, od_results);
        time_end = mpp_time();
        seg_dbg_time("trans_detect_result(postprocess) time: %0.2f ms\n", (float)(time_end - time_start) / 1000);
    }

#endif
    seg_dbg_func("leave\n");

    return ROCKIVA_RET_SUCCESS;
}

static void get_blk_object(int blk_pos_x, int blk_pos_y, int pic_width,int pic_height,
                           uint8_t *seg_mask, uint8_t *object_map, int pos_in_16x16_blk)
{
    int roi_calc_list[6];
    int k, l, pos_idx, m;
    int blk_end_x, blk_end_y;

    if (blk_pos_x > pic_width || blk_pos_y > pic_height) {
        object_map[pos_in_16x16_blk] = 0; // 0 means background
        return;
    }

    // calculate the block end position
    blk_end_x = VPU_MIN(blk_pos_x + 15, pic_width - 1);
    blk_end_y = VPU_MIN(blk_pos_y + 15, pic_height - 1);

    memset(&roi_calc_list, 0, sizeof(int) * 6);
    // calculate the number of pixels (in a 16x16 block) in each category
    for (k = blk_pos_y; k <= blk_end_y; k += 2) {
        for (l = blk_pos_x; l <= blk_end_x; l += 2) {
            pos_idx = l + k * pic_width;
            if (seg_mask[pos_idx] == 0)
                roi_calc_list[0]++;
            else if (seg_mask[pos_idx] == 1)
                roi_calc_list[1]++;
            else if (seg_mask[pos_idx] == 2)
                roi_calc_list[2]++;
            else if (seg_mask[pos_idx] == 3)
                roi_calc_list[3]++;
            else if (seg_mask[pos_idx] == 4)
                roi_calc_list[4]++;
            else if (seg_mask[pos_idx] == 5)
                roi_calc_list[5]++;
        }
    }
    // default value is 6, which means this block has different object or at the boundary of the image
    object_map[pos_in_16x16_blk] = 6;
    // get the category with the most pixels
    for (m = 0; m < 6; m++) {
        if (roi_calc_list[m] > (blk_end_y - blk_pos_y + 1) * (blk_end_x - blk_pos_x + 1) / 4 * 8 / 10) {
            object_map[pos_in_16x16_blk] = m;
            break;
        }
    }
}

RKYOLORetCode seg_mask_to_class_map(RknnCtx *rknn_nn_ctx, object_detect_result_list *od_results,
                                    object_map_result_list *object_results, uint8_t ctu_size, int frame_count)
{
    int i, j, k, l, m;
    int h, w, block_num, pos_idx;
    int blk_pos_x, blk_pos_y, blk_end_x, blk_end_y;
    int pic_width = rknn_nn_ctx->input_image_width;
    int pic_height = rknn_nn_ctx->input_image_height;
    uint8_t *object_map = object_results->object_seg_map;
    uint8_t *seg_mask = od_results->results_seg[0].seg_mask;
    int b16_num = MPP_ALIGN(pic_width, ctu_size) / 16 * MPP_ALIGN(pic_height, ctu_size) / 16;
    int fg_b16_num = 0;
    RK_S64 time_start, time_end;

    seg_dbg_func("enter\n");
    time_start = mpp_time();

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
                        // calculate the number of pixels (in a 16x16 block) in each category
                        get_blk_object(blk_pos_x, blk_pos_y, pic_width, pic_height,
                                       seg_mask, object_map, block_num);
                        fg_b16_num += (object_map[block_num] >= 1);
                        if (object_map[block_num] || (block_num == b16_num - 1))
                            FPRINT(rknn_nn_ctx->fp_segmap, "frame %d blk_idx %d (%d, %d) object_map %d\n",
                                   frame_count, block_num, blk_pos_x, blk_pos_y, object_map[block_num]);
                        block_num++;
                    }
                }
            }
        }
    } else if (rknn_nn_ctx->fp_segmap) {
        /* for nn result debug */
        FPRINT(rknn_nn_ctx->fp_segmap, "frame %d blk_idx %d (0, 0) object_map 0\n", frame_count, b16_num - 1);
    }

    object_results->foreground_area = fg_b16_num * 100 / b16_num;

    if (seg_mask && !rknn_nn_ctx->pre_alloc_mask) {
        free(od_results->results_seg[0].seg_mask);
        od_results->results_seg[0].seg_mask = NULL;
    }

    time_end = mpp_time();
    seg_dbg_time("seg_mask_to_class_map(postprocess) time: %0.2f ms\n", (float)(time_end - time_start) / 1000);
    seg_dbg_func("leave\n");

    return ROCKIVA_RET_SUCCESS;
}

RKYOLORetCode release_yolov5_seg_model(RknnCtx *nn_ctx)
{
    seg_dbg_func("enter\n");

    SE_FREE(nn_ctx->input_attrs);
    SE_FREE(nn_ctx->output_attrs);

    if (nn_ctx->rknn_ctx != 0) {
        rknn_destroy(nn_ctx->rknn_ctx);
        nn_ctx->rknn_ctx = 0;
    }

    seg_dbg_func("leave\n");

    return ROCKIVA_RET_SUCCESS;
}