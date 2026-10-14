#ifdef RKNN_V3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "yolov6_rknn3.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "mpp_log.h"
#include "mpp_err.h"
#include "mpp_time.h"
#include "mpp_common.h"
#include "super_enc_v3_test.h"
#include "postproc_yolov6_rknn3.h"
#include "rknn3_api.h"

#define Y6R3_DBG_FUNCTION             (0x00000001)
#define Y6R3_DBG_MODEL                (0x00000002)
#define Y6R3_DBG_TIME                 (0x00000004)

#define y6r3_log(cond, fmt, ...)   do { if (cond) mpp_log(fmt, ## __VA_ARGS__); } while (0)
#define y6r3_dbg(flag, fmt, ...)   y6r3_log((y6r3_debug & flag), fmt, ## __VA_ARGS__)
#define y6r3_dbg_func(fmt, ...)    y6r3_dbg(Y6R3_DBG_FUNCTION, "%s "fmt, __FUNCTION__, ## __VA_ARGS__)
#define y6r3_dbg_model(fmt, ...)   y6r3_dbg(Y6R3_DBG_MODEL, "%s "fmt, __FUNCTION__, ## __VA_ARGS__)
#define y6r3_dbg_time(fmt, ...)    y6r3_dbg(Y6R3_DBG_TIME, "%s "fmt, __FUNCTION__, ## __VA_ARGS__)

RK_S32 y6r3_debug = 4;

static void dump_tensor_attr(rknn3_tensor_attr *attr)
{
    y6r3_dbg_func("enter\n");
    y6r3_dbg_model("index %d name %s n_dims %d dims=[%d, %d, %d, %d] "
        "n_stride %d [%d, %d, %d, %d] \n"
        "n_elems %d aligned_size %d fmt %s type %s qnt_type %s core_id %d\n\n",
        attr->index, attr->name, attr->n_dims,
        attr->shape[0], attr->shape[1], attr->shape[2], attr->shape[3],
        attr->n_stride, attr->stride[0], attr->stride[1], attr->stride[2], attr->stride[3],
        attr->n_elems, attr->aligned_size,
        rknn3_get_layout_string(attr->layout),
        rknn3_get_type_string(attr->dtype),
        rknn3_get_qnt_type_string(attr->qnt_type),
        attr->core_id);
    y6r3_dbg_func("exit\n");
}

MPP_RET init_yolov6_model_rknn3(Rknn3Ctx* nn3_ctx)
{
    SuperEncCtx *sec = (SuperEncCtx *)nn3_ctx->sec;
    rknn3_tensor_attr *in_attr, *out_attr;
    rknn3_tensor_attr *p = NULL;
    rknn3_input_output_num io_num;
    rknn3_context ctx = 0;
    rknn3_config config;
    int ret;

    y6r3_dbg_func("enter\n");
    memset(&config, 0, sizeof(config));
    config.run_core_mask = 1; /* ?? */

    ret = rknn3_init(&ctx, NULL);
    if (ret < 0) {
        mpp_err_f("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, sec->args->model_path, sec->args->weight);
    if (ret < 0) {
        mpp_err_f("load_model fail!\n");
        return MPP_NOK;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        mpp_err_f("rknn_model_init failed! ret=%d\n", ret);
        return MPP_NOK;
    }

    if (/*postprocess_plugin_path != NULL*/0) {
        nn3_ctx->use_postprocess_plugin = true;

        ret = rknn3_register_custom_ops_plugins(ctx, /*postprocess_plugin_path*/ NULL, /*strlen(postprocess_plugin_path)*/ 0);
        if (ret != RKNN3_SUCCESS)
        {
          mpp_err_f("rknn3_register_custom_ops_plugins failed! ret=%d\n", ret);
          rknn3_destroy(ctx);
          return MPP_NOK;
        }
        mpp_log("rknn3_register_custom_ops_plugins success\n");
    }
    else {
        nn3_ctx->use_postprocess_plugin = false;
    }

    rknn3_query_cmd cmd = nn3_ctx->use_postprocess_plugin ? RKNN3_QUERY_POSTPROCESS_IN_OUT_NUM : RKNN3_QUERY_IN_OUT_NUM;
    ret = rknn3_query(ctx, cmd, &io_num, sizeof(io_num));
    if (ret < 0) {
        mpp_err_f("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    y6r3_dbg_model("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    in_attr = (rknn3_tensor_attr *)malloc(io_num.n_input * sizeof(rknn3_tensor_attr));
    if (in_attr == NULL) {
        mpp_err_f("malloc input_attr fail!\n");
        return MPP_NOK;
    }

    out_attr = (rknn3_tensor_attr *)malloc(io_num.n_output * sizeof(rknn3_tensor_attr));
    if (out_attr == NULL) {
        mpp_err_f("malloc output_attr fail!\n");
        return MPP_NOK;
    }

    {
        mpp_log("input tensors:\n");
        for (int i = 0; i < io_num.n_input; i++) {
            p = in_attr + i;
            p->index = i;

            ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, p, sizeof(rknn3_tensor_attr));
            if (ret < 0) {
                mpp_err_f("RKNN3_QUERY_INPUT_ATTR fail! ret=%d\n", ret);
                return ret;
            }
            dump_tensor_attr(p);
        }

        mpp_log("output  tensors:\n");
        for (int i = 0; i < io_num.n_output; i++) {
            rknn3_query_cmd cmd = nn3_ctx->use_postprocess_plugin ? RKNN3_QUERY_POSTPROCESS_OUTPUT_ATTR :
                                                                    RKNN3_QUERY_OUTPUT_ATTR;
            p = out_attr + i;
            p->index = i;

            ret = rknn3_query(ctx, cmd, p, sizeof(rknn3_tensor_attr));
            if (ret < 0) {
                mpp_err_f("RKNN3_QUERY_OUTPUT_ATTR fail! ret=%d\n", ret);
                return ret;
            }
            dump_tensor_attr(p);
        }
    }

    // Set to context
    nn3_ctx->inputs = (rknn3_tensor *)malloc(io_num.n_input * sizeof(rknn3_tensor));
    if (nn3_ctx->inputs == NULL) {
        mpp_err_f("malloc inputs fail!\n");
        return MPP_NOK;
    }

    nn3_ctx->outputs = (rknn3_tensor *)malloc(io_num.n_output * sizeof(rknn3_tensor));
    if (nn3_ctx->outputs == NULL) {
        mpp_err_f("malloc outputs fail!\n");
        return MPP_NOK;
    }
    nn3_ctx->rknn_ctx = ctx;
    nn3_ctx->io_num = io_num;

    for (int i = 0; i < nn3_ctx->io_num.n_input; i++) {
        p = in_attr + i;
        nn3_ctx->inputs[i].mem  = rknn3_create_mem(ctx, p->aligned_size, p->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);

        nn3_ctx->inputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        if (nn3_ctx->inputs[i].attr == NULL) {
            mpp_err_f("malloc input_attr fail!\n");
            return MPP_NOK;
        }
        memcpy(nn3_ctx->inputs[i].attr, p, sizeof(rknn3_tensor_attr));
    }

    for (int i = 0; i < nn3_ctx->io_num.n_output; i++) {
        p = out_attr + i;
        nn3_ctx->outputs[i].mem  = rknn3_create_mem(ctx, p->aligned_size, p->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);

        nn3_ctx->outputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        if (nn3_ctx->outputs[i].attr == NULL) {
            mpp_err_f("malloc output_attr fail!\n");
            return MPP_NOK;
        }
        memcpy(nn3_ctx->outputs[i].attr, p, sizeof(rknn3_tensor_attr));
    }

    if (out_attr[0].qnt_type == RKNN3_TENSOR_PER_LAYER_ASYMMETRIC &&
        out_attr[0].dtype == RKNN3_TENSOR_INT8) {
        nn3_ctx->is_quant = true;
    } else {
        nn3_ctx->is_quant = false;
    }

    if (in_attr[0].layout == RKNN3_TENSOR_NHWC) {
        y6r3_dbg_model("model is NHWC input fmt\n");
        nn3_ctx->model_channel = in_attr[0].shape[3];
        nn3_ctx->model_height = in_attr[0].shape[1];
        nn3_ctx->model_width = in_attr[0].shape[2];
    } else {
        y6r3_dbg_model("model is NCHW input fmt\n");
        nn3_ctx->model_height = in_attr[0].shape[1];
        nn3_ctx->model_width = in_attr[0].shape[2];
        nn3_ctx->model_channel = in_attr[0].shape[3];
    }
    y6r3_dbg_model("model input height=%d, width=%d, channel=%d\n",
            nn3_ctx->model_height, nn3_ctx->model_width, nn3_ctx->model_channel);

    free(in_attr);
    free(out_attr);

    y6r3_dbg_func("exit\n");
    return 0;
}

MPP_RET release_yolov6_model_rknn3(Rknn3Ctx* nn3_ctx)
{
    y6r3_dbg_func("enter\n");
    for (int i = 0; i < nn3_ctx->io_num.n_input; i++) {
        if (nn3_ctx->inputs[i].mem) {
            rknn3_destroy_mem(nn3_ctx->rknn_ctx, nn3_ctx->inputs[i].mem);
        }

        if (nn3_ctx->inputs[i].attr != NULL) {
            free(nn3_ctx->inputs[i].attr);
            nn3_ctx->inputs[i].attr = NULL;
        }
    }

    for (int i = 0; i < nn3_ctx->io_num.n_output; i++) {
        if (nn3_ctx->outputs[i].mem) {
            rknn3_destroy_mem(nn3_ctx->rknn_ctx, nn3_ctx->outputs[i].mem);
        }

        if (nn3_ctx->outputs[i].attr != NULL) {
            free(nn3_ctx->outputs[i].attr);
            nn3_ctx->outputs[i].attr = NULL;
        }
    }

    if (nn3_ctx->rknn_ctx != 0) {
        rknn3_destroy(nn3_ctx->rknn_ctx);
        nn3_ctx->rknn_ctx = 0;
    }

    y6r3_dbg_func("exit\n");
    return 0;
}

MPP_RET inference_yolov6_model_rknn3(Rknn3Ctx *nn3_ctx, image_buffer_t *img, ObjDetRetLst *od_results)
{
    rknn3_query_cmd query_cmd = nn3_ctx->use_postprocess_plugin ? RKNN3_QUERY_POSTPROCESS_OUTPUT_ATTR :
                                                                  RKNN3_QUERY_OUTPUT_ATTR;
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    double preprocess_time, inference_time, postprocess_time;
    struct timeval start_time, end_time;
    rknn3_tensor_attr output_attrs;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    int bg_color = 114;
    int ret = MPP_OK;

    y6r3_dbg_func("enter\n");
    if ((!nn3_ctx) || (!img) || (!od_results)) {
        mpp_err_f("Invalid input parameters!\n");
        return MPP_NOK;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));

    // Pre Process
    gettimeofday(&start_time, NULL);
    dst_img.width = nn3_ctx->model_width;
    dst_img.height = nn3_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = nn3_ctx->inputs[0].mem->virt_addr;
    assert(nn3_ctx->inputs[0].attr->aligned_size >= dst_img.size);

    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0) {
        mpp_err_f("convert_image_with_letterbox fail! ret=%d\n", ret);
        return MPP_NOK;
    }

    gettimeofday(&end_time, NULL);
    preprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    y6r3_dbg_time("Pre-process time: %.2f ms\n", preprocess_time);

    // Sync inputs
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < nn3_ctx->io_num.n_input; i++) {
        ret = rknn3_mem_sync(nn3_ctx->rknn_ctx, nn3_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0) {
            mpp_err_f("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            goto out;
        }
    }

    ret = rknn3_run(nn3_ctx->rknn_ctx, nn3_ctx->inputs, nn3_ctx->io_num.n_input,
                    nn3_ctx->outputs, nn3_ctx->io_num.n_output);
    if (ret < 0) {
        mpp_err_f("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Sync outputs
    for (int i = 0; i < nn3_ctx->io_num.n_output; i++) {
        ret = rknn3_mem_sync(nn3_ctx->rknn_ctx, nn3_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0) {
            mpp_err_f("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            goto out;
        }
    }
    gettimeofday(&end_time, NULL);
    inference_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    y6r3_dbg_time("Inference time: %.2f ms\n", inference_time);

    gettimeofday(&start_time, NULL);
    output_attrs.index = 0;
    ret = rknn3_query(nn3_ctx->rknn_ctx, query_cmd, &output_attrs, sizeof(rknn3_tensor_attr));
    if (ret < 0) {
        mpp_err_f("rknn_query fail! ret=%d\n", ret);
        goto out;
    }

    if(nn3_ctx->io_num.n_output == 1 && output_attrs.n_dims == 3 && output_attrs.shape[2] == 6) {
        post_process_after_exYoloPostProcess(nn3_ctx, nn3_ctx->outputs, output_attrs, &letter_box, od_results);
    } else {
        postproc_yolov6_rknn3(nn3_ctx, nn3_ctx->outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
    }

    gettimeofday(&end_time, NULL);
    postprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    y6r3_dbg_time("Post-process time: %.2f ms\n", postprocess_time);
    y6r3_dbg_time("Total time: %.2f ms\n", preprocess_time + inference_time + postprocess_time);

out:
    y6r3_dbg_func("exit\n");

    return ret;
}
#endif