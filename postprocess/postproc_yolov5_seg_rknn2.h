#ifndef __POSTPROC_YOLOV5_SEG_RKNN2_H__
#define __POSTPROC_YOLOV5_SEG_RKNN2_H__

#include <stdint.h>
#include "common.h"
#include "rknn2_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int init_post_process(RknnCtx *app_ctx);
void deinit_post_process(RknnCtx *app_ctx);
int calc_instance_mask(RknnCtx *nn_ctx, rknn_output *outputs,
                       letterbox_t *letter_box, float conf_threshold, float nms_threshold,
                       ObjDetRetLst *od_results);
int trans_detect_result(RknnCtx *nn_ctx, letterbox_t *letter_box, ObjDetRetLst *od_results);

#ifdef __cplusplus
}
#endif

#endif //__POSTPROC_YOLOV5_SEG_RKNN2_H__
