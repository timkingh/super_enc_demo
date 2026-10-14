#ifndef __POSTPROC_YOLOV6_RKNN3_H__
#define __POSTPROC_YOLOV6_RKNN3_H__

#ifdef RKNN_V3

#include "rknn3_common.h"
#include "common.h"
#include "image_utils.h"
#include "postproc_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int init_postproc_yolov6_rknn3(Rknn3Ctx *nn3_ctx);
void deinit_postproc_yolov6_rknn3(Rknn3Ctx *nn3_ctx);
int postproc_yolov6_rknn3(Rknn3Ctx *nn3_ctx, void *outputs, letterbox_t *letter_box,
                          float conf_threshold, float nms_threshold,
                          ObjDetRetLst *od_results);
int post_process_after_exYoloPostProcess(Rknn3Ctx *nn3_ctx, void *outputs,
                                         rknn3_tensor_attr output_attrs,
                                         letterbox_t *letter_box,
                                         ObjDetRetLst *od_results);

#ifdef __cplusplus
}
#endif

#endif

#endif