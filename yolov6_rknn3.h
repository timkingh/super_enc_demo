#ifndef __YOLOV6_RKNN3_H__
#define __YOLOV6_RKNN3_H__

#ifdef RKNN_V3

#include "mpp_err.h"
#include "rknn3_common.h"
#include "postproc_common.h"

#ifdef __cplusplus
extern "C" {
#endif

MPP_RET init_yolov6_model_rknn3(Rknn3Ctx* nn3_ctx);

MPP_RET release_yolov6_model_rknn3(Rknn3Ctx* nn3_ctx);

MPP_RET inference_yolov6_model_rknn3(Rknn3Ctx* nn3_ctx, image_buffer_t* img, ObjDetRetLst* od_results);

#ifdef __cplusplus
}
#endif

#endif

#endif