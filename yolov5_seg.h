#ifndef __YOLOV5_SEG_H__
#define __YOLOV5_SEG_H__

#include "mpp_err.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "rknn_api.h"
#include "postproc_yolov5_seg_rknn2.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化
 *
 * @param model_path [IN] rknn模型路径
 * @param nn_ctx [IN] rknn输入参数
 * @return MPP_RET
 */
MPP_RET init_yolov5_seg_model(const char *model_path, RknnCtx *nn_ctx);

/**
 * @brief 运行模型推理及结果处理
 *
 * @param nn_ctx [IN] rknn输入参数
 * @param img [IN] 输入图像RGB
 * @return MPP_RET
 */
MPP_RET inference_yolov5_seg_model(RknnCtx *nn_ctx, image_buffer_t *img, rknn_output outputs[]);

/**
 * @brief 模型输出结果处理及buffer释放
 *
 * @param nn_ctx [IN] rknn输入参数
 * @param od_results [IN] 检测结果的结构体
 * @return MPP_RET
 */
MPP_RET post_process_image(RknnCtx *nn_ctx, ObjDetRetLst *od_results, rknn_output outputs[]);

/**
 * @brief 对seg_mask进行处理映射成对应的object_map
 *
 * @param nn_ctx [IN] rknn输入参数
 * @param od_results [IN] 检测结果的结构体
 * @param object_results [IN] class_map的结构体
 * @param ctu_size [IN] 16/32/64
 * @return MPP_RET
 */
MPP_RET seg_mask_to_class_map(RknnCtx *nn_ctx, ObjDetRetLst *od_results,
                              ObjMapRetLst *object_results, uint8_t ctu_size, int frame_count);

/**
 * @brief 释放资源
 *
 * @param nn_ctx [in] rknn输入参数
 * @return MPP_RET
 */
MPP_RET release_yolov5_seg_model(RknnCtx *nn_ctx);

#ifdef __cplusplus
}
#endif

#endif /* end of #ifndef __YOLOV5_SEG_H__ */
