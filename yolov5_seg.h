#ifndef __ROCKIVA_COMMON_H__
#define __ROCKIVA_COMMON_H__

#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "rknn_api.h"
#include "postprocess.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ROCKIVA_RET_SUCCESS = 0,                 /* 成功 */
    ROCKIVA_RET_FAIL = -1,                   /* 失败 */
    ROCKIVA_RET_NULL_PTR = -2,               /* 空指针 */
    ROCKIVA_RET_INVALID_HANDLE = -3,         /* 无效句柄 */
    ROCKIVA_RET_LICENSE_ERROR = -4,          /* License错误 */
    ROCKIVA_RET_UNSUPPORTED = -5,            /* 不支持 */
    ROCKIVA_RET_STREAM_SWITCH = -6,          /* 码流切换 */
    ROCKIVA_RET_BUFFER_FULL = -7,            /* 缓存区域满 */
} RKYOLORetCode;

/**
 * @brief 初始化
 *
 * @param model_path [IN] rknn模型路径
 * @param nn_ctx [IN] rknn输入参数
 * @return RKYOLORetCode
 */
RKYOLORetCode init_yolov5_seg_model(const char *model_path, RknnCtx *nn_ctx);

/**
 * @brief 运行模型推理及结果处理
 *
 * @param nn_ctx [IN] rknn输入参数
 * @param img [IN] 输入图像RGB
 * @return RKYOLORetCode
 */
RKYOLORetCode inference_yolov5_seg_model(RknnCtx *nn_ctx, image_buffer_t *img, rknn_output outputs[]);

/**
 * @brief 模型输出结果处理及buffer释放
 *
 * @param nn_ctx [IN] rknn输入参数
 * @param od_results [IN] 检测结果的结构体
 * @return RKYOLORetCode
 */
RKYOLORetCode post_process_image(RknnCtx *nn_ctx, object_detect_result_list *od_results, rknn_output outputs[]);

/**
 * @brief 对seg_mask进行处理映射成对应的object_map
 *
 * @param nn_ctx [IN] rknn输入参数
 * @param od_results [IN] 检测结果的结构体
 * @param object_results [IN] class_map的结构体
 * @param ctu_size [IN] 16/32/64
 * @return RKYOLORetCode
 */
RKYOLORetCode seg_mask_to_class_map(RknnCtx *nn_ctx, object_detect_result_list *od_results,
                                    object_map_result_list *object_results, uint8_t ctu_size, int frame_count);

/**
 * @brief 释放资源
 *
 * @param nn_ctx [in] rknn输入参数
 * @return RKYOLORetCode
 */
RKYOLORetCode release_yolov5_seg_model(RknnCtx *nn_ctx);

#ifdef __cplusplus
}
#endif

#endif /* end of #ifndef __ROCKIVA_DET_API_H__ */
