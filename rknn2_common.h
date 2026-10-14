#ifndef __RKNN2_COMMON_H__
#define __RKNN2_COMMON_H__

#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "image_utils.h"
#include "postproc_common.h"

#ifdef RV1126B_ARMHF
#define USE_FP_MASK_MAP /* set 0 to reduce cpu usage */
#else
#define USE_FP_RESIZE /* RK3588 and RK3576. (20250624) */
#endif

typedef struct {
    void *sec; /* SuperEncCtx */
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    letterbox_t letter_box;
    int model_channel;
    int model_width;
    int model_height;
    int input_image_width;
    int input_image_height;
    int8_t is_quant;
    image_buffer_t *dst_img;

    rknn_matmul_ctx matmul_ctx;
    rknn_matmul_shape shapes[OBJ_NUM_MAX];
    rknn_matmul_io_attr io_attr[OBJ_NUM_MAX];
    rknn_tensor_mem *tensor_a;
    rknn_tensor_mem *tensor_b;
    rknn_tensor_mem *tensor_c;

#if defined(USE_FP_RESIZE) && defined(USE_FP_MASK_MAP)
    float *matmul_out; /* C = A * B */
    float *seg_mask; /* 640x640 * N, resize from 160x160 */
#else
    uint8_t *matmul_out; /* C = A * B */
    uint8_t *seg_mask; /* 640x640 * N, resize from 160x160 */
#endif
    uint8_t *all_mask_in_one; /* 640x640, all object mask in one image */
    uint8_t *cropped_seg_mask; /* max is 640x640. 640x360 if input is 1920x1080 */
    uint8_t *real_seg_mask; /* width * height, real seg mask of org picture */
    uint8_t pre_alloc_mask; /* 0 or 1, pre allocate mask memory or not */

    float *proto; /* proto mask */
    uint16_t *vector_b; /* float32 to float16 */
    float filterBoxes_by_nms[OBJ_NUM_MAX * 4];
    int cls_id[OBJ_NUM_MAX];

    int run_type;
    int scene_mode; /* pick different class for different scene */
    int segmap_calc_en; /* 0 or 1, enable segmap calculation or not */
    int show_time_lvl;

    FILE *fp_segmap_external; /* external segmap file */
    FILE *fp_segmap;

    /* CME MV statistics per detection box */
    int cme_mv_count;
    int cme_mv_x[OBJ_NUM_MAX];
    int cme_mv_y[OBJ_NUM_MAX];

    /* CME predicted next-frame detection boxes */
    int cme_pred_count;
    image_rect_t cme_pred_boxes[OBJ_NUM_MAX];

    /* CME previous-frame detection boxes (for MV-based prediction) */
    int cme_prev_count;
    image_rect_t cme_prev_boxes[OBJ_NUM_MAX];
    int cme_reset_cnt; /* frames since last prediction reset */

    int show_pred_info;
} RknnCtx;


#endif /* __RKNN2_COMMON_H__ */
