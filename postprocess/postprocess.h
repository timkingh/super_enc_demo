#ifndef _RKNN_YOLOV5_SEG_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV5_SEG_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "common.h"
#include "image_utils.h"

// #ifndef RV1126B_ARMHF
// #define RV1126B_ARMHF
// #endif

#ifdef RV1126B_ARMHF
#define USE_FP_MASK_MAP /* set 0 to reduce cpu usage */
#else
#define USE_FP_RESIZE /* RK3588 and RK3576. (20250624) */
#endif

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

#define PROTO_CHANNEL 32
#define PROTO_HEIGHT 160
#define PROTO_WEIGHT 160

#define VPU_MIN(a, b) ((a) < (b) ? (a) : (b))

typedef enum {
    LABEL_PERSON = 0,
    LABEL_BICYCLE,
    LABEL_CAR,
    LABEL_MOTORCYCLE,
    LABEL_AIRPLANE,
    LABEL_BUS,
    LABEL_TRAIN,
    LABEL_TRUCK,
    LABEL_BOAT,
    LABEL_TRAFFIC_LIGHT,
    LABEL_FIRE_HYDRANT,
    LABEL_STOP_SIGN,
    LABEL_PARKING_METER,
    LABEL_BENCH,
    LABEL_BIRD,
    LABEL_CAT,
    LABEL_DOG,
    LABEL_HORSE,
    LABEL_SHEEP,
    LABEL_COW,
    LABEL_ELEPHANT,
    LABEL_BEAR,
    LABEL_ZEBRA,
    LABEL_GIRAFFE,
    LABEL_BACKPACK,
    LABEL_UMBRELLA,
    LABEL_HANDBAG,
    LABEL_TIE,
    LABEL_SUITCASE,
    LABEL_FRISBEE,
    LABEL_SKIS,
    LABEL_SNOWBOARD,
    LABEL_SPORTS_BALL,
    LABEL_KITE,
    LABEL_BASEBALL_BAT,
    LABEL_BASEBALL_GLOVE,
    LABEL_SKATEBOARD,
    LABEL_SURFBOARD,
    LABEL_TENNIS_RACKET,
    LABEL_BOTTLE,
    LABEL_WINE_GLASS,
    LABEL_CUP,
    LABEL_FORK,
    LABEL_KNIFE,
    LABEL_SPOON,
    LABEL_BOWL,
    LABEL_BANANA,
    LABEL_APPLE,
    LABEL_SANDWICH,
    LABEL_ORANGE,
    LABEL_BROCCOLI,
    LABEL_CARROT,
    LABEL_HOT_DOG,
    LABEL_PIZZA,
    LABEL_DONUT,
    LABEL_CAKE,
    LABEL_CHAIR,
    LABEL_COUCH,
    LABEL_POTTED_PLANT,
    LABEL_BED,
    LABEL_DINING_TABLE,
    LABEL_TOILET,
    LABEL_TV,
    LABEL_LAPTOP,
    LABEL_MOUSE,
    LABEL_REMOTE,
    LABEL_KEYBOARD,
    LABEL_CELL_PHONE,
    LABEL_MICROWAVE,
    LABEL_OVEN,
    LABEL_TOASTER,
    LABEL_SINK,
    LABEL_REFRIGERATOR,
    LABEL_BOOK,
    LABEL_CLOCK,
    LABEL_VASE,
    LABEL_SCISSORS,
    LABEL_TEDDY_BEAR,
    LABEL_HAIR_DRYER,
    LABEL_TOOTHBRUSH,
    LABEL_BUTT,
} LabelName;

typedef struct {
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
    rknn_matmul_shape shapes[OBJ_NUMB_MAX_SIZE];
    rknn_matmul_io_attr io_attr[OBJ_NUMB_MAX_SIZE];
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
    float filterBoxes_by_nms[OBJ_NUMB_MAX_SIZE * 4];
    int cls_id[OBJ_NUMB_MAX_SIZE];

    int scene_mode; /* pick different class for different scene */
    int segmap_calc_en; /* 0 or 1, enable segmap calculation or not */
    FILE *fp_segmap;
    FILE *fp_rect;
} RknnCtx;

typedef struct
{
    int found_objects;
    int foreground_area; /* [0, 100] */
    uint8_t *object_seg_map;
} object_map_result_list;

typedef struct
{
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct
{
    uint8_t *seg_mask;
} object_segment_result;

typedef struct
{
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
    object_segment_result results_seg[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

#ifdef __cplusplus
extern "C" {
#endif

int init_post_process(RknnCtx *app_ctx);
void deinit_post_process(RknnCtx *app_ctx);
int calc_instance_mask(RknnCtx *nn_ctx, rknn_output *outputs,
    letterbox_t *letter_box, float conf_threshold, float nms_threshold,
    object_detect_result_list *od_results);
    int trans_detect_result(RknnCtx *nn_ctx, letterbox_t *letter_box, object_detect_result_list *od_results);

#ifdef __cplusplus
}
#endif

#endif //_RKNN_YOLOV5_SEG_DEMO_POSTPROCESS_H_
