#ifndef __POSTPROC_COMMON_H__
#define __POSTPROC_COMMON_H__

#include <stdint.h>
#include "nn_utils/common.h"

#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define OBJ_NUM_MAX 128 /* max number of objects in a frame */
#define OBJ_CLASS_NUM 80 /* COCO dataset has 80 classes */
#define LABEL_NAME_TXT_PATH "./model/coco_80_labels_list.txt"

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
    image_rect_t box;
    float prop;
    int cls_id;
} ObjDetRet;

typedef struct {
    uint8_t *seg_mask;
} ObjSegRet;

typedef struct {
    int id;
    int count;
    ObjDetRet results[OBJ_NUM_MAX];
    ObjSegRet results_seg[OBJ_NUM_MAX];
} ObjDetRetLst;

typedef struct {
    int found_objects;
    int fg_area; /* [0, 100] */
    uint8_t *object_seg_map;
} ObjMapRetLst;

extern LabelName label_names[2][5];
extern char *labels[OBJ_CLASS_NUM];

#ifdef __cplusplus
extern "C" {
#endif

char *coco_cls_to_name(int cls_id);
int loadLabelName(const char *locationFilename, char *label[]);

#ifdef __cplusplus
}
#endif

#endif