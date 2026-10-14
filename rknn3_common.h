#ifndef __RKNN3_COMMON_H__
#define __RKNN3_COMMON_H__

#ifdef RKNN_V3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknn3_api.h"

typedef struct {
    void *sec; /* SuperEncCtx */
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;

    bool use_postprocess_plugin;

    int scene_mode; /* pick different class for different scene */
} Rknn3Ctx;

#endif /* RKNN_V3 */

#endif /* __RKNN3_COMMON_H__ */