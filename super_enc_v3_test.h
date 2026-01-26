#ifndef __SUPER_ENC_V3_TEST_H__
#define __SUPER_ENC_V3_TEST_H__

#include <stdio.h>
#include "postprocess.h"
#include "mpi_enc_utils.h"
#include "super_enc_common.h"

typedef struct {
    MpiEncTestArgs *args;
    SocName soc_name;

    image_buffer_t src_image;
    image_buffer_t dst_image;
    RknnCtx rknn_ctx;
    rknn_output outputs[7];
    object_map_result_list om_results;
    object_detect_result_list od_results;

    void *mpp_ctx;
    uint8_t *src_buf; /* input yuv buffer */

    int frame_count;
    uint8_t get_sps_pps;

    FILE *fp_input;
    FILE *fp_output;
    FILE *fp_nn_out; /* used for ff_tools input */
    FILE *fp_nn_dect_rect;
} SuperEncCtx;

#endif