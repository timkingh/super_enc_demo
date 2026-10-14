#ifndef __SUPER_ENC_V3_TEST_H__
#define __SUPER_ENC_V3_TEST_H__

#include <stdio.h>
#include "postproc_yolov5_seg_rknn2.h"
#include "mpi_enc_utils.h"
#include "rknn3_common.h"
#include "super_enc_common.h"

typedef struct {
    MpiEncTestArgs *args;
    SocName soc_name;

    image_buffer_t src_image;
    image_buffer_t dst_image;
    RknnCtx rknn_ctx;
    rknn_output outputs[7];
    ObjMapRetLst om_results;
    ObjDetRetLst od_results;

#ifdef RKNN_V3
    /* only used for RK3572 */
    Rknn3Ctx rknn3_ctx;
#endif

    void *mpp_ctx;
    uint8_t *src_buf; /* input yuv buffer */

    int frame_count;
    int loop_times;
    uint8_t get_sps_pps;

    FILE *fp_input;
    FILE *fp_output;
    FILE *fp_nn_in; /* external NPU input */
    FILE *fp_nn_out; /* used for ff_tools input */
    FILE *fp_nn_dect_rect;
    FILE *fp_cme_out; /* CME output */

    /* CME single-frame buffer, filled before each super_enc_rknn_process call */
    CmeFrameData cme_cur_frame;
    int cme_ctu_size;
    int cme_w_in_ctu;
    int cme_h_in_ctu;
} SuperEncCtx;

#endif