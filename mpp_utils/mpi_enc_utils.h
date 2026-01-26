/*
 * Copyright 2015 Rockchip Electronics Co. LTD
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __MPI_ENC_UTILS_H__
#define __MPI_ENC_UTILS_H__

#include <stdio.h>

#include "rk_venc_cmd.h"
#include "iniparser.h"

typedef void* FpsCalc;

typedef enum {
    RUN_JPEG_RKNN = 0, /* run rknn only, input is jpeg */
    RUN_JPEG_RKNN_MPP, /* run rknn + mpp, input is jpeg */
    RUN_YUV_RKNN, /* run rknn only, input is yuv */
    RUN_YUV_MPP, /* run mpp only, input is yuv */
    RUN_YUV_RKNN_MPP, /* run rknn + mpp, input is yuv */
    RUn_TYPE_BUTT
} RunType;

typedef struct MpiEncTestArgs_t {
    char                *file_input;
    char                *file_output;
    char                *file_cfg;
    char                *model_path;
    char                *nn_out;
    char                *nn_dect_rect; /* detect rectangles output file */
    dictionary          *cfg_ini;

    MppCodingType       type;
    MppCodingType       type_src;       /* for file source input */
    MppFrameFormat      format;
    RK_S32              frame_num;
    RK_S32              loop_cnt;
    RK_S32              nthreads;

    RK_S32              width;
    RK_S32              height;
    RK_S32              hor_stride;
    RK_S32              ver_stride;

    /* -rc */
    RK_S32              rc_mode;
    /* -se_mode */
    RK_S32              se_mode;

    /* -bps */
    RK_S32              bps_target;
    RK_S32              bps_max;
    RK_S32              bps_min;

    /* -fps */
    RK_S32              fps_in_flex;
    RK_S32              fps_in_num;
    RK_S32              fps_in_den;
    RK_S32              fps_out_flex;
    RK_S32              fps_out_num;
    RK_S32              fps_out_den;

    /* -qc */
    RK_S32              qp_init;
    RK_S32              qp_min;
    RK_S32              qp_max;
    RK_S32              qp_min_i;
    RK_S32              qp_max_i;

    /* -fqc */
    RK_S32              fqp_min_i;
    RK_S32              fqp_min_p;
    RK_S32              fqp_max_i;
    RK_S32              fqp_max_p;

    /* -fqc_v3 */
    RK_S32              min_fg_fqp;
    RK_S32              max_fg_fqp;
    RK_S32              min_bg_fqp;
    RK_S32              max_bg_fqp;

    /* -dqp_v3 */
    RK_S32              fg_delta_qp_i;
    RK_S32              fg_delta_qp_p;
    RK_S32              bg_delta_qp_i;
    RK_S32              bg_delta_qp_p;

    /* -bmap_qc */
    RK_S32              bmap_qpmin_i;
    RK_S32              bmap_qpmin_p;
    RK_S32              bmap_qpmax_i;
    RK_S32              bmap_qpmax_p;

    /* -g gop mode */
    RK_S32              gop_mode;
    RK_S32              gop_len;
    RK_S32              vi_len;

    /* -sm scene_mode */
    RK_S32              scene_mode;
    RK_S32              rc_container;
    RK_S32              bias_i;
    RK_S32              bias_p;
    RK_S32              lambda_idx_p;
    RK_S32              lambda_idx_i;

    /* -qpdd cu_qp_delta_depth */
    RK_S32              cu_qp_delta_depth;
    RK_S32              anti_flicker_str;
    RK_S32              atr_str_i;
    RK_S32              atr_str_p;
    RK_S32              atl_str;
    RK_S32              sao_str_i;
    RK_S32              sao_str_p;

    /* -dbe deblur enable flag
     * -dbs deblur strength
     */
    RK_S32              deblur_en;
    RK_S32              deblur_str;

    /* -lcl light change level */
    RK_S32              lgt_chg_lvl;

    /* -v q runtime log disable flag */
    RK_U32              quiet;
    /* -v f runtime fps log flag */
    RK_U32              trace_fps;
    FpsCalc             fps;
    RK_U32              psnr_en;
    RK_U32              ssim_en;

    /* 0 - jpeg + rknn ; 1 - jpeg + rknn + mpp
     * 2 - yuv + rknn; 3 - yuv + rknn + mpp
     */
    RunType             run_type;

    /* 0 - RK3576; 1 - RK3588 */
    RK_S32              soc_id;
    /* 0 - person + book, for dep2
     * 1 - person + bicycle + car + motorcycle + bus, for dep1
     */
    RK_S32              yolo_scene_mode;

    RK_U32              kmpp_en; /* for rv1126b */

    RK_U32              adjust_rect_coord;
    RK_U32              rect_to_segmap_en; /* rectangle to segment map flag */
    RK_U32              smart_en; /* 0 - disable, 1 - smart v1, 3 - smart v3 */
} MpiEncTestArgs;

#ifdef __cplusplus
extern "C" {
#endif

RK_S32 mpi_enc_width_default_stride(RK_S32 width, MppFrameFormat fmt);

MpiEncTestArgs *mpi_enc_test_cmd_get(void);
MPP_RET mpi_enc_test_cmd_update_by_args(MpiEncTestArgs* cmd, int argc, char **argv);
MPP_RET mpi_enc_test_cmd_put(MpiEncTestArgs* cmd);

MPP_RET mpi_enc_test_cmd_show_opt(MpiEncTestArgs* cmd);

#ifdef __cplusplus
}
#endif

#endif /*__MPI_ENC_UTILS_H__*/
