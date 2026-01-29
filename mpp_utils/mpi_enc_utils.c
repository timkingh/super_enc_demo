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

#define MODULE_TAG "mpi_enc_utils"

#include <string.h>

#include "mpp_mem.h"
#include "mpp_debug.h"
#include "mpp_buffer.h"

#include "rk_mpi.h"
#include "utils.h"
#include "mpp_common.h"

#include "mpp_opt.h"
#include "mpi_enc_utils.h"

#define MAX_FILE_NAME_LENGTH        256

RK_S32 mpi_enc_width_default_stride(RK_S32 width, MppFrameFormat fmt)
{
    RK_S32 stride = 0;

    switch (fmt & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV400 :
    case MPP_FMT_YUV420SP :
    case MPP_FMT_YUV420SP_VU : {
        stride = MPP_ALIGN(width, 8);
    } break;
    case MPP_FMT_YUV420P : {
        /* NOTE: 420P need to align to 16 so chroma can align to 8 */
        stride = MPP_ALIGN(width, 16);
    } break;
    case MPP_FMT_YUV422P:
    case MPP_FMT_YUV422SP:
    case MPP_FMT_YUV422SP_VU: {
        /* NOTE: 422 need to align to 8 so chroma can align to 16 */
        stride = MPP_ALIGN(width, 8);
    } break;
    case MPP_FMT_YUV444SP :
    case MPP_FMT_YUV444P : {
        stride = MPP_ALIGN(width, 8);
    } break;
    case MPP_FMT_RGB565:
    case MPP_FMT_BGR565:
    case MPP_FMT_RGB555:
    case MPP_FMT_BGR555:
    case MPP_FMT_RGB444:
    case MPP_FMT_BGR444:
    case MPP_FMT_YUV422_YUYV :
    case MPP_FMT_YUV422_YVYU :
    case MPP_FMT_YUV422_UYVY :
    case MPP_FMT_YUV422_VYUY : {
        /* NOTE: for vepu limitation */
        stride = MPP_ALIGN(width, 8) * 2;
    } break;
    case MPP_FMT_RGB888 :
    case MPP_FMT_BGR888 : {
        /* NOTE: for vepu limitation */
        stride = MPP_ALIGN(width, 8) * 3;
    } break;
    case MPP_FMT_RGB101010 :
    case MPP_FMT_BGR101010 :
    case MPP_FMT_ARGB8888 :
    case MPP_FMT_ABGR8888 :
    case MPP_FMT_BGRA8888 :
    case MPP_FMT_RGBA8888 : {
        /* NOTE: for vepu limitation */
        stride = MPP_ALIGN(width, 8) * 4;
    } break;
    default : {
        mpp_err_f("do not support type %d\n", fmt);
    } break;
    }

    return stride;
}

MpiEncTestArgs *mpi_enc_test_cmd_get(void)
{
    MpiEncTestArgs *args = mpp_calloc(MpiEncTestArgs, 1);

    if (args)
        args->nthreads = 1;

    return args;
}

RK_S32 mpi_enc_opt_rknn(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        size_t len = strnlen(next, MAX_FILE_NAME_LENGTH);
        if (len) {
            cmd->model_path = mpp_calloc(char, len + 1);
            strcpy(cmd->model_path, next);
        }

        return 1;
    }

    mpp_err("model file is invalid\n");
    return 0;
}

RK_S32 mpi_enc_opt_nn_out(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        size_t len = strnlen(next, MAX_FILE_NAME_LENGTH);
        if (len) {
            cmd->nn_out = mpp_calloc(char, len + 1);
            strcpy(cmd->nn_out, next);
        }

        return 1;
    }

    mpp_err("nn output file is invalid\n");
    return 0;
}

RK_S32 mpi_enc_opt_nn_dect_rect(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        size_t len = strnlen(next, MAX_FILE_NAME_LENGTH);
        if (len) {
            cmd->nn_dect_rect = mpp_calloc(char, len + 1);
            strcpy(cmd->nn_dect_rect, next);
        }

        return 1;
    }

    mpp_err("nn detection rectangle file is invalid\n");
    return 0;
}

RK_S32 mpi_enc_opt_i(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        size_t len = strnlen(next, MAX_FILE_NAME_LENGTH);
        if (len) {
            cmd->file_input = mpp_calloc(char, len + 1);
            strcpy(cmd->file_input, next);
            name_to_frame_format(cmd->file_input, &cmd->format);

            if (cmd->type_src == MPP_VIDEO_CodingUnused)
                name_to_coding_type(cmd->file_input, &cmd->type_src);
        }

        return 1;
    }

    mpp_err("input file is invalid\n");
    return 0;
}

RK_S32 mpi_enc_opt_o(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        size_t len = strnlen(next, MAX_FILE_NAME_LENGTH);
        if (len) {
            cmd->file_output = mpp_calloc(char, len + 1);
            strcpy(cmd->file_output, next);
            name_to_coding_type(cmd->file_output, &cmd->type);
        }

        return 1;
    }

    mpp_log("output file is invalid\n");
    return 0;
}

RK_S32 mpi_enc_opt_w(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->width = atoi(next);
        return 1;
    }

    mpp_err("invalid input width\n");
    return 0;
}

RK_S32 mpi_enc_opt_h(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->height = atoi(next);
        return 1;
    }

    mpp_err("invalid input height\n");
    return 0;
}

RK_S32 mpi_enc_opt_hstride(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->hor_stride = atoi(next);
        return 1;
    }

    mpp_err("invalid input horizontal stride\n");
    return 0;
}

RK_S32 mpi_enc_opt_vstride(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->ver_stride = atoi(next);
        return 1;
    }

    mpp_err("invalid input vertical stride\n");
    return 0;
}

RK_S32 mpi_enc_opt_f(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        long number = 0;
        MppFrameFormat format = MPP_FMT_BUTT;

        if (MPP_OK == str_to_frm_fmt(next, &number)) {
            format = (MppFrameFormat)number;

            if (MPP_FRAME_FMT_IS_BE(format) &&
                (MPP_FRAME_FMT_IS_YUV(format) || MPP_FRAME_FMT_IS_RGB(format))) {
                cmd->format = format;
                return 1;
            }

            mpp_err("invalid input format 0x%x\n", format);
        }
    }

    cmd->format = MPP_FMT_YUV420SP;
    return 0;
}

RK_S32 mpi_enc_opt_t(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    MppCodingType type = MPP_VIDEO_CodingUnused;

    if (next) {
        type = (MppCodingType)atoi(next);
        if (!mpp_check_support_format(MPP_CTX_ENC, type))
            cmd->type = type;
        return 1;
    }

    mpp_err("invalid input coding type %d\n", type);
    return 0;
}

RK_S32 mpi_enc_opt_tsrc(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    MppCodingType type = MPP_VIDEO_CodingUnused;

    if (next) {
        type = (MppCodingType)atoi(next);
        if (!mpp_check_support_format(MPP_CTX_DEC, type))
            cmd->type_src = type;
        return 1;
    }

    mpp_err("invalid input coding type %d\n", type);
    return 0;
}


RK_S32 mpi_enc_opt_n(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->frame_num = atoi(next);
        return 1;
    }

    mpp_err("invalid input max number of frames\n");
    return 0;
}

RK_S32 mpi_enc_opt_g(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;

    if (next) {
        cnt = sscanf(next, "%d:%d:%d",
                     &cmd->gop_mode, &cmd->gop_len, &cmd->vi_len);
        if (cnt)
            return 1;
    }

    mpp_err("invalid gop mode use -g gop_mode:gop_len:vi_len\n");
    return 0;
}

RK_S32 mpi_enc_opt_rc(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;

    if (next) {
        cnt = sscanf(next, "%d", &cmd->rc_mode);
        if (cnt)
            return 1;
    }

    mpp_err("invalid rate control usage -rc rc_mode\n");
    mpp_err("rc_mode 0:vbr 1:cbr 2:fixqp 3:avbr 4:smtv1 5:smtv3\n");
    return 0;
}

RK_S32 mpi_enc_opt_se(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;
    if (next) {
        cnt = sscanf(next, "%d", &cmd->se_mode);
        if (cnt)
            return 1;
    }
    mpp_err("invalid rate control usage -semode se_mode\n");
    mpp_err("smtv3 se_mode 0:balance 1:quality first 2:bitrate first\n");
    return 0;
}

RK_S32 mpi_enc_opt_bps(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;

    if (next) {
        cnt = sscanf(next, "%d:%d:%d",
                     &cmd->bps_target, &cmd->bps_min, &cmd->bps_max);
        if (cnt)
            return 1;
    }

    mpp_err("invalid bit rate usage -bps bps_target:bps_min:bps_max\n");
    return 0;
}

RK_S32 mpi_enc_opt_fps(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        RK_U32 num = sscanf(next, "%d:%d:%d/%d:%d:%d",
                            &cmd->fps_in_num, &cmd->fps_in_den, &cmd->fps_in_flex,
                            &cmd->fps_out_num, &cmd->fps_out_den, &cmd->fps_out_flex);
        switch (num) {
        case 1 : {
            cmd->fps_out_num = cmd->fps_in_num;
            cmd->fps_out_den = cmd->fps_in_den = 1;
            cmd->fps_out_flex = cmd->fps_in_flex = 0;
        } break;
        case 2 : {
            cmd->fps_out_num = cmd->fps_in_num;
            cmd->fps_out_den = cmd->fps_in_den;
            cmd->fps_out_flex = cmd->fps_in_flex = 0;
        } break;
        case 3 : {
            cmd->fps_out_num = cmd->fps_in_num;
            cmd->fps_out_den = cmd->fps_in_den;
            cmd->fps_out_flex = cmd->fps_in_flex;
        } break;
        case 4 : {
            cmd->fps_out_den = 1;
            cmd->fps_out_flex = 0;
        } break;
        case 5 : {
            cmd->fps_out_flex = 0;
        } break;
        case 6 : {
        } break;
        default : {
            mpp_err("invalid in/out frame rate,"
                    " use \"-fps numerator:denominator:flex\""
                    " for set the input to the same fps as the output, such as 50:1:1\n"
                    " or \"-fps numerator:denominator:flex/numerator:denominator:flex\""
                    " for set input and output separately, such as 40:1:1/30:1:0\n");
        } break;
        }

        return (num && num <= 6);
    }

    mpp_err("invalid output frame rate\n");
    return 0;
}

RK_S32 mpi_enc_opt_qc(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;

    if (next) {
        cnt = sscanf(next, "%d:%d:%d:%d:%d", &cmd->qp_init,
                     &cmd->qp_min, &cmd->qp_max, &cmd->qp_min_i, &cmd->qp_max_i);
        if (cnt)
            return 1;
    }

    mpp_err("invalid quality control usage -qc qp_init:min:max:min_i:max_i\n");
    return 0;
}

RK_S32 mpi_enc_opt_fqc(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;

    if (next) {
        cnt = sscanf(next, "%d:%d:%d:%d", &cmd->fqp_min_i, &cmd->fqp_max_i,
                     &cmd->fqp_min_p, &cmd->fqp_max_p);
        if (cnt)
            return 1;
    }

    mpp_err("invalid frame quality control usage -fqc min_i:max_i:min_p:max_p\n");
    return 0;
}

RK_S32 mpi_enc_opt_s(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    cmd->nthreads = -1;
    if (next) {
        cmd->nthreads = atoi(next);
        if (cmd->nthreads >= 1)
            return 1;
    }

    mpp_err("invalid nthreads %d\n", cmd->nthreads);
    cmd->nthreads = 1;
    return 0;
}

RK_S32 mpi_enc_opt_l(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->loop_cnt = atoi(next);
        return 1;
    }

    mpp_err("invalid loop count\n");
    return 0;
}

RK_S32 mpi_enc_opt_v(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        if (strstr(next, "q"))
            cmd->quiet = 1;
        if (strstr(next, "f"))
            cmd->trace_fps = 1;

        return 1;
    }

    return 0;
}

RK_S32 mpi_enc_opt_ini(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        size_t len = strnlen(next, MAX_FILE_NAME_LENGTH);
        if (len) {
            cmd->file_cfg = mpp_calloc(char, len + 1);
            strncpy(cmd->file_cfg, next, len);
            cmd->cfg_ini = iniparser_load(cmd->file_cfg);

            return 1;
        }
    }

    mpp_err("input ini file is invalid\n");
    return 0;
}

RK_S32 mpi_enc_opt_sm(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->scene_mode = atoi(next);
        return 1;
    }

    mpp_err("invalid scene mode\n");
    return 0;
}

RK_S32 mpi_enc_opt_qpdd(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->cu_qp_delta_depth = atoi(next);
        return 1;
    }

    mpp_err("invalid cu_qp_delta_depth\n");
    return 0;
}

RK_S32 mpi_enc_opt_dbe(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->deblur_en = atoi(next);
        return 1;
    }

    mpp_err("invalid deblur en\n");
    return 0;
}

RK_S32 mpi_enc_opt_dbs(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->deblur_str = atoi(next);
        return 1;
    }

    mpp_err("invalid deblur str\n");

    return 0;
}

RK_S32 mpi_enc_opt_help(void *ctx, const char *next)
{
    (void)ctx;
    (void)next;
    return -1;
}

RK_S32 mpi_enc_opt_atf(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->anti_flicker_str = atoi(next);
        return 1;
    }

    mpp_err("invalid cu_qp_delta_depth\n");
    return 0;
}

RK_S32 mpi_enc_opt_atl(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->atl_str = atoi(next);
        return 1;
    }

    mpp_err("invalid atl_str\n");
    return 0;
}

RK_S32 mpi_enc_opt_atr_i(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->atr_str_i = atoi(next);
        return 1;
    }

    mpp_err("invalid atr_str_i\n");
    return 0;
}

RK_S32 mpi_enc_opt_atr_p(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->atr_str_p = atoi(next);
        return 1;
    }

    mpp_err("invalid atr_str_p\n");
    return 0;
}

RK_S32 mpi_enc_opt_sao_i(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->sao_str_i = atoi(next);
        return 1;
    }

    mpp_err("invalid sao_str_i\n");
    return 0;
}

RK_S32 mpi_enc_opt_sao_p(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->sao_str_p = atoi(next);
        return 1;
    }

    mpp_err("invalid sao_str_p\n");
    return 0;
}

RK_S32 mpi_enc_opt_bc(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->rc_container = atoi(next);
        return 1;
    }

    mpp_err("invalid bitrate container\n");
    return 0;
}

RK_S32 mpi_enc_opt_bias_i(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->bias_i = atoi(next);
        return 1;
    }

    mpp_err("invalid bias i\n");
    return 0;
}

RK_S32 mpi_enc_opt_bias_p(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->bias_p = atoi(next);
        return 1;
    }

    mpp_err("invalid bias p\n");
    return 0;
}

RK_S32 mpi_enc_opt_lmd(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->lambda_idx_p = atoi(next);
        return 1;
    }

    mpp_err("invalid lambda idx\n");
    return 0;
}

RK_S32 mpi_enc_opt_lmdi(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->lambda_idx_i = atoi(next);
        return 1;
    }

    mpp_err("invalid intra lambda idx\n");
    return 0;
}

RK_S32 mpi_enc_opt_run_type(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->run_type = (RunType)atoi(next);
        return 1;
    }

    mpp_err("invalid run type\n");
    return 0;
}

RK_S32 mpi_enc_opt_soc_id(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->soc_id = atoi(next);
        return 1;
    }

    mpp_err_f("invalid SoC id\n");
    return 0;
}

RK_S32 mpi_enc_opt_segmap_calc_en(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        /* segmap calc enable means no need to convert rect to segmap */
        cmd->rect_to_segmap_en = !(atoi(next));
        return 1;
    }

    mpp_err("invalid segmap_calc_en\n");
    return 0;
}

RK_S32 mpi_enc_opt_smart_en(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->smart_en = atoi(next);
        return 1;
    }

    mpp_err("invalid smart_en\n");
    return 0;
}

RK_S32 mpi_enc_opt_fqc_v3(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;

    if (next) {
        cnt = sscanf(next, "%d:%d:%d:%d", &cmd->min_fg_fqp, &cmd->max_fg_fqp,
                     &cmd->min_bg_fqp, &cmd->max_bg_fqp);
        if (cnt == 4)
            return 1;
    }

    cmd->min_fg_fqp = 30;
    cmd->max_fg_fqp = 35;
    cmd->min_bg_fqp = 35;
    cmd->max_bg_fqp = 40;

    mpp_err("invalid frame quality control usage -fqc_v3 min_fg_fqp:max_fg_fqp:min_bg_fqp:max_bg_fqp\n");
    return 0;
}

RK_S32 mpi_enc_opt_dqp_v3(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;

    if (next) {
        cnt = sscanf(next, "%d:%d:%d:%d", &cmd->fg_delta_qp_i, &cmd->fg_delta_qp_p,
                     &cmd->bg_delta_qp_i, &cmd->bg_delta_qp_p);
        if (cnt == 4)
            return 1;
    }

    cmd->fg_delta_qp_i = 4;
    cmd->fg_delta_qp_p = 3;
    cmd->bg_delta_qp_i = -2;
    cmd->bg_delta_qp_p = -2;

    mpp_err("invalid frame quality control usage -dqp_v3 fg_delta_qp_i:fg_delta_qp_p:bg_delta_qp_i:bg_delta_qp_p\n");
    return 0;
}

RK_S32 mpi_enc_opt_bmap_qc(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;
    RK_S32 cnt = 0;

    if (next) {
        cnt = sscanf(next, "%d:%d:%d:%d", &cmd->bmap_qpmin_i, &cmd->bmap_qpmin_p,
                     &cmd->bmap_qpmax_i, &cmd->bmap_qpmax_p);
        if (cnt == 4)
            return 1;
    }

    cmd->bmap_qpmin_i = 20;
    cmd->bmap_qpmin_p = 20;
    cmd->bmap_qpmax_i = 51;
    cmd->bmap_qpmax_p = 51;

    mpp_err("invalid bmap qc usage -bmap_qc qpmin_i:qpmin_p:qpmax_i:qpmax_p\n");
    return 0;
}

RK_S32 mpi_enc_opt_show_time(void *ctx, const char *next)
{
    MpiEncTestArgs *cmd = (MpiEncTestArgs *)ctx;

    if (next) {
        cmd->show_time = atoi(next);
        return 1;
    }

    mpp_err("invalid smart_en\n");
    return 0;
}

static MppOptInfo enc_opts[] = {
    {"i",       "input_file",           "input frame file",                         mpi_enc_opt_i},
    {"o",       "output_file",          "output encoded bitstream file",            mpi_enc_opt_o},
    {"w",       "width",                "the width of input picture",               mpi_enc_opt_w},
    {"h",       "height",               "the height of input picture",              mpi_enc_opt_h},
    {"hstride", "hor_stride",           "the horizontal stride of input picture",   mpi_enc_opt_hstride},
    {"vstride", "ver_stride",           "the vertical stride of input picture",     mpi_enc_opt_vstride},
    {"f",       "format",               "the format of input picture",              mpi_enc_opt_f},
    {"t",       "type",                 "output stream coding type",                mpi_enc_opt_t},
    {"tsrc",    "source type",          "input file source coding type",            mpi_enc_opt_tsrc},
    {"n",       "max frame number",     "max encoding frame number",                mpi_enc_opt_n},
    {"g",       "gop reference mode",   "gop_mode:gop_len:vi_len",                  mpi_enc_opt_g},
    {"rc",      "rate control mode",    "rc_mode, 0:vbr 1:cbr 2:fixqp 3:avbr 4:smtv1 5:smtv3", mpi_enc_opt_rc},
    {"semode",  "smart v3 mode",        "se_mode, 0:balance 1:quality first 2:bitrate first",  mpi_enc_opt_se},
    {"bps",     "bps target:min:max",   "set tareget:min:max bps",                  mpi_enc_opt_bps},
    {"fps",     "in/output fps",        "set input and output frame rate",          mpi_enc_opt_fps},
    {"qc",      "quality control",      "set qp_init:min:max:min_i:max_i",          mpi_enc_opt_qc},
    {"bmap_qc", "quality control",      "set bmap_qpmin_i:bmap_qpmin_p:bmap_qpmax_i:bmap_qpmax_p", mpi_enc_opt_bmap_qc},
    {"dqp_v3",  "quality control",      "set dqp_v3 fg_delta_qp_i:fg_delta_qp_p:bg_delta_qp_i:bg_delta_qp_p", mpi_enc_opt_dqp_v3},
    {"fqc",     "frame quality control", "set fqp min_i:max_i:min_p:max_p",         mpi_enc_opt_fqc},
    {"fqc_v3",  "frame quality control", "set fqc_v3 min_fg_fqp:max_fg_fqp:min_bg_fqp:max_bg_fqp", mpi_enc_opt_fqc_v3},
    {"s",       "instance_nb",          "number of instances",                      mpi_enc_opt_s},
    {"v",       "trace option",         "q - quiet f - show fps",                   mpi_enc_opt_v},
    {"l",       "loop count",           "loop encoding times for each frame",       mpi_enc_opt_l},
    {"ini",     "ini file",             "encoder extra ini config file",            mpi_enc_opt_ini},
    {"sm",      "scene mode",           "scene_mode, 0:default 1:ipc",              mpi_enc_opt_sm},
    {"qpdd",    "cu_qp_delta_depth",    "cu_qp_delta_depth, 0:1:2",                 mpi_enc_opt_qpdd},
    {"dbe",     "deblur enable",        "deblur_en or qpmap_en, 0:close 1:open",           mpi_enc_opt_dbe},
    {"dbs",     "deblur strength",      "deblur_str 0~3: hw + sw scheme; 4~7: hw scheme",  mpi_enc_opt_dbs},
    {"atf",     "anti_flicker_str",     "anti_flicker_str, 0:off 1 2 3",            mpi_enc_opt_atf},
    {"atl",     "atl_str",              "atl_str, 0:off 1 open",                    mpi_enc_opt_atl},
    {"atr_i",   "atr_str_i",            "atr_str_i, 0:off 1 2 3",                   mpi_enc_opt_atr_i},
    {"atr_p",   "atr_str_p",            "atr_str_p, 0:off 1 2 3",                   mpi_enc_opt_atr_p},
    {"sao_i",   "sao_str_i",            "sao_str_i, 0:off 1 2 3",                   mpi_enc_opt_sao_i},
    {"sao_p",   "sao_str_p",            "sao_str_p, 0:off 1 2 3",                   mpi_enc_opt_sao_p},
    {"bc",      "bitrate container",    "rc_container, 0:off 1:weak 2:strong",      mpi_enc_opt_bc},
    {"ibias",   "bias i",               "bias_i",                                   mpi_enc_opt_bias_i},
    {"pbias",   "bias p",               "bias_p",                                   mpi_enc_opt_bias_p},
    {"lmd",     "lambda idx",           "lambda_idx_p 0~8",                         mpi_enc_opt_lmd},
    {"lmdi",    "lambda i idx",         "lambda_idx_i 0~8",                         mpi_enc_opt_lmdi},
    {"rknn",    "rknn file",            "rknn model file",                          mpi_enc_opt_rknn},
    {"nn_out",  "rknn output file",     "rknn output file",                         mpi_enc_opt_nn_out},
    {"nn_dect_rect",  "rknn rect output file", "rknn rect output file",             mpi_enc_opt_nn_dect_rect},
    {"rt",      "run type",  "run type: 0 jpeg rknn; 1 jpeg rknn mpp; 2 yuv rknn; 3 yuv mpp; 4 yuv rknn mpp",
                                                                                mpi_enc_opt_run_type},
    {"soc",     "SoC id",               "0 3576; 1 3588",                           mpi_enc_opt_soc_id},
    {"segmap_calc_en", "segmap_calc_en", "segmap_calc_en, 0:off 1:on",              mpi_enc_opt_segmap_calc_en},
    {"smart_en", "smart_en", "smart_en, 0:off 1:v1 3:v3",                           mpi_enc_opt_smart_en},
    {"show_time", "show_time", "show time, 0, 1, 2",                                mpi_enc_opt_show_time},
};

static RK_U32 enc_opt_cnt = MPP_ARRAY_ELEMS(enc_opts);

RK_S32 mpi_enc_show_help(const char *name)
{
    RK_U32 max_name = 1;
    RK_U32 max_full_name = 1;
    RK_U32 max_help = 1;
    char logs[256];
    RK_U32 len;
    RK_U32 i;

    mpp_log("usage: %s [options]\n", name);

    for (i = 0; i < enc_opt_cnt; i++) {
        MppOptInfo *opt = &enc_opts[i];

        if (opt->name) {
            len = strlen(opt->name);
            if (len > max_name)
                max_name = len;
        }

        if (opt->full_name) {
            len = strlen(opt->full_name);
            if (len > max_full_name)
                max_full_name = len;
        }

        if (opt->help) {
            len = strlen(opt->help);
            if (len > max_help)
                max_help = len;
        }
    }

    snprintf(logs, sizeof(logs) - 1, "-%%-%ds %%-%ds %%-%ds\n", max_name, max_full_name, max_help);

    for (i = 0; i < enc_opt_cnt; i++) {
        MppOptInfo *opt = &enc_opts[i];

        mpp_log(logs, opt->name, opt->full_name, opt->help);
    }
    mpp_log("\n");
    mpp_show_support_format();
    mpp_log("\n");
    mpp_show_color_format();

    return -1;
}

void show_enc_fps(RK_S64 total_time, RK_S64 total_count, RK_S64 last_time, RK_S64 last_count)
{
    float avg_fps = (float)total_count * 1000000 / total_time;
    float ins_fps = (float)last_count * 1000000 / last_time;

    mpp_log("encoded %10lld frame fps avg %7.2f ins %7.2f\n",
            total_count, avg_fps, ins_fps);
}

MPP_RET mpi_enc_test_cmd_update_by_args(MpiEncTestArgs* cmd, int argc, char **argv)
{
    MppOpt opts = NULL;
    RK_S32 ret = -1;
    RK_U32 i;

    if ((argc < 2) || NULL == cmd || NULL == argv)
        goto done;

    cmd->rc_mode = MPP_ENC_RC_MODE_BUTT;

    mpp_opt_init(&opts);
    /* should change node count when option increases */
    mpp_opt_setup(opts, cmd);

    for (i = 0; i < enc_opt_cnt; i++)
        mpp_opt_add(opts, &enc_opts[i]);

    /* mark option end */
    mpp_opt_add(opts, NULL);
    ret = mpp_opt_parse(opts, argc, argv);

    /* check essential parameter */
    if (cmd->run_type == RUN_JPEG_RKNN_MPP) {
        mpp_err("run_type %d not support yet\n", cmd->run_type);
        ret = MPP_NOK;
    }

    if ((cmd->type <= MPP_VIDEO_CodingAutoDetect) &&
        (cmd->run_type >= RUN_YUV_RKNN && cmd->run_type <= RUN_YUV_RKNN_MPP)) {
        mpp_err("invalid type %d\n", cmd->type);
        ret = MPP_NOK;
    }

    if (cmd->rc_mode == MPP_ENC_RC_MODE_BUTT)
        cmd->rc_mode = (cmd->type == MPP_VIDEO_CodingMJPEG) ?
                       MPP_ENC_RC_MODE_FIXQP : MPP_ENC_RC_MODE_VBR;

    if (!cmd->hor_stride)
        cmd->hor_stride = mpi_enc_width_default_stride(cmd->width, cmd->format);
    if (!cmd->ver_stride)
        cmd->ver_stride = MPP_ALIGN(cmd->height, 2);

    if (cmd->type_src == MPP_VIDEO_CodingUnused) {
        if (cmd->width <= 0 || cmd->height <= 0 ||
            cmd->hor_stride <= 0 || cmd->ver_stride <= 0) {
            mpp_err("invalid w:h [%d:%d] stride [%d:%d]\n",
                    cmd->width, cmd->height, cmd->hor_stride, cmd->ver_stride);
            ret = MPP_NOK;
        }
    }

    if (cmd->rc_mode == MPP_ENC_RC_MODE_FIXQP) {
        if (!cmd->qp_init) {
            if (cmd->type == MPP_VIDEO_CodingAVC ||
                cmd->type == MPP_VIDEO_CodingHEVC)
                cmd->qp_init = 26;
        }
    }

    if (cmd->trace_fps) {
        fps_calc_init(&cmd->fps);
        mpp_assert(cmd->fps);
        fps_calc_set_cb(cmd->fps, show_enc_fps);
    }

done:
    if (opts) {
        mpp_opt_deinit(opts);
        opts = NULL;
    }
    if (ret)
        mpi_enc_show_help(argv[0]);

    return ret;
}

MPP_RET mpi_enc_test_cmd_put(MpiEncTestArgs* cmd)
{
    if (NULL == cmd)
        return MPP_OK;

    if (cmd->cfg_ini) {
        iniparser_freedict(cmd->cfg_ini);
        cmd->cfg_ini = NULL;
    }

    if (cmd->fps) {
        fps_calc_deinit(cmd->fps);
        cmd->fps = NULL;
    }

    MPP_FREE(cmd->file_input);
    MPP_FREE(cmd->file_output);
    MPP_FREE(cmd->file_cfg);
    MPP_FREE(cmd->model_path);
    MPP_FREE(cmd->nn_out);
    MPP_FREE(cmd->nn_dect_rect);
    MPP_FREE(cmd);

    return MPP_OK;
}

MPP_RET mpi_enc_gen_ref_cfg(MppEncRefCfg ref, RK_S32 gop_mode)
{
    MppEncRefLtFrmCfg lt_ref[4];
    MppEncRefStFrmCfg st_ref[16];
    RK_S32 lt_cnt = 0;
    RK_S32 st_cnt = 0;
    MPP_RET ret = MPP_OK;

    memset(&lt_ref, 0, sizeof(lt_ref));
    memset(&st_ref, 0, sizeof(st_ref));

    switch (gop_mode) {
    case 3 : {
        // tsvc4
        //      /-> P1      /-> P3        /-> P5      /-> P7
        //     /           /             /           /
        //    //--------> P2            //--------> P6
        //   //                        //
        //  ///---------------------> P4
        // ///
        // P0 ------------------------------------------------> P8
        lt_cnt = 1;

        /* set 8 frame lt-ref gap */
        lt_ref[0].lt_idx        = 0;
        lt_ref[0].temporal_id   = 0;
        lt_ref[0].ref_mode      = REF_TO_PREV_LT_REF;
        lt_ref[0].lt_gap        = 8;
        lt_ref[0].lt_delay      = 0;

        st_cnt = 9;
        /* set tsvc4 st-ref struct */
        /* st 0 layer 0 - ref */
        st_ref[0].is_non_ref    = 0;
        st_ref[0].temporal_id   = 0;
        st_ref[0].ref_mode      = REF_TO_TEMPORAL_LAYER;
        st_ref[0].ref_arg       = 0;
        st_ref[0].repeat        = 0;
        /* st 1 layer 3 - non-ref */
        st_ref[1].is_non_ref    = 1;
        st_ref[1].temporal_id   = 3;
        st_ref[1].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[1].ref_arg       = 0;
        st_ref[1].repeat        = 0;
        /* st 2 layer 2 - ref */
        st_ref[2].is_non_ref    = 0;
        st_ref[2].temporal_id   = 2;
        st_ref[2].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[2].ref_arg       = 0;
        st_ref[2].repeat        = 0;
        /* st 3 layer 3 - non-ref */
        st_ref[3].is_non_ref    = 1;
        st_ref[3].temporal_id   = 3;
        st_ref[3].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[3].ref_arg       = 0;
        st_ref[3].repeat        = 0;
        /* st 4 layer 1 - ref */
        st_ref[4].is_non_ref    = 0;
        st_ref[4].temporal_id   = 1;
        st_ref[4].ref_mode      = REF_TO_PREV_LT_REF;
        st_ref[4].ref_arg       = 0;
        st_ref[4].repeat        = 0;
        /* st 5 layer 3 - non-ref */
        st_ref[5].is_non_ref    = 1;
        st_ref[5].temporal_id   = 3;
        st_ref[5].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[5].ref_arg       = 0;
        st_ref[5].repeat        = 0;
        /* st 6 layer 2 - ref */
        st_ref[6].is_non_ref    = 0;
        st_ref[6].temporal_id   = 2;
        st_ref[6].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[6].ref_arg       = 0;
        st_ref[6].repeat        = 0;
        /* st 7 layer 3 - non-ref */
        st_ref[7].is_non_ref    = 1;
        st_ref[7].temporal_id   = 3;
        st_ref[7].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[7].ref_arg       = 0;
        st_ref[7].repeat        = 0;
        /* st 8 layer 0 - ref */
        st_ref[8].is_non_ref    = 0;
        st_ref[8].temporal_id   = 0;
        st_ref[8].ref_mode      = REF_TO_TEMPORAL_LAYER;
        st_ref[8].ref_arg       = 0;
        st_ref[8].repeat        = 0;
    } break;
    case 2 : {
        // tsvc3
        //     /-> P1      /-> P3
        //    /           /
        //   //--------> P2
        //  //
        // P0/---------------------> P4
        lt_cnt = 0;

        st_cnt = 5;
        /* set tsvc4 st-ref struct */
        /* st 0 layer 0 - ref */
        st_ref[0].is_non_ref    = 0;
        st_ref[0].temporal_id   = 0;
        st_ref[0].ref_mode      = REF_TO_TEMPORAL_LAYER;
        st_ref[0].ref_arg       = 0;
        st_ref[0].repeat        = 0;
        /* st 1 layer 2 - non-ref */
        st_ref[1].is_non_ref    = 1;
        st_ref[1].temporal_id   = 2;
        st_ref[1].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[1].ref_arg       = 0;
        st_ref[1].repeat        = 0;
        /* st 2 layer 1 - ref */
        st_ref[2].is_non_ref    = 0;
        st_ref[2].temporal_id   = 1;
        st_ref[2].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[2].ref_arg       = 0;
        st_ref[2].repeat        = 0;
        /* st 3 layer 2 - non-ref */
        st_ref[3].is_non_ref    = 1;
        st_ref[3].temporal_id   = 2;
        st_ref[3].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[3].ref_arg       = 0;
        st_ref[3].repeat        = 0;
        /* st 4 layer 0 - ref */
        st_ref[4].is_non_ref    = 0;
        st_ref[4].temporal_id   = 0;
        st_ref[4].ref_mode      = REF_TO_TEMPORAL_LAYER;
        st_ref[4].ref_arg       = 0;
        st_ref[4].repeat        = 0;
    } break;
    case 1 : {
        // tsvc2
        //   /-> P1
        //  /
        // P0--------> P2
        lt_cnt = 0;

        st_cnt = 3;
        /* set tsvc4 st-ref struct */
        /* st 0 layer 0 - ref */
        st_ref[0].is_non_ref    = 0;
        st_ref[0].temporal_id   = 0;
        st_ref[0].ref_mode      = REF_TO_TEMPORAL_LAYER;
        st_ref[0].ref_arg       = 0;
        st_ref[0].repeat        = 0;
        /* st 1 layer 2 - non-ref */
        st_ref[1].is_non_ref    = 1;
        st_ref[1].temporal_id   = 1;
        st_ref[1].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[1].ref_arg       = 0;
        st_ref[1].repeat        = 0;
        /* st 2 layer 1 - ref */
        st_ref[2].is_non_ref    = 0;
        st_ref[2].temporal_id   = 0;
        st_ref[2].ref_mode      = REF_TO_PREV_REF_FRM;
        st_ref[2].ref_arg       = 0;
        st_ref[2].repeat        = 0;
    } break;
    default : {
        mpp_err_f("unsupport gop mode %d\n", gop_mode);
    } break;
    }

    if (lt_cnt || st_cnt) {
        ret = mpp_enc_ref_cfg_set_cfg_cnt(ref, lt_cnt, st_cnt);

        if (lt_cnt)
            ret = mpp_enc_ref_cfg_add_lt_cfg(ref, lt_cnt, lt_ref);

        if (st_cnt)
            ret = mpp_enc_ref_cfg_add_st_cfg(ref, st_cnt, st_ref);

        /* check and get dpb size */
        ret = mpp_enc_ref_cfg_check(ref);
    }

    return ret;
}

MPP_RET mpi_enc_gen_smart_gop_ref_cfg(MppEncRefCfg ref, RK_S32 gop_len, RK_S32 vi_len)
{
    MppEncRefLtFrmCfg lt_ref[4];
    MppEncRefStFrmCfg st_ref[16];
    RK_S32 lt_cnt = 1;
    RK_S32 st_cnt = 8;
    RK_S32 pos = 0;
    MPP_RET ret = MPP_OK;

    memset(&lt_ref, 0, sizeof(lt_ref));
    memset(&st_ref, 0, sizeof(st_ref));

    ret = mpp_enc_ref_cfg_set_cfg_cnt(ref, lt_cnt, st_cnt);

    /* set 8 frame lt-ref gap */
    lt_ref[0].lt_idx        = 0;
    lt_ref[0].temporal_id   = 0;
    lt_ref[0].ref_mode      = REF_TO_PREV_LT_REF;
    lt_ref[0].lt_gap        = gop_len;
    lt_ref[0].lt_delay      = 0;

    ret = mpp_enc_ref_cfg_add_lt_cfg(ref, 1, lt_ref);

    /* st 0 layer 0 - ref */
    st_ref[pos].is_non_ref  = 0;
    st_ref[pos].temporal_id = 0;
    st_ref[pos].ref_mode    = REF_TO_PREV_INTRA;
    st_ref[pos].ref_arg     = 0;
    st_ref[pos].repeat      = 0;
    pos++;

    /* st 1 layer 1 - non-ref */
    if (vi_len > 1) {
        st_ref[pos].is_non_ref  = 0;
        st_ref[pos].temporal_id = 0;
        st_ref[pos].ref_mode    = REF_TO_PREV_REF_FRM;
        st_ref[pos].ref_arg     = 0;
        st_ref[pos].repeat      = vi_len - 2;
        pos++;
    }

    st_ref[pos].is_non_ref  = 0;
    st_ref[pos].temporal_id = 0;
    st_ref[pos].ref_mode    = REF_TO_PREV_INTRA;
    st_ref[pos].ref_arg     = 0;
    st_ref[pos].repeat      = 0;
    pos++;

    ret = mpp_enc_ref_cfg_add_st_cfg(ref, pos, st_ref);

    /* check and get dpb size */
    ret = mpp_enc_ref_cfg_check(ref);

    return ret;
}

MPP_RET mpi_enc_gen_osd_plt(MppEncOSDPlt *osd_plt, RK_U32 frame_cnt)
{
    /*
     * osd idx size range from 16x16 bytes(pixels) to hor_stride*ver_stride(bytes).
     * for general use, 1/8 Y buffer is enough.
     */
    static RK_U32 plt_table[8] = {
        MPP_ENC_OSD_PLT_RED,
        MPP_ENC_OSD_PLT_YELLOW,
        MPP_ENC_OSD_PLT_BLUE,
        MPP_ENC_OSD_PLT_GREEN,
        MPP_ENC_OSD_PLT_CYAN,
        MPP_ENC_OSD_PLT_TRANS,
        MPP_ENC_OSD_PLT_BLACK,
        MPP_ENC_OSD_PLT_WHITE,
    };

    if (osd_plt) {
        RK_U32 k = 0;
        RK_U32 base = frame_cnt & 7;

        for (k = 0; k < 256; k++)
            osd_plt->data[k].val = plt_table[(base + k) % 8];
    }
    return MPP_OK;
}

MPP_RET mpi_enc_gen_osd_data(MppEncOSDData *osd_data, MppBufferGroup group,
                             RK_U32 width, RK_U32 height, RK_U32 frame_cnt)
{
    MppEncOSDRegion *region = NULL;
    RK_U32 k = 0;
    RK_U32 num_region = 8;
    RK_U32 buf_offset = 0;
    RK_U32 buf_size = 0;
    RK_U32 mb_w_max = MPP_ALIGN(width, 16) / 16;
    RK_U32 mb_h_max = MPP_ALIGN(height, 16) / 16;
    RK_U32 step_x = MPP_ALIGN(mb_w_max, 8) / 8;
    RK_U32 step_y = MPP_ALIGN(mb_h_max, 16) / 16;
    RK_U32 mb_x = (frame_cnt * step_x) % mb_w_max;
    RK_U32 mb_y = (frame_cnt * step_y) % mb_h_max;
    RK_U32 mb_w = step_x;
    RK_U32 mb_h = step_y;
    MppBuffer buf = osd_data->buf;

    if (buf)
        buf_size = mpp_buffer_get_size(buf);

    /* generate osd region info */
    osd_data->num_region = num_region;

    region = osd_data->region;

    for (k = 0; k < num_region; k++, region++) {
        // NOTE: offset must be 16 byte aligned
        RK_U32 region_size = MPP_ALIGN(mb_w * mb_h * 256, 16);

        region->inverse = 1;
        region->start_mb_x = mb_x;
        region->start_mb_y = mb_y;
        region->num_mb_x = mb_w;
        region->num_mb_y = mb_h;
        region->buf_offset = buf_offset;
        region->enable = (mb_w && mb_h);

        buf_offset += region_size;

        mb_x += step_x;
        mb_y += step_y;
        if (mb_x >= mb_w_max)
            mb_x -= mb_w_max;
        if (mb_y >= mb_h_max)
            mb_y -= mb_h_max;
    }

    /* create buffer and write osd index data */
    if (buf_size < buf_offset) {
        if (buf)
            mpp_buffer_put(buf);

        mpp_buffer_get(group, &buf, buf_offset);
        if (NULL == buf)
            mpp_err_f("failed to create osd buffer size %d\n", buf_offset);
    }

    if (buf) {
        void *ptr = mpp_buffer_get_ptr(buf);
        region = osd_data->region;

        for (k = 0; k < num_region; k++, region++) {
            mb_w = region->num_mb_x;
            mb_h = region->num_mb_y;
            buf_offset = region->buf_offset;

            memset(ptr + buf_offset, k, mb_w * mb_h * 256);
        }
    }

    osd_data->buf = buf;

    return MPP_OK;
}

MPP_RET mpi_enc_test_cmd_show_opt(MpiEncTestArgs* cmd)
{
    mpp_log("cmd parse result:\n");
    mpp_log("run type %d (0 jpeg rknn; 1 jpeg rknn mpp; 2 yuv rknn; 3 yuv mpp; 4 yuv rknn mpp)\n", cmd->run_type);
    mpp_log("input  file name: %s\n", cmd->file_input);
    mpp_log("output file name: %s\n", cmd->file_output);
    mpp_log("frames to be encoded: %d\n", cmd->frame_num);
    mpp_log("width      : %d\n", cmd->width);
    mpp_log("height     : %d\n", cmd->height);
    mpp_log("format     : %d\n", cmd->format);
    mpp_log("type       : %d\n", cmd->type);
    mpp_log("SoC        : %s\n", cmd->soc_id ? "RK3588" : "RK3576");
    mpp_log("show_time  : %d\n", cmd->show_time);

    return MPP_OK;
}
