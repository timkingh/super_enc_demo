#include <string.h>
#include <math.h>
#include "rk_mpi.h"

#include "rk_venc_kcfg.h"
#include "rk_venc_cfg.h"

#include "mpp_env.h"
#include "mpp_mem.h"
#include "mpp_time.h"
#include "mpp_debug.h"
#include "mpp_common.h"
#include "mpp_soc.h"

#include "utils.h"
#include "mpi_enc_utils.h"
#include "mpp_rc_api.h"
#include "mpp_process.h"
#include "super_enc_common.h"

#include "kmpp_buffer.h"
#include "kmpp_frame.h"
#include "kmpp_packet.h"
#include "kmpp_meta.h"
#include "kmpp_obj.h"


#define MPPP_DBG_FUNCTION             (0x00000001)
#define MPPP_DBG_INFO                 (0x00000002)

#define mppp_log(cond, fmt, ...)   do { if (cond) mpp_log_f(fmt, ## __VA_ARGS__); } while (0)
#define mppp_dbg(flag, fmt, ...)   mppp_log((mppp_debug & flag), fmt, ## __VA_ARGS__)
#define mppp_dbg_func(fmt, ...)    mppp_dbg(MPPP_DBG_FUNCTION, fmt, ## __VA_ARGS__)
#define mppp_dbg_info(fmt, ...)    mppp_dbg(MPPP_DBG_INFO, fmt, ## __VA_ARGS__)

/*
 * gop_mode
 * 0     - default IPPPP gop
 * 1 ~ 3 - tsvc2 ~ tsvc4
 * >=  4 - smart gop mode
 */
extern MPP_RET mpi_enc_gen_ref_cfg(MppEncRefCfg ref, RK_S32 gop_mode);
extern MPP_RET mpi_enc_gen_smart_gop_ref_cfg(MppEncRefCfg ref, RK_S32 gop_len, RK_S32 vi_len);
extern MPP_RET mpi_enc_gen_osd_data(MppEncOSDData *osd_data, MppBufferGroup group,
                             RK_U32 width, RK_U32 height, RK_U32 frame_cnt);
extern MPP_RET mpi_enc_gen_osd_plt(MppEncOSDPlt *osd_plt, RK_U32 frame_cnt);

static RK_S32 mppp_debug = 2;

static RK_S32 aq_thd_smart[16] = {
    0,  0,  0,  0,  3,  3,  5,  5,
    8,  8,  8, 15, 15, 20, 25, 28
};

static RK_S32 aq_step_smart[16] = {
    -8, -7, -6, -5, -4, -3, -2, -1,
    0,  1,  2,  3,  4,  6,  7, 8
};

static RK_S32 aq_thd[16] = {
    0,  0,  0,  0,
    3,  3,  5,  5,
    8,  8,  8,  15,
    15, 20, 25, 25
};

static RK_S32 aq_step_i_ipc[16] = {
    -8, -7, -6, -5,
    -4, -3, -2, -1,
    0,  1,  2,  3,
    5,  7,  7,  8,
};

static RK_S32 aq_step_p_ipc[16] = {
    -8, -7, -6, -5,
    -4, -2, -1, -1,
    0,  2,  3,  4,
    6,  8,  9,  10,
};

typedef struct {
    // base flow context
    MppCtx ctx;
    MppApi *mpi;
    RK_S32 chn;

    // global flow control flag
    RK_U32 frm_eos;
    RK_U32 pkt_eos;
    RK_U32 frm_pkt_cnt;
    RK_S32 frame_num;
    RK_S32 frame_count;
    RK_U64 stream_size;
    /* end of encoding flag when set quit the loop */
    volatile RK_U32 loop_end;

    // src and dst
    FILE *fp_input;
    FILE *fp_output;
    FILE *fp_verify;

    /* encoder config set */
    MppEncCfg       cfg;
    MppEncPrepCfg   prep_cfg;
    MppEncRcCfg     rc_cfg;
    MppEncSliceSplit split_cfg;
    MppEncOSDPltCfg osd_plt_cfg;
    MppEncOSDPlt    osd_plt;
    MppEncOSDData   osd_data;
    // RoiRegionCfg    roi_region;
    MppEncROICfg    roi_cfg;

    // input / output
    MppBufferGroup buf_grp;
    MppBuffer frm_buf;
    MppBuffer pkt_buf;
    MppBuffer md_info;

    MppEncSeiMode sei_mode;
    MppEncHeaderMode header_mode;

    // paramter for resource malloc
    RK_U32 width;
    RK_U32 height;
    RK_U32 hor_stride;
    RK_U32 ver_stride;
    MppFrameFormat fmt;
    MppCodingType type;
    RK_S32 loop_times;

    // MppEncRoiCtx roi_ctx;

    MppVencKcfg     init_kcfg;
    KmppBufGrp      kbuf_grp;
    KmppBufGrpCfg   kgrp_cfg;
    KmppBuffer      kfrm_buf;
    KmppBufCfg      kfrm_buf_cfg;
    KmppBuffer      md_info_buf;
    KmppBufCfg      md_info_buf_cfg;
    KmppShmPtr      *shm;

    KmppBuffer      obj_kbuf;
    KmppBufCfg      obj_kbuf_cfg;

    // resources
    size_t header_size;
    size_t frame_size;
    size_t mdinfo_size;
    /* NOTE: packet buffer may overflow */
    size_t packet_size;
    size_t obj_size;

    RK_U32 osd_enable;
    RK_U32 osd_mode;
    RK_U32 split_mode;
    RK_U32 split_arg;
    RK_U32 split_out;

    RK_U32 user_data_enable;
    RK_U32 roi_enable;

    // rate control runtime parameter
    RK_S32 fps_in_flex;
    RK_S32 fps_in_den;
    RK_S32 fps_in_num;
    RK_S32 fps_out_flex;
    RK_S32 fps_out_den;
    RK_S32 fps_out_num;
    RK_S32 bps;
    RK_S32 bps_max;
    RK_S32 bps_min;
    RK_S32 rc_mode;
    RK_S32 gop_mode;
    RK_S32 gop_len;
    RK_S32 vi_len;
    RK_S32 scene_mode;
    RK_S32 cu_qp_delta_depth;
    RK_S32 anti_flicker_str;
    RK_S32 atr_str_i;
    RK_S32 atr_str_p;
    RK_S32 atl_str;
    RK_S32 sao_str_i;
    RK_S32 sao_str_p;
    RK_S64 first_frm;
    RK_S64 first_pkt;
} MppTestCtx;

static MPP_RET test_ctx_init(MpiEncTestArgs *cmd, MppTestCtx *p)
{
    MPP_RET ret = MPP_OK;

    mppp_dbg_func("enter\n");

    // get paramter from cmd
    p->width        = cmd->width;
    p->height       = cmd->height;
    p->hor_stride   = (cmd->hor_stride) ? (cmd->hor_stride) :
                      (MPP_ALIGN(cmd->width, 16));
    p->ver_stride   = (cmd->ver_stride) ? (cmd->ver_stride) :
                      (MPP_ALIGN(cmd->height, 16));
    p->fmt          = cmd->format;
    p->type         = cmd->type;
    p->bps          = cmd->bps_target;
    p->bps_min      = cmd->bps_min;
    p->bps_max      = cmd->bps_max;
    p->rc_mode      = cmd->rc_mode;
    p->frame_num    = cmd->frame_num;
    if (cmd->type == MPP_VIDEO_CodingMJPEG && p->frame_num == 0) {
        mpp_log("jpege default encode only one frame. Use -n [num] for rc case\n");
        p->frame_num = 1;
    }
    p->gop_mode     = cmd->gop_mode;
    p->gop_len      = cmd->gop_len;
    p->vi_len       = cmd->vi_len;
    p->fps_in_flex  = cmd->fps_in_flex;
    p->fps_in_den   = cmd->fps_in_den;
    p->fps_in_num   = cmd->fps_in_num;
    p->fps_out_flex = cmd->fps_out_flex;
    p->fps_out_den  = cmd->fps_out_den;
    p->fps_out_num  = cmd->fps_out_num;
    p->scene_mode   = cmd->scene_mode;
    p->cu_qp_delta_depth = cmd->cu_qp_delta_depth;
    p->anti_flicker_str = cmd->anti_flicker_str;
    p->atr_str_i = cmd->atr_str_i;
    p->atr_str_p = cmd->atr_str_p;
    p->atl_str = cmd->atl_str;
    p->sao_str_i = cmd->sao_str_i;
    p->sao_str_p = cmd->sao_str_p;

    if (cmd->soc_id == SOC_RK3588) {
        p->mdinfo_size = (MPP_ALIGN(p->hor_stride, 64) >> 6) *
                         (MPP_ALIGN(p->ver_stride, 64) >> 6) * 32;
    } else {
        /* 3576 */
        p->mdinfo_size = (MPP_VIDEO_CodingHEVC == cmd->type) ?
                         (MPP_ALIGN(p->hor_stride, 32) >> 5) *
                         (MPP_ALIGN(p->ver_stride, 32) >> 5) * 16 :
                         (MPP_ALIGN(p->hor_stride, 64) >> 6) *
                         (MPP_ALIGN(p->ver_stride, 16) >> 4) * 16;
    }

    // update resource parameter
    switch (p->fmt & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:
    case MPP_FMT_YUV420P: {
        p->frame_size = MPP_ALIGN(p->hor_stride, 64) * MPP_ALIGN(p->ver_stride, 64) * 3 / 2;
    } break;

    case MPP_FMT_YUV422_YUYV :
    case MPP_FMT_YUV422_YVYU :
    case MPP_FMT_YUV422_UYVY :
    case MPP_FMT_YUV422_VYUY :
    case MPP_FMT_YUV422P :
    case MPP_FMT_YUV422SP : {
        p->frame_size = MPP_ALIGN(p->hor_stride, 64) * MPP_ALIGN(p->ver_stride, 64) * 2;
    } break;
    case MPP_FMT_YUV400:
    case MPP_FMT_RGB444 :
    case MPP_FMT_BGR444 :
    case MPP_FMT_RGB555 :
    case MPP_FMT_BGR555 :
    case MPP_FMT_RGB565 :
    case MPP_FMT_BGR565 :
    case MPP_FMT_RGB888 :
    case MPP_FMT_BGR888 :
    case MPP_FMT_RGB101010 :
    case MPP_FMT_BGR101010 :
    case MPP_FMT_ARGB8888 :
    case MPP_FMT_ABGR8888 :
    case MPP_FMT_BGRA8888 :
    case MPP_FMT_RGBA8888 : {
        p->frame_size = MPP_ALIGN(p->hor_stride, 64) * MPP_ALIGN(p->ver_stride, 64);
    } break;

    default: {
        p->frame_size = MPP_ALIGN(p->hor_stride, 64) * MPP_ALIGN(p->ver_stride, 64) * 4;
    } break;
    }

    if (MPP_FRAME_FMT_IS_FBC(p->fmt)) {
        if ((p->fmt & MPP_FRAME_FBC_MASK) == MPP_FRAME_FBC_AFBC_V1)
            p->header_size = MPP_ALIGN(MPP_ALIGN(p->width, 16) * MPP_ALIGN(p->height, 16) / 16, SZ_4K);
        else
            p->header_size = MPP_ALIGN(p->width, 16) * MPP_ALIGN(p->height, 16) / 16;
    } else {
        p->header_size = 0;
    }

    //TODO: maximum length
    p->obj_size = MPP_ALIGN(p->width, 64) * MPP_ALIGN(p->height, 64) / 16 / 16 + SZ_1K;

    mppp_dbg_func("exit\n");

    return ret;
}

static MPP_RET test_ctx_deinit(MppTestCtx *p)
{
    mppp_dbg_func("enter\n");

    mppp_dbg_func("exit\n");

    return MPP_OK;
}

void *mpi_enc_test_ctx_get(void)
{
    mppp_dbg_func("enter\n");
    void *args = mpp_calloc(MppTestCtx, 1);

    mppp_dbg_func("exit\n");

    return args;
}

static MPP_RET test_mpp_enc_cfg_setup(MpiEncTestArgs *cmd, MppTestCtx *p)
{
    MppApi *mpi = p->mpi;
    MppCtx ctx = p->ctx;
    MppEncCfg cfg = p->cfg;
    RK_U32 quiet = cmd->quiet;
    MPP_RET ret;
    RK_U32 rotation;
    RK_U32 mirroring;
    RK_U32 flip;
    RK_U32 gop_mode = p->gop_mode;
    MppEncRefCfg ref = NULL;
    /* setup default parameter */
    if (p->fps_in_den == 0)
        p->fps_in_den = 1;
    if (p->fps_in_num == 0)
        p->fps_in_num = 30;
    if (p->fps_out_den == 0)
        p->fps_out_den = 1;
    if (p->fps_out_num == 0)
        p->fps_out_num = 30;

    if (!p->bps)
        p->bps = p->width * p->height / 8 * (p->fps_out_num / p->fps_out_den);

    if (cmd->rc_mode == MPP_ENC_RC_MODE_SMTRC ||
        cmd->rc_mode == MPP_ENC_RC_MODE_SE) {
        mpp_enc_cfg_set_st(cfg, "hw:aq_thrd_i", aq_thd_smart);
        mpp_enc_cfg_set_st(cfg, "hw:aq_thrd_p", aq_thd_smart);
        mpp_enc_cfg_set_st(cfg, "hw:aq_step_i", aq_step_smart);
        mpp_enc_cfg_set_st(cfg, "hw:aq_step_p", aq_step_smart);
    } else {
        mpp_enc_cfg_set_st(cfg, "hw:aq_thrd_i", aq_thd);
        mpp_enc_cfg_set_st(cfg, "hw:aq_thrd_p", aq_thd);
        mpp_enc_cfg_set_st(cfg, "hw:aq_step_i", aq_step_i_ipc);
        mpp_enc_cfg_set_st(cfg, "hw:aq_step_p", aq_step_p_ipc);
    }

    mpp_enc_cfg_set_s32(cfg, "rc:debreath_en", 0);
    mpp_enc_cfg_set_s32(cfg, "rc:debreath_strength", 16);
#ifdef RV1126B_ARMHF
    mpp_enc_cfg_set_s32(cfg, "rc:inst_br_lvl", 8);
    mpp_enc_cfg_set_s32(cfg, "base:smt3_en", cmd->rc_mode == MPP_ENC_RC_MODE_SE);
#endif

    mpp_enc_cfg_set_s32(cfg, "rc:max_reenc_times", 0);
    mpp_enc_cfg_set_s32(cfg, "tune:anti_flicker_str", p->anti_flicker_str);
    mpp_enc_cfg_set_s32(cfg, "tune:atr_str_i", p->atr_str_i);
    mpp_enc_cfg_set_s32(cfg, "tune:atr_str_p", p->atr_str_p);
    mpp_enc_cfg_set_s32(cfg, "tune:atl_str", p->atl_str);
    mpp_enc_cfg_set_s32(cfg, "tune:sao_str_i", p->sao_str_i);
    mpp_enc_cfg_set_s32(cfg, "tune:sao_str_p", p->sao_str_p);

    mpp_enc_cfg_set_s32(cfg, "tune:scene_mode", p->scene_mode);
    mpp_enc_cfg_set_s32(cfg, "tune:lambda_idx_i", cmd->lambda_idx_i ? cmd->lambda_idx_i : 4);
    mpp_enc_cfg_set_s32(cfg, "tune:lambda_idx_p", cmd->lambda_idx_p ? cmd->lambda_idx_p : 4);
    mpp_enc_cfg_set_s32(cfg, "tune:deblur_en", cmd->deblur_en);
    mpp_enc_cfg_set_s32(cfg, "tune:deblur_str", cmd->deblur_str);
    mpp_enc_cfg_set_s32(cfg, "tune:lgt_chg_lvl", cmd->lgt_chg_lvl);
    mpp_enc_cfg_set_s32(cfg, "tune:se_mode", cmd->se_mode);
    mpp_enc_cfg_set_s32(cfg, "tune:bg_delta_qp_i", cmd->bg_delta_qp_i ? cmd->bg_delta_qp_i : -2);
    mpp_enc_cfg_set_s32(cfg, "tune:bg_delta_qp_p", cmd->bg_delta_qp_p ? cmd->bg_delta_qp_p : -2);
    mpp_enc_cfg_set_s32(cfg, "tune:fg_delta_qp_i", cmd->fg_delta_qp_i ? cmd->fg_delta_qp_i : 3);
    mpp_enc_cfg_set_s32(cfg, "tune:fg_delta_qp_p", cmd->fg_delta_qp_p ? cmd->fg_delta_qp_p : 4);
    mpp_enc_cfg_set_s32(cfg, "tune:bmap_qpmin_i", cmd->bmap_qpmin_i ? cmd->bmap_qpmin_i : 20);
    mpp_enc_cfg_set_s32(cfg, "tune:bmap_qpmin_p", cmd->bmap_qpmin_p ? cmd->bmap_qpmin_p : 20);
    mpp_enc_cfg_set_s32(cfg, "tune:bmap_qpmax_i", cmd->bmap_qpmax_i ? cmd->bmap_qpmax_i : 51);
    mpp_enc_cfg_set_s32(cfg, "tune:bmap_qpmax_p", cmd->bmap_qpmax_p ? cmd->bmap_qpmax_p : 51);
    mpp_enc_cfg_set_s32(cfg, "tune:min_bg_fqp", cmd->min_bg_fqp ? cmd->min_bg_fqp : 35);
    mpp_enc_cfg_set_s32(cfg, "tune:max_bg_fqp", cmd->max_bg_fqp ? cmd->max_bg_fqp : 40);
    mpp_enc_cfg_set_s32(cfg, "tune:min_fg_fqp", cmd->min_fg_fqp ? cmd->min_fg_fqp : 30);
    mpp_enc_cfg_set_s32(cfg, "tune:max_fg_fqp", cmd->max_fg_fqp ? cmd->max_fg_fqp : 35);

    mpp_enc_cfg_set_s32(cfg, "prep:width", p->width);
    mpp_enc_cfg_set_s32(cfg, "prep:height", p->height);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", p->hor_stride);
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", p->ver_stride);
    mpp_enc_cfg_set_s32(cfg, "prep:format", p->fmt);

    mpp_enc_cfg_set_s32(cfg, "rc:mode", p->rc_mode);

    /* fix input / output frame rate */
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", p->fps_in_flex);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", p->fps_in_num);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denom", p->fps_in_den);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", p->fps_out_flex);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", p->fps_out_num);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denom", p->fps_out_den);

    /* drop frame or not when bitrate overflow */
    mpp_enc_cfg_set_u32(cfg, "rc:drop_mode", MPP_ENC_RC_DROP_FRM_DISABLED);
    mpp_enc_cfg_set_u32(cfg, "rc:drop_thd", 20);        /* 20% of max bps */
    mpp_enc_cfg_set_u32(cfg, "rc:drop_gap", 1);         /* Do not continuous drop frame */

    /* setup bitrate for different rc_mode */
    mpp_enc_cfg_set_s32(cfg, "rc:bps_target", p->bps);
    switch (p->rc_mode) {
    case MPP_ENC_RC_MODE_FIXQP : {
        /* do not setup bitrate on FIXQP mode */
    } break;
    case MPP_ENC_RC_MODE_CBR : {
        /* CBR mode has narrow bound */
        mpp_enc_cfg_set_s32(cfg, "rc:bps_max", p->bps_max ? p->bps_max : p->bps * 17 / 16);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_min", p->bps_min ? p->bps_min : p->bps * 15 / 16);
    } break;
    case MPP_ENC_RC_MODE_VBR :
    case MPP_ENC_RC_MODE_AVBR :
    case MPP_ENC_RC_MODE_SMTRC :
    case MPP_ENC_RC_MODE_SE : {
        /* VBR mode has wide bound */
        mpp_enc_cfg_set_s32(cfg, "rc:bps_max", p->bps_max ? p->bps_max : p->bps * 17 / 16);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_min", p->bps_min ? p->bps_min : p->bps * 1 / 16);
    } break;
    default : {
        /* default use CBR mode */
        mpp_enc_cfg_set_s32(cfg, "rc:bps_max", p->bps_max ? p->bps_max : p->bps * 17 / 16);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_min", p->bps_min ? p->bps_min : p->bps * 15 / 16);
    } break;
    }

    /* setup qp for different codec and rc_mode */
    switch (p->type) {
    case MPP_VIDEO_CodingAVC :
    case MPP_VIDEO_CodingHEVC : {
        switch (p->rc_mode) {
        case MPP_ENC_RC_MODE_FIXQP : {
            RK_S32 fix_qp = cmd->qp_init;

            mpp_enc_cfg_set_s32(cfg, "rc:qp_init", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_max", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_min", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 0);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_min_i", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_max_i", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_min_p", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_max_p", fix_qp);
        } break;
        case MPP_ENC_RC_MODE_CBR :
        case MPP_ENC_RC_MODE_VBR :
        case MPP_ENC_RC_MODE_AVBR :
        case MPP_ENC_RC_MODE_SMTRC:
        case MPP_ENC_RC_MODE_SE: {
            mpp_enc_cfg_set_s32(cfg, "rc:qp_init", cmd->qp_init ? cmd->qp_init : -1);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_max", cmd->qp_max ? cmd->qp_max : 51);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_min", cmd->qp_min ? cmd->qp_min : 10);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", cmd->qp_max_i ? cmd->qp_max_i : 51);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", cmd->qp_min_i ? cmd->qp_min_i : 10);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 2);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_min_i", cmd->fqp_min_i ? cmd->fqp_min_i : 10);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_max_i", cmd->fqp_max_i ? cmd->fqp_max_i : 51);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_min_p", cmd->fqp_min_p ? cmd->fqp_min_p : 10);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_max_p", cmd->fqp_max_p ? cmd->fqp_max_p : 51);
        } break;
        default : {
            mpp_err_f("unsupport encoder rc mode %d\n", p->rc_mode);
        } break;
        }
    } break;
    case MPP_VIDEO_CodingVP8 : {
        /* vp8 only setup base qp range */
        mpp_enc_cfg_set_s32(cfg, "rc:qp_init", cmd->qp_init ? cmd->qp_init : 40);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_max",  cmd->qp_max ? cmd->qp_max : 127);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_min",  cmd->qp_min ? cmd->qp_min : 0);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", cmd->qp_max_i ? cmd->qp_max_i : 127);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", cmd->qp_min_i ? cmd->qp_min_i : 0);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 6);
    } break;
    case MPP_VIDEO_CodingMJPEG : {
        /* jpeg use special codec config to control qtable */
        mpp_enc_cfg_set_s32(cfg, "jpeg:q_factor", cmd->qp_init ? cmd->qp_init : 80);
        mpp_enc_cfg_set_s32(cfg, "jpeg:qf_max", cmd->qp_max ? cmd->qp_max : 99);
        mpp_enc_cfg_set_s32(cfg, "jpeg:qf_min", cmd->qp_min ? cmd->qp_min : 1);
    } break;
    default : {
    } break;
    }

    /* setup codec  */
    mpp_enc_cfg_set_s32(cfg, "codec:type", p->type);
    switch (p->type) {
    case MPP_VIDEO_CodingAVC : {
        RK_U32 constraint_set;

        /*
         * H.264 profile_idc parameter
         * 66  - Baseline profile
         * 77  - Main profile
         * 100 - High profile
         */
        mpp_enc_cfg_set_s32(cfg, "h264:profile", 100);
        /*
         * H.264 level_idc parameter
         * 10 / 11 / 12 / 13    - qcif@15fps / cif@7.5fps / cif@15fps / cif@30fps
         * 20 / 21 / 22         - cif@30fps / half-D1@@25fps / D1@12.5fps
         * 30 / 31 / 32         - D1@25fps / 720p@30fps / 720p@60fps
         * 40 / 41 / 42         - 1080p@30fps / 1080p@30fps / 1080p@60fps
         * 50 / 51 / 52         - 4K@30fps
         */
        mpp_enc_cfg_set_s32(cfg, "h264:level", 40);
        mpp_enc_cfg_set_s32(cfg, "h264:cabac_en", 1);
        mpp_enc_cfg_set_s32(cfg, "h264:cabac_idc", 0);
        mpp_enc_cfg_set_s32(cfg, "h264:trans8x8", 1);

        mpp_env_get_u32("constraint_set", &constraint_set, 0);
        if (constraint_set & 0x3f0000)
            mpp_enc_cfg_set_s32(cfg, "h264:constraint_set", constraint_set);
    } break;
    case MPP_VIDEO_CodingHEVC : {
        mpp_enc_cfg_set_s32(cfg, "h265:diff_cu_qp_delta_depth", p->cu_qp_delta_depth);
    } break;
    case MPP_VIDEO_CodingMJPEG :
    case MPP_VIDEO_CodingVP8 : {
    } break;
    default : {
        mpp_err_f("unsupport encoder coding type %d\n", p->type);
    } break;
    }

    p->split_mode = 0;
    p->split_arg = 0;
    p->split_out = 0;

    mpp_env_get_u32("split_mode", &p->split_mode, MPP_ENC_SPLIT_NONE);
    mpp_env_get_u32("split_arg", &p->split_arg, 0);
    mpp_env_get_u32("split_out", &p->split_out, 0);

    if (p->split_mode) {
        mpp_log_q(quiet, "%p split mode %d arg %d out %d\n", ctx,
                  p->split_mode, p->split_arg, p->split_out);
        mpp_enc_cfg_set_s32(cfg, "split:mode", p->split_mode);
        mpp_enc_cfg_set_s32(cfg, "split:arg", p->split_arg);
        mpp_enc_cfg_set_s32(cfg, "split:out", p->split_out);
    }

    mpp_env_get_u32("mirroring", &mirroring, 0);
    mpp_env_get_u32("rotation", &rotation, 0);
    mpp_env_get_u32("flip", &flip, 0);

    mpp_enc_cfg_set_s32(cfg, "prep:mirroring", mirroring);
    mpp_enc_cfg_set_s32(cfg, "prep:rotation", rotation);
    mpp_enc_cfg_set_s32(cfg, "prep:flip", flip);

    // config gop_len and ref cfg
    mpp_enc_cfg_set_s32(cfg, "rc:gop", p->gop_len ? p->gop_len : p->fps_out_num * 2);

    mpp_env_get_u32("gop_mode", &gop_mode, gop_mode);
    if (gop_mode) {
        mpp_enc_ref_cfg_init(&ref);

        if (p->gop_mode < 4)
            mpi_enc_gen_ref_cfg(ref, gop_mode);
        else
            mpi_enc_gen_smart_gop_ref_cfg(ref, p->gop_len, p->vi_len);

        mpp_enc_cfg_set_ptr(cfg, "rc:ref_cfg", ref);
    }

    ret = mpi->control(ctx, MPP_ENC_SET_CFG, cfg);
    if (ret) {
        mpp_err("mpi control enc set cfg failed ret %d\n", ret);
        goto RET;
    }

    if (cmd->type == MPP_VIDEO_CodingAVC || cmd->type == MPP_VIDEO_CodingHEVC) {
        RcApiBrief rc_api_brief;
        rc_api_brief.type = cmd->type;
        rc_api_brief.name = (cmd->rc_mode == MPP_ENC_RC_MODE_SMTRC || cmd->rc_mode == MPP_ENC_RC_MODE_SE) ?
                            "smart" : "default";

        ret = mpi->control(ctx, MPP_ENC_SET_RC_API_CURRENT, &rc_api_brief);
        if (ret) {
            mpp_err("mpi control enc set rc api failed ret %d\n", ret);
            goto RET;
        }
    }

    if (ref)
        mpp_enc_ref_cfg_deinit(&ref);

    /* optional */
    {
        RK_U32 sei_mode;

        mpp_env_get_u32("sei_mode", &sei_mode, MPP_ENC_SEI_MODE_DISABLE);
        p->sei_mode = sei_mode;
        ret = mpi->control(ctx, MPP_ENC_SET_SEI_CFG, &p->sei_mode);
        if (ret) {
            mpp_err("mpi control enc set sei cfg failed ret %d\n", ret);
            goto RET;
        }
    }

    if (p->type == MPP_VIDEO_CodingAVC || p->type == MPP_VIDEO_CodingHEVC) {
        p->header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
        ret = mpi->control(ctx, MPP_ENC_SET_HEADER_MODE, &p->header_mode);
        if (ret) {
            mpp_err("mpi control enc set header mode failed ret %d\n", ret);
            goto RET;
        }
    }

    /* setup test mode by env */
    mpp_env_get_u32("osd_enable", &p->osd_enable, 0);
    mpp_env_get_u32("osd_mode", &p->osd_mode, MPP_ENC_OSD_PLT_TYPE_DEFAULT);
    mpp_env_get_u32("roi_enable", &p->roi_enable, 0);
    mpp_env_get_u32("user_data_enable", &p->user_data_enable, 0);

    // if (p->roi_enable) {
    //     mpp_enc_roi_init(&p->roi_ctx, p->width, p->height, p->type, 4);
    //     mpp_assert(p->roi_ctx);
    // }

RET:
    return ret;
}

static MPP_RET kmpp_cfg_init(MppTestCtx *p, RK_U32 deblur_en)
{
    MppVencKcfg init_kcfg = NULL;
    MPP_RET ret = MPP_NOK;

    mpp_venc_kcfg_init(&init_kcfg, MPP_VENC_KCFG_TYPE_INIT);
    if (!init_kcfg) {
        mpp_err_f("kmpp_venc_init_cfg_init failed\n");
        return ret;
    }

    p->init_kcfg = init_kcfg;

    mpp_venc_kcfg_set_u32(init_kcfg, "type", MPP_CTX_ENC);
    mpp_venc_kcfg_set_u32(init_kcfg, "coding", p->type);
    mpp_venc_kcfg_set_s32(init_kcfg, "chan_id", 0);
    mpp_venc_kcfg_set_s32(init_kcfg, "online", 0);
    mpp_venc_kcfg_set_u32(init_kcfg, "buf_size", 0);
    mpp_venc_kcfg_set_u32(init_kcfg, "max_strm_cnt", 0);
    mpp_venc_kcfg_set_u32(init_kcfg, "shared_buf_en", 0);
    mpp_venc_kcfg_set_u32(init_kcfg, "smart_en", (p->rc_mode == MPP_ENC_RC_MODE_SMTRC) ? 1 :
                                                 (p->rc_mode == MPP_ENC_RC_MODE_SE) ? 3 : 0);
    mpp_venc_kcfg_set_u32(init_kcfg, "max_width", p->width);
    mpp_venc_kcfg_set_u32(init_kcfg, "max_height", p->height);
    mpp_venc_kcfg_set_u32(init_kcfg, "max_lt_cnt", 0);
    mpp_venc_kcfg_set_u32(init_kcfg, "qpmap_en", 1);
    mpp_venc_kcfg_set_u32(init_kcfg, "deblur_en", deblur_en);
    mpp_venc_kcfg_set_u32(init_kcfg, "chan_dup", 0);
    mpp_venc_kcfg_set_u32(init_kcfg, "tmvp_enable", 0);
    mpp_venc_kcfg_set_u32(init_kcfg, "only_smartp", 0);
    /* set notify mode to zero to disable rockit ko call back */
    mpp_venc_kcfg_set_u32(init_kcfg, "ntfy_mode", 0);
    /* set input timeout to block mode to insure put_frame ioctl return while encoding finished */
    mpp_venc_kcfg_set_s32(init_kcfg, "input_timeout", MPP_POLL_BLOCK);
    mpp_venc_kcfg_set_s32(init_kcfg, "scene_mode", 0);

    ret = p->mpi->control(p->ctx, MPP_SET_VENC_INIT_KCFG, init_kcfg);
    if (ret)
        mpp_err_f("mpi control set kmpp enc cfg failed ret %d\n", ret);

    return ret;
}

static MPP_RET setup_kmpp_buffer(MppTestCtx *p)
{
    MPP_RET ret = MPP_OK;

    ret = kmpp_buf_grp_get(&p->kbuf_grp);
    if (ret) {
        mpp_err_f("kmpp_buf_grp_get failed");
        return ret;
    }

    p->kgrp_cfg = kmpp_buf_grp_to_cfg(p->kbuf_grp);
    kmpp_buf_grp_cfg_set_count(p->kgrp_cfg, 10);
    ret = kmpp_buf_grp_setup(p->kbuf_grp);
    if (ret) {
        mpp_err_f("setup buf grp cfg failed");
        return ret;
    }

    ret = kmpp_buffer_get(&p->kfrm_buf);
    if (ret) {
        mpp_err_f("kmpp_buffer_get failed");
        return ret;
    }

    p->kfrm_buf_cfg = kmpp_buffer_to_cfg(p->kfrm_buf);
    p->shm = kmpp_obj_to_shm(p->kbuf_grp);
    kmpp_buf_cfg_set_group(p->kfrm_buf_cfg, p->shm);
    kmpp_buf_cfg_set_size(p->kfrm_buf_cfg, p->frame_size + p->header_size);
    ret = kmpp_buffer_setup(p->kfrm_buf);
    if (ret)
        mpp_err_f("kmpp_buffer_setup failed");

    /* test for send buf through kmeta*/
    {
        ret = kmpp_buffer_get(&p->md_info_buf);
        if (ret) {
            mpp_err_f("kmpp_buffer_get for kmeta_buf failed");
            return ret;
        }
        p->md_info_buf_cfg = kmpp_buffer_to_cfg(p->md_info_buf);
        kmpp_buf_cfg_set_size(p->md_info_buf_cfg, 6666);
        ret = kmpp_buffer_setup(p->md_info_buf);
        if (ret) {
            mpp_err_f("kmpp_buffer_setup for kmeta_buf failed");
        }
    }

    /* object detection buffer through kmeta*/
    {
        ret = kmpp_buffer_get(&p->obj_kbuf);
        if (ret) {
            mpp_err_f("kmpp_buffer_get for obj_kbuf failed");
            return ret;
        }

        p->obj_kbuf_cfg = kmpp_buffer_to_cfg(p->obj_kbuf);
        kmpp_buf_cfg_set_size(p->obj_kbuf_cfg, p->obj_size);
        ret = kmpp_buffer_setup(p->obj_kbuf);
        if (ret) {
            mpp_err_f("kmpp_buffer_setup for obj_kbuf failed");
        }
    }

    return ret;
}

MPP_RET super_enc_mpp_init(SuperEncCtx *sec)
{
    MPP_RET ret = MPP_OK;
    MppTestCtx *ctx = (MppTestCtx *)sec->mpp_ctx;
    MppPollType timeout = MPP_POLL_BLOCK;

    mppp_dbg_func("enter\n");

    ret = test_ctx_init(sec->args, ctx);
    if (ret) {
        mpp_err("test ctx init failed\n");
        return ret;
    }

    ret = mpp_buffer_group_get_internal(&ctx->buf_grp, MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE);
    if (ret) {
        mpp_err_f("failed to get mpp buffer group ret %d\n", ret);
        return ret;
    }

    ret = mpp_buffer_get(ctx->buf_grp, &ctx->frm_buf, ctx->frame_size + ctx->header_size);
    if (ret) {
        mpp_err_f("failed to get buffer for input frame ret %d\n", ret);
        return ret;
    }

    ret = mpp_buffer_get(ctx->buf_grp, &ctx->pkt_buf, ctx->frame_size);
    if (ret) {
        mpp_err_f("failed to get buffer for output packet ret %d\n", ret);
        return ret;
    }

    ret = mpp_buffer_get(ctx->buf_grp, &ctx->md_info, ctx->mdinfo_size);
    if (ret) {
        mpp_err_f("failed to get buffer for motion info output packet ret %d\n", ret);
        return ret;
    }

    // encoder demo
    ret = mpp_create(&ctx->ctx, &ctx->mpi);
    if (ret) {
        mpp_err("mpp_create failed ret %d\n", ret);
        return ret;
    }

    mpp_log("%p encoder test start w %d h %d type %d\n",
            ctx->ctx, ctx->width, ctx->height, ctx->type);

    ret = ctx->mpi->control(ctx->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
    if (MPP_OK != ret) {
        mpp_err("mpi control set output timeout %d ret %d\n", timeout, ret);
        return ret;
    }

    if (sec->args->kmpp_en) {
        kmpp_cfg_init(ctx, sec->args->deblur_en);
        setup_kmpp_buffer(ctx);
    }

    ret = mpp_init(ctx->ctx, MPP_CTX_ENC, ctx->type);
    if (ret) {
        mpp_err("mpp_init failed ret %d\n", ret);
        return ret;
    }

    if (sec->args->kmpp_en)
        ret = mpp_enc_cfg_init_k(&ctx->cfg);
    else
        ret = mpp_enc_cfg_init(&ctx->cfg);
    if (ret) {
        mpp_err_f("mpp_enc_cfg_init failed ret %d\n", ret);
        return ret;
    }

    ret = ctx->mpi->control(ctx->ctx, MPP_ENC_GET_CFG, ctx->cfg);
    if (ret) {
        mpp_err_f("get enc cfg failed ret %d\n", ret);
        return ret;
    }

    ret = test_mpp_enc_cfg_setup(sec->args, ctx);
    if (ret) {
        mpp_err_f("test mpp setup failed ret %d\n", ret);
        return ret;
    }

    ctx->fp_input = sec->fp_input;
    ctx->fp_output = sec->fp_output;
#ifndef RV1126B_ARMHF
    sec->src_buf = mpp_buffer_get_ptr(ctx->frm_buf);
#endif

    mppp_dbg_func("exit\n");

    return ret;
}

#ifdef RV1126B_ARMHF
MPP_RET fread_input_file(SuperEncCtx *sec)
{
    MppTestCtx *p = (MppTestCtx *)sec->mpp_ctx;
    MPP_RET ret = MPP_OK;
    KmppShmPtr sptr;
    void *kbuf = NULL;

    mppp_dbg_func("enter\n");
    kmpp_buf_cfg_get_sptr(p->kfrm_buf_cfg, &sptr);
    kbuf = sptr.uptr;
    sec->src_buf = sptr.uptr;

    ret = read_image_mpp(kbuf, p->fp_input, p->width, p->height,
                         p->hor_stride, p->ver_stride, p->fmt);
    if (ret == MPP_NOK) {
        mpp_log_f("read image failed\n");
        return MPP_NOK;
    }
    kmpp_buffer_flush(p->kfrm_buf);
    mppp_dbg_func("exit\n");
    return MPP_OK;
}
#else /* 3588/3576 */
MPP_RET fread_input_file(SuperEncCtx *sec)
{
    MppTestCtx *p = (MppTestCtx *)sec->mpp_ctx;
    char *buf = mpp_buffer_get_ptr(p->frm_buf);
    MPP_RET ret = MPP_OK;

    mppp_dbg_func("enter\n");

    mpp_buffer_sync_begin(p->frm_buf);
    ret = read_image_mpp(buf, p->fp_input, p->width, p->height,
                         p->hor_stride, p->ver_stride, p->fmt);
    if (ret == MPP_NOK) {
        mpp_log_f("read image failed\n");
        return MPP_NOK;
    }
    mpp_buffer_sync_end(p->frm_buf);

    mppp_dbg_func("exit\n");

    return MPP_OK;
}
#endif

static MPP_RET mpp_get_sps_pps(MppTestCtx *p)
{
    MPP_RET ret = MPP_OK;
    MppApi *mpi = p->mpi;
    MppCtx ctx = p->ctx;
    MppPacket packet = NULL;

    /*
     * Can use packet with normal malloc buffer as input not pkt_buf.
     * Please refer to vpu_api_legacy.cpp for normal buffer case.
     * Using pkt_buf buffer here is just for simplifing demo.
     */
    mpp_packet_init_with_buffer(&packet, p->pkt_buf);
    /* NOTE: It is important to clear output packet length!! */
    mpp_packet_set_length(packet, 0);

    ret = mpi->control(ctx, MPP_ENC_GET_HDR_SYNC, packet);
    if (ret) {
        mpp_err("mpi control enc get extra info failed\n");
        return ret;
    } else {
        /* get and write sps/pps for H.264 */
        void *ptr   = mpp_packet_get_pos(packet);
        size_t len  = mpp_packet_get_length(packet);

        if (p->fp_output)
            fwrite(ptr, 1, len, p->fp_output);
    }

    mpp_packet_deinit(&packet);

    return ret;
}

#ifdef RV1126B_ARMHF
MPP_RET super_enc_mpp_process(SuperEncCtx *sec)
{
    MppTestCtx *p = (MppTestCtx *)sec->mpp_ctx;
    MpiEncTestArgs *cmd = sec->args;
    MPP_RET ret = MPP_OK;
    MppApi *mpi = p->mpi;
    MppCtx ctx = p->ctx;
    RK_U32 quiet = cmd->quiet;
    RK_S32 chn = 0;
    RK_U32 cap_num = 0;
    RK_FLOAT psnr_const = 0;
    RK_U32 sse_unit_in_pixel = 0;

    mppp_dbg_func("enter\n");

    if (!sec->get_sps_pps && (p->type == MPP_VIDEO_CodingAVC || p->type == MPP_VIDEO_CodingHEVC))
        sec->get_sps_pps = (mpp_get_sps_pps(p) == MPP_OK);

    if (p->type == MPP_VIDEO_CodingAVC || p->type == MPP_VIDEO_CodingHEVC) {
        sse_unit_in_pixel = p->type == MPP_VIDEO_CodingAVC ? 16 : 8;
        psnr_const = (16 + log2(MPP_ALIGN(p->width, sse_unit_in_pixel) *
                                MPP_ALIGN(p->height, sse_unit_in_pixel)));
    }

    {
        MppMeta meta = NULL;
        MppFrame frame = NULL;
        MppPacket packet = NULL;
        KmppShmPtr  sptr;
        KmppShmPtr  *sptr_p;
        KmppFrame   kframe = NULL;
        KmppPacket  kpacket = NULL;
        KmppMeta    kmeta = NULL;
        KmppBuffer  kbuffer = NULL;
        RK_U32 eoi = 1;

        kmpp_frame_get(&kframe);
        kmpp_frame_set_width(kframe, p->width);
        kmpp_frame_set_height(kframe, p->height);
        kmpp_frame_set_hor_stride(kframe, p->hor_stride);
        kmpp_frame_set_ver_stride(kframe, p->ver_stride);
        kmpp_frame_set_fmt(kframe, p->fmt);
        kmpp_frame_set_eos(kframe, p->frm_eos);

        sptr_p = kmpp_obj_to_shm(p->kfrm_buf);
        kmpp_frame_set_buffer(kframe, sptr_p);
        kmpp_buffer_inc_ref(p->kfrm_buf);
        kmpp_frame_get_meta(kframe, &kmeta);

        /* kmeta send obj buf */
        if (p->rc_mode == MPP_ENC_RC_MODE_SE || cmd->smart_en == 3) {
            KmppShmPtr sptr_test;
            kmpp_buf_cfg_get_sptr(p->obj_kbuf_cfg, &sptr_test);
            if (sptr_test.uptr) {
                object_map_result_list *dst = sptr_test.uptr;
                memset(dst, 0, p->obj_size);
                if (sec->om_results.object_seg_map)
                    memcpy((void *)dst, sec->om_results.object_seg_map, p->obj_size);
                mpp_log("obj_buf %p size %d\n", dst, p->obj_size);
                kmpp_buffer_flush(p->obj_kbuf);
            }

            sptr_p = kmpp_obj_to_shm(p->obj_kbuf);
            kmpp_meta_set_shm(kmeta, KEY_NPU_SOBJ_FLAG, sptr_p);

            mpp_enc_cfg_set_s32(p->cfg, "tune:fg_area", sec->om_results.foreground_area);
            ret = mpi->control(ctx, MPP_ENC_SET_CFG, p->cfg);
            if (ret) {
                mpp_err("mpi control enc set cfg failed ret %d\n", ret);
                goto RET;
            }
        }

        if (!p->first_frm)
            p->first_frm = mpp_time();
        /*
         * NOTE: in non-block mode the frame can be resent.
         * The default input timeout mode is block.
         *
         * User should release the input frame to meet the requirements of
         * resource creator must be the resource destroyer.
         */
        ret = mpi->encode_put_frame(ctx, kframe);

        if (ret) {
            mpp_err("chn %d encode put frame failed\n", chn);
            // kmpp_frame_put(kframe);
            goto RET;
        }

        do {
            ret = mpi->encode_get_packet(ctx, &packet);
            if (ret) {
                mpp_err("chn %d encode get packet failed\n", chn);
                kmpp_frame_put(kframe);
                goto RET;
            }

            mpp_assert(packet);

            if (packet) {
                // write packet to file here
                kmpp_frame_put(kframe);

                void *ptr   = mpp_packet_get_pos(packet);
                size_t len  = mpp_packet_get_length(packet);
                char log_buf[256];
                RK_S32 log_size = sizeof(log_buf) - 1;
                RK_S32 log_len = 0;

                if (!p->first_pkt)
                    p->first_pkt = mpp_time();

                p->pkt_eos = mpp_packet_get_eos(packet);

                if (p->fp_output)
                    fwrite(ptr, 1, len, p->fp_output);

                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                    "encoded frame %-4d", p->frame_count);

                /* for low delay partition encoding */
                if (mpp_packet_is_partition(packet)) {
                    eoi = mpp_packet_is_eoi(packet);

                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                        " pkt %d", p->frm_pkt_cnt);
                    p->frm_pkt_cnt = (eoi) ? (0) : (p->frm_pkt_cnt + 1);
                }

                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                    " size %-7zu", len);

                if (mpp_packet_has_meta(packet)) {
                    meta = mpp_packet_get_meta(packet);
                    RK_S32 temporal_id = 0;
                    RK_S32 lt_idx = -1;
                    RK_S32 avg_qp = -1, bps_rt = -1;
                    RK_S32 use_lt_idx = -1;
                    RK_S64 sse = 0;
                    RK_FLOAT psnr = 0;

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_TEMPORAL_ID, &temporal_id))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " tid %d", temporal_id);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_LONG_REF_IDX, &lt_idx))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " lt %d", lt_idx);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_AVERAGE_QP, &avg_qp))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " qp %2d", avg_qp);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_BPS_RT, &bps_rt))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " bps_rt %d", bps_rt);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_USE_LTR, &use_lt_idx))
                        log_len += snprintf(log_buf + log_len, log_size - log_len, " vi");

                    if (MPP_OK == mpp_meta_get_s64(meta, KEY_ENC_SSE, &sse)) {
                        psnr = 3.01029996 * (psnr_const - log2(sse));
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " psnr %.4f", psnr);
                    }
                }

                mpp_log_q(quiet, "chn %d %s\n", chn, log_buf);

                mpp_packet_deinit(&packet);
                fps_calc_inc(cmd->fps);

                p->stream_size += len;
                p->frame_count += eoi;

                if (p->pkt_eos) {
                    mpp_log_q(quiet, "chn %d found last packet\n", chn);
                    mpp_assert(p->frm_eos);
                }
            }
        } while (!eoi);
    }

RET:
    mppp_dbg_func("exit\n");

    return ret;
}


#else /* 3588/3576 */

MPP_RET super_enc_mpp_process(SuperEncCtx *sec)
{
    MPP_RET ret = MPP_OK;
    MppTestCtx *p = (MppTestCtx *)sec->mpp_ctx;
    MpiEncTestArgs *cmd = sec->args;
    MppApi *mpi = p->mpi;
    MppCtx ctx = p->ctx;
    RK_U32 quiet = cmd->quiet;
    RK_S32 chn = 0;
    RK_U32 cap_num = 0;
    RK_FLOAT psnr_const = 0;
    RK_U32 sse_unit_in_pixel = 0;

    mppp_dbg_func("enter\n");

    if (!sec->get_sps_pps && (p->type == MPP_VIDEO_CodingAVC || p->type == MPP_VIDEO_CodingHEVC))
        sec->get_sps_pps = (mpp_get_sps_pps(p) == MPP_OK);

    if (p->type == MPP_VIDEO_CodingAVC || p->type == MPP_VIDEO_CodingHEVC) {
        sse_unit_in_pixel = p->type == MPP_VIDEO_CodingAVC ? 16 : 8;
        psnr_const = (16 + log2(MPP_ALIGN(p->width, sse_unit_in_pixel) *
                                MPP_ALIGN(p->height, sse_unit_in_pixel)));
    }

    {
        MppMeta meta = NULL;
        MppFrame frame = NULL;
        MppPacket packet = NULL;
        void *buf = mpp_buffer_get_ptr(p->frm_buf);
        RK_U32 eoi = 1;

        ret = mpp_frame_init(&frame);
        if (ret) {
            mpp_err_f("mpp_frame_init failed\n");
            goto RET;
        }

        mpp_frame_set_width(frame, p->width);
        mpp_frame_set_height(frame, p->height);
        mpp_frame_set_hor_stride(frame, p->hor_stride);
        mpp_frame_set_ver_stride(frame, p->ver_stride);
        mpp_frame_set_fmt(frame, p->fmt);
        mpp_frame_set_eos(frame, p->frm_eos);

        if (p->fp_input && feof(p->fp_input))
            mpp_frame_set_buffer(frame, NULL);
        else
            mpp_frame_set_buffer(frame, p->frm_buf);

        meta = mpp_frame_get_meta(frame);
        mpp_packet_init_with_buffer(&packet, p->pkt_buf);
        /* NOTE: It is important to clear output packet length!! */
        mpp_packet_set_length(packet, 0);
        mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);
        mpp_meta_set_buffer(meta, KEY_MOTION_INFO, p->md_info);

        if (p->rc_mode == MPP_ENC_RC_MODE_SE || cmd->smart_en == 3) {
            mpp_enc_cfg_set_s32(p->cfg, "tune:fg_area", sec->om_results.foreground_area);
            ret = mpi->control(ctx, MPP_ENC_SET_CFG, p->cfg);
            if (ret) {
                mpp_err("mpi control enc set cfg failed ret %d\n", ret);
                goto RET;
            }

            ret = mpp_meta_set_ptr(meta, KEY_NPU_UOBJ_FLAG, sec->om_results.object_seg_map);
            if(ret)
                mpp_err_f("meta %p set npu obj flag %p failed ret %d\n",
                          meta, sec->om_results.object_seg_map, ret);
        }

        if (p->osd_enable || p->user_data_enable || p->roi_enable) {
            if (p->user_data_enable) {
                MppEncUserData user_data;
                char *str = "this is user data\n";

                if ((p->frame_count & 10) == 0) {
                    user_data.pdata = str;
                    user_data.len = strlen(str) + 1;
                    mpp_meta_set_ptr(meta, KEY_USER_DATA, &user_data);
                }
                static RK_U8 uuid_debug_info[16] = {
                    0x57, 0x68, 0x97, 0x80, 0xe7, 0x0c, 0x4b, 0x65,
                    0xa9, 0x06, 0xae, 0x29, 0x94, 0x11, 0xcd, 0x9a
                };

                MppEncUserDataSet data_group;
                MppEncUserDataFull datas[2];
                char *str1 = "this is user data 1\n";
                char *str2 = "this is user data 2\n";
                data_group.count = 2;
                datas[0].len = strlen(str1) + 1;
                datas[0].pdata = str1;
                datas[0].uuid = uuid_debug_info;

                datas[1].len = strlen(str2) + 1;
                datas[1].pdata = str2;
                datas[1].uuid = uuid_debug_info;

                data_group.datas = datas;

                mpp_meta_set_ptr(meta, KEY_USER_DATAS, &data_group);
            }

            if (p->osd_enable) {
                /* gen and cfg osd plt */
                mpi_enc_gen_osd_plt(&p->osd_plt, p->frame_count);

                p->osd_plt_cfg.change = MPP_ENC_OSD_PLT_CFG_CHANGE_ALL;
                p->osd_plt_cfg.type = MPP_ENC_OSD_PLT_TYPE_USERDEF;
                p->osd_plt_cfg.plt = &p->osd_plt;

                ret = mpi->control(ctx, MPP_ENC_SET_OSD_PLT_CFG, &p->osd_plt_cfg);
                if (ret) {
                    mpp_err("mpi control enc set osd plt failed ret %d\n", ret);
                    goto RET;
                }

                /* gen and cfg osd plt */
                mpi_enc_gen_osd_data(&p->osd_data, p->buf_grp, p->width,
                                     p->height, p->frame_count);
                mpp_meta_set_ptr(meta, KEY_OSD_DATA, (void*)&p->osd_data);
            }
        }

        if (!p->first_frm)
            p->first_frm = mpp_time();
        /*
         * NOTE: in non-block mode the frame can be resent.
         * The default input timeout mode is block.
         *
         * User should release the input frame to meet the requirements of
         * resource creator must be the resource destroyer.
         */
        ret = mpi->encode_put_frame(ctx, frame);
        if (ret) {
            mpp_err("chn %d encode put frame failed\n", chn);
            mpp_frame_deinit(&frame);
            goto RET;
        }

        mpp_frame_deinit(&frame);

        do {
            ret = mpi->encode_get_packet(ctx, &packet);
            if (ret) {
                mpp_err("chn %d encode get packet failed\n", chn);
                goto RET;
            }

            mpp_assert(packet);

            if (packet) {
                // write packet to file here
                void *ptr   = mpp_packet_get_pos(packet);
                size_t len  = mpp_packet_get_length(packet);
                char log_buf[256];
                RK_S32 log_size = sizeof(log_buf) - 1;
                RK_S32 log_len = 0;

                if (!p->first_pkt)
                    p->first_pkt = mpp_time();

                p->pkt_eos = mpp_packet_get_eos(packet);

                if (p->fp_output)
                    fwrite(ptr, 1, len, p->fp_output);

                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                    "encoded frame %-4d", p->frame_count);

                /* for low delay partition encoding */
                if (mpp_packet_is_partition(packet)) {
                    eoi = mpp_packet_is_eoi(packet);

                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                        " pkt %d", p->frm_pkt_cnt);
                    p->frm_pkt_cnt = (eoi) ? (0) : (p->frm_pkt_cnt + 1);
                }

                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                    " size %-7zu", len);

                if (mpp_packet_has_meta(packet)) {
                    meta = mpp_packet_get_meta(packet);
                    RK_S32 temporal_id = 0;
                    RK_S32 lt_idx = -1;
                    RK_S32 avg_qp = -1, bps_rt = -1;
                    RK_S32 use_lt_idx = -1;
                    RK_S64 sse = 0;
                    RK_FLOAT psnr = 0;

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_TEMPORAL_ID, &temporal_id))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " tid %d", temporal_id);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_LONG_REF_IDX, &lt_idx))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " lt %d", lt_idx);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_AVERAGE_QP, &avg_qp))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " qp %d", avg_qp);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_BPS_RT, &bps_rt))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " rt_bps %d", bps_rt);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_USE_LTR, &use_lt_idx))
                        log_len += snprintf(log_buf + log_len, log_size - log_len, " vi");

                    if (MPP_OK == mpp_meta_get_s64(meta, KEY_ENC_SSE, &sse)) {
                        psnr = 3.01029996 * (psnr_const - log2(sse));
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " psnr %.4f", psnr);
                    }
                }

                mpp_log_q(quiet, "chn %d %s\n", chn, log_buf);

                mpp_packet_deinit(&packet);
                fps_calc_inc(cmd->fps);

                p->stream_size += len;
                p->frame_count += eoi;

                if (p->pkt_eos) {
                    mpp_log_q(quiet, "chn %d found last packet\n", chn);
                    mpp_assert(p->frm_eos);
                }
            }
        } while (!eoi);
    }

RET:
    mppp_dbg_func("exit\n");

    return ret;
}
#endif

MPP_RET super_enc_mpp_deinit(SuperEncCtx *sec)
{
    MPP_RET ret = MPP_OK;
    MppTestCtx *p = (MppTestCtx *)sec->mpp_ctx;

    mppp_dbg_func("enter\n");

    ret = test_ctx_deinit(p);

    if (p->ctx) {
        mpp_destroy(p->ctx);
        p->ctx = NULL;
    }

    if (p->cfg) {
        mpp_enc_cfg_deinit(p->cfg);
        p->cfg = NULL;
    }

    if (p->frm_buf) {
        mpp_buffer_put(p->frm_buf);
        p->frm_buf = NULL;
    }

    if (p->pkt_buf) {
        mpp_buffer_put(p->pkt_buf);
        p->pkt_buf = NULL;
    }

    if (p->md_info) {
        mpp_buffer_put(p->md_info);
        p->md_info = NULL;
    }

    if (p->kfrm_buf) {
        kmpp_buffer_put(p->kfrm_buf);
        p->kfrm_buf = NULL;
    }

    if (p->md_info_buf) {
        kmpp_buffer_put(p->md_info_buf);
        p->md_info_buf = NULL;
    }

    if (p->obj_kbuf) {
        kmpp_buffer_put(p->obj_kbuf);
        p->obj_kbuf = NULL;
    }

    if (p->kbuf_grp) {
        kmpp_buf_grp_put(p->kbuf_grp);
        p->kbuf_grp = NULL;
    }

    if (p->buf_grp) {
        mpp_buffer_group_put(p->buf_grp);
        p->buf_grp = NULL;
    }

    if (p->init_kcfg)
        mpp_venc_kcfg_deinit(p->init_kcfg);

    mppp_dbg_func("exit\n");

    return ret;
}