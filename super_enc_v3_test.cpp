#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include "mpp_log.h"
#include "mpp_err.h"
#include "mpp_time.h"
#include "utils.h"
#include "rknn_process.h"
#include "mpp_process.h"
#include "super_enc_common.h"
#include "super_enc_v3_test.h"
#include "svn_info.h"

#define SUPER_DBG_FUNCTION             (0x00000001)
#define SUPER_DBG_MODEL                (0x00000002)

#define super_log(cond, fmt, ...)   do { if (cond) mpp_log_f(fmt, ## __VA_ARGS__); } while (0)
#define super_dbg(flag, fmt, ...)   super_log((super_debug & flag), fmt, ## __VA_ARGS__)
#define super_dbg_func(fmt, ...)    super_dbg(SUPER_DBG_FUNCTION, fmt, ## __VA_ARGS__)

static RK_S32 super_debug = 0;

/* read CME data for a single frame from cme_out file into sec->cme_cur_frame */
static MPP_RET cme_read_frame(SuperEncCtx *sec, int frame_id)
{
    FILE *fp = sec->fp_cme_out;
    CmeFrameData *cfd = &sec->cme_cur_frame;
    int ctu_total = cfd->ctu_total;
    int ctu_size = sec->cme_ctu_size;
    int w_in_ctu = sec->cme_w_in_ctu;
    int h_in_ctu = sec->cme_h_in_ctu;
    char line[256];

    /* zero out current frame buffer */
    memset(cfd->cu_data, 0, ctu_total * sizeof(CmeCuInfo));

    rewind(fp);
    while (fgets(line, sizeof(line), fp)) {
        int frm, cu_x, cu_y, cu_size, mv_x, mv_y, sad_b16;
        if (sscanf(line,
                   "frame=%d, cu_x=%d, cu_y=%d, cu_size=%d, mv_x=%d, mv_y=%d sad_b16=%d",
                   &frm, &cu_x, &cu_y, &cu_size, &mv_x, &mv_y, &sad_b16) == 7) {
            /* file is sorted by frame number; frame > target means current frame done */
            if (frm > frame_id)
                break;
            if (frm != frame_id)
                continue;

            int ctu_x = cu_x * cu_size / ctu_size;
            int ctu_y = cu_y * cu_size / ctu_size;
            if (ctu_x < 0 || ctu_x >= w_in_ctu ||
                ctu_y < 0 || ctu_y >= h_in_ctu)
                continue;

            int ctu_idx = ctu_y * w_in_ctu + ctu_x;
            CmeCuInfo *cu = &cfd->cu_data[ctu_idx];
            cu->cu_x = cu_x;
            cu->cu_y = cu_y;
            cu->cu_size = cu_size;
            cu->mv_x = mv_x;
            cu->mv_y = mv_y;
            cu->sad_b16 = sad_b16;
        }
    }

    return MPP_OK;
}

/* debug: print CME data of current frame to terminal, same format as cme_out file */
static void cme_dbg_print_frame(SuperEncCtx *sec, int frame_id)
{
    CmeFrameData *cfd = &sec->cme_cur_frame;
    int ctu_total = cfd->ctu_total;
    int w_in_ctu = sec->cme_w_in_ctu;
    int h_in_ctu = sec->cme_h_in_ctu;
    int i;

    mpp_log("--- frame %d (%dx%d CTU grid, %d CTUs, ctu_size=%d) ---\n",
            frame_id, w_in_ctu, h_in_ctu, ctu_total, sec->cme_ctu_size);

    for (i = 0; i < ctu_total; i++) {
        CmeCuInfo *cu = &cfd->cu_data[i];

        if (cu->mv_x || cu->mv_y)
            mpp_log("frame=%d, cu_x=%d, cu_y=%d, cu_size=%d, mv_x=%d, mv_y=%d sad_b16=%d\n",
                frame_id,
                cu->cu_x, cu->cu_y, cu->cu_size,
                cu->mv_x, cu->mv_y, cu->sad_b16);
    }
}

static MPP_RET mpi_enc_test_cmd_show_opt(MpiEncTestArgs* cmd)
{
    mpp_log("cmd parse result:\n");
    mpp_log("run type %d (0 jpeg rknn; 1 jpeg rknn mpp; 2 yuv rknn; 3 yuv mpp; 4 yuv rknn mpp; 5 yuv external rknn mpp)\n", cmd->run_type);
    mpp_log("input  file name: %s\n", cmd->file_input);
    mpp_log("output file name: %s\n", cmd->file_output);
    mpp_log("frames to be encoded: %d\n", cmd->frame_num);
    mpp_log("width      : %d\n", cmd->width);
    mpp_log("height     : %d\n", cmd->height);
    mpp_log("format     : %d\n", cmd->format);
    mpp_log("type       : %d\n", cmd->type);
    mpp_log("SoC        : %s\n", (cmd->soc_id == SOC_RK3576) ? "RK3576" :
                                 (cmd->soc_id == SOC_RK3588) ? "RK3588" : "RK3572");
    mpp_log("rknn_ver   : %d\n", cmd->rknn_ver);
    mpp_log("show_time  : %d\n", cmd->show_time);

    return MPP_OK;
}

static MPP_RET parse_command_options(int argc, char **argv, SuperEncCtx *sec)
{
    MpiEncTestArgs* cmd = mpi_enc_test_cmd_get();

    super_dbg_func("enter\n");

    if (!cmd) {
        mpp_err_f("alloc command options failed\n");
        return MPP_NOK;
    }

    if (mpi_enc_test_cmd_update_by_args(cmd, argc, argv)) {
        mpp_err_f("update command options failed\n");
        return MPP_NOK;
    }

#ifdef RV1126B_ARMHF
    cmd->kmpp_en = 1;  //TODO: parse from cmd line
#endif

    cmd->adjust_rect_coord = 1; //TODO：parse from cmd line
    cmd->yolo_scene_mode = 1;
    sec->args = cmd;
    sec->soc_name = (SocName)cmd->soc_id;
    mpi_enc_test_cmd_show_opt(cmd);

    super_dbg_func("exit\n");

    return MPP_OK;
}

static MPP_RET super_enc_v3_test_init(SuperEncCtx *sec)
{
    MPP_RET ret = MPP_OK;
    RunType run_type = sec->args->run_type;

    super_dbg_func("enter\n");

    if (run_type != RUN_JPEG_RKNN && run_type != RUN_JPEG_RKNN_MPP) {
        sec->fp_input = fopen64(sec->args->file_input, "rb");
        if (!sec->fp_input) {
            mpp_err_f("open input file %s failed: %s\n", sec->args->file_input, strerror(errno));
            return MPP_NOK;
        }
        if (sec->args->seek > 0) {
            RK_S64 frame_size = (RK_S64)sec->args->width * sec->args->height * 3 / 2;
            RK_S64 offset = (RK_S64)sec->args->seek * frame_size;
            mpp_log_f("seek input YUV to frame %d (offset=%ld bytes)\n", sec->args->seek, offset);

            if (fseeko(sec->fp_input, (off_t)offset, SEEK_SET) != 0) {
                mpp_err_f("seek input file to frame %d failed: %s\n", sec->args->seek, strerror(errno));
                return MPP_NOK;
            }
        }
    }

    if (sec->args->file_output) {
        sec->fp_output = fopen(sec->args->file_output, "wb");
        if (!sec->fp_output) {
            mpp_err_f("open output file %s failed\n", sec->args->file_output);
            return MPP_NOK;
        }
    }

    if (sec->args->nn_out) {
        sec->fp_nn_out = fopen(sec->args->nn_out, "wb");
        if (!sec->fp_nn_out) {
            mpp_err_f("open nn output file %s failed\n", sec->args->nn_out);
            return MPP_NOK;
        }
    }

    if (sec->args->nn_in) {
        sec->fp_nn_in = fopen(sec->args->nn_in, "rb");
        if (!sec->fp_nn_in) {
            mpp_err_f("open external npu input file %s failed\n", sec->args->nn_in);
            return MPP_NOK;
        }
    }

    if (sec->args->nn_dect_rect) {
        sec->fp_nn_dect_rect = fopen(sec->args->nn_dect_rect, "wb");
        if (!sec->fp_nn_dect_rect) {
            mpp_err_f("open nn output file %s failed\n", sec->args->nn_dect_rect);
            return MPP_NOK;
        }
    }

    if (sec->args->cme_out) {
        int ctu_size, ctu_total;

        sec->fp_cme_out = fopen(sec->args->cme_out, "rb");
        if (!sec->fp_cme_out) {
            mpp_err_f("open CME output file %s failed\n", sec->args->cme_out);
            return MPP_NOK;
        }

        /* CTU size: same logic as rknn_process.cpp */
        ctu_size = (sec->args->type == MPP_VIDEO_CodingAVC) ? 16 :
                   (sec->soc_name == SOC_RK3576) ? 32 : 64;

        sec->cme_ctu_size = ctu_size;
        sec->cme_w_in_ctu = (sec->args->width + ctu_size - 1) / ctu_size;
        sec->cme_h_in_ctu = (sec->args->height + ctu_size - 1) / ctu_size;
        ctu_total = sec->cme_w_in_ctu * sec->cme_h_in_ctu;

        /* allocate single-frame CTU buffer */
        sec->cme_cur_frame.ctu_total = ctu_total;
        sec->cme_cur_frame.cu_data = (CmeCuInfo *)calloc(ctu_total, sizeof(CmeCuInfo));
        if (!sec->cme_cur_frame.cu_data) {
            mpp_err_f("alloc cme cu_data failed\n");
            return MPP_NOK;
        }

        mpp_log("CME: %dx%d CTU grid, ctu_size=%d, %d CTUs/frame\n",
                sec->cme_w_in_ctu, sec->cme_h_in_ctu, ctu_size, ctu_total);
    }

    if (run_type != RUN_YUV_MPP) {
        ret = super_enc_rknn_init(sec);
        if (ret != MPP_OK) {
            mpp_err_f("super_enc_rknn_init failed\n");
            return MPP_NOK;
        }
    }

    if (run_type != RUN_JPEG_RKNN) {
        sec->mpp_ctx = mpi_enc_test_ctx_get();
        if (!sec->mpp_ctx) {
            mpp_err_f("mpi_enc_test_ctx_get failed\n");
            return MPP_NOK;
        }

        ret = super_enc_mpp_init(sec);
        if (ret != MPP_OK) {
            mpp_err_f("super_enc_mpp_init failed\n");
            return MPP_NOK;
        }
    }

    super_dbg_func("exit\n");

    return MPP_OK;
}

static MPP_RET super_enc_v3_test_deinit(SuperEncCtx *sec)
{
    MPP_RET ret = MPP_OK;
    RunType run_type = sec->args->run_type;

    super_dbg_func("enter\n");

    if (run_type != RUN_YUV_MPP) {
        ret = super_enc_rknn_release(sec);
        if (ret != MPP_OK) {
            mpp_err_f("super_enc_rknn_release failed\n");
            return MPP_NOK;
        }
    }

    if (run_type != RUN_JPEG_RKNN) {
        ret = super_enc_mpp_deinit(sec);
        if (ret != MPP_OK) {
            mpp_err_f("super_enc_mpp_deinit failed\n");
            return MPP_NOK;
        }
    }

    SE_FREE(sec->mpp_ctx);

    FCLOSE(sec->fp_input);
    FCLOSE(sec->fp_output);
    FCLOSE(sec->fp_nn_out);
    FCLOSE(sec->fp_nn_in);
    FCLOSE(sec->fp_nn_dect_rect);

    SE_FREE(sec->cme_cur_frame.cu_data);
    FCLOSE(sec->fp_cme_out);

    super_dbg_func("exit\n");

    return MPP_OK;
}

int main(int argc, char **argv)
{
    SuperEncCtx super_enc_ctx;
    SuperEncCtx *sec = &super_enc_ctx;
    RunType run_type = RUN_JPEG_RKNN;
    MPP_RET ret = MPP_OK;
    RK_S64 start_time, end_time;
    float total_time = 0;

    mpp_log("%s\n", SE_COMPILE_INFO);
    memset(sec, 0, sizeof(SuperEncCtx));

    if (parse_command_options(argc, argv, sec)) {
        mpp_err_f("parse command options failed\n");
        goto done;
    }

    if (super_enc_v3_test_init(sec)) {
        mpp_err_f("super_enc_v3_test_init failed\n");
        goto done;
    }

    run_type = sec->args->run_type;
    do {
        start_time = mpp_time();
        if (run_type != RUN_JPEG_RKNN && run_type != RUN_JPEG_RKNN_MPP) {
            if (fread_input_file(sec)) {
                if (feof(sec->fp_input)) {
                    clearerr(sec->fp_input);
                    rewind(sec->fp_input);
                    mpp_log("fp input loop times %d\n", ++sec->loop_times);
                    continue;
                } else {
                    mpp_err_f("fread input file exit\n");
                    goto done;
                }
            }
        }

        if (sec->fp_cme_out && (run_type == RUN_YUV_RKNN || run_type == RUN_YUV_RKNN_MPP)) {
            cme_read_frame(sec, sec->frame_count);
            if (sec->frame_count == 110 && 0)
                cme_dbg_print_frame(sec, sec->frame_count);
        }

        if (run_type != RUN_YUV_MPP) {
            if (super_enc_rknn_process(sec)) {
                mpp_err_f("super_enc_rknn_process failed\n");
                goto done;
            }
        }

        if (run_type != RUN_JPEG_RKNN && run_type != RUN_YUV_RKNN) {
            RK_S64 enc_start, enc_end;

            enc_start = mpp_time();
            if (super_enc_mpp_process(sec)) {
                mpp_err_f("super_enc_mpp_process failed\n");
                goto done;
            }
            enc_end = mpp_time();
            mpp_log("mpp enc cost %0.2f ms\n", (float)(enc_end - enc_start) / 1000);
        }

        end_time = mpp_time();
        total_time += (float)(end_time - start_time) / 1000;
        mpp_log("frame %d cost %0.2f ms\n\n", sec->frame_count, (float)(end_time - start_time) / 1000);
    } while (++sec->frame_count < sec->args->frame_num);

done:
    mpi_enc_test_cmd_put(sec->args);

    super_enc_v3_test_deinit(sec);

    mpp_log("super enc v3 test %d frame(s) is done in %0.2f ms\n", sec->frame_count, total_time);

    return 0;
}