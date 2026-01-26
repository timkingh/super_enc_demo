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

    if (sec->args->nn_dect_rect) {
        sec->fp_nn_dect_rect = fopen(sec->args->nn_dect_rect, "wb");
        if (!sec->fp_nn_dect_rect) {
            mpp_err_f("open nn output file %s failed\n", sec->args->nn_dect_rect);
            return MPP_NOK;
        }
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
    FCLOSE(sec->fp_nn_dect_rect);

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
                mpp_err_f("fread input file exit\n");
                goto done;
            }
        }

        if (run_type != RUN_YUV_MPP) {
            if (super_enc_rknn_process(sec)) {
                mpp_err_f("super_enc_rknn_process failed\n");
                goto done;
            }
        }

        if (run_type != RUN_JPEG_RKNN && run_type != RUN_YUV_RKNN) {
            if (super_enc_mpp_process(sec)) {
                mpp_err_f("super_enc_mpp_process failed\n");
                goto done;
            }
        }

        end_time = mpp_time();
        total_time += (float)(end_time - start_time) / 1000;
        mpp_log("frame %d cost %f ms\n", sec->frame_count, (float)(end_time - start_time) / 1000);
    } while (++sec->frame_count < sec->args->frame_num);

done:
    mpi_enc_test_cmd_put(sec->args);

    super_enc_v3_test_deinit(sec);

    mpp_log("super enc v3 test %d frame(s) is done in %f ms\n", sec->frame_count, total_time);

    return 0;
}