#ifndef __MPP_PROCESS_H__
#define __MPP_PROCESS_H__

#include "super_enc_v3_test.h"
#include "mpp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

void *mpi_enc_test_ctx_get(void);
MPP_RET fread_input_file(SuperEncCtx *sec);
MPP_RET super_enc_mpp_init(SuperEncCtx *sec);
MPP_RET super_enc_mpp_process(SuperEncCtx *sec);
MPP_RET super_enc_mpp_deinit(SuperEncCtx *sec);

#ifdef __cplusplus
}
#endif

#endif // __MPP_PROCESS_H__