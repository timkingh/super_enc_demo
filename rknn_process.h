#ifndef __RKNN_PROCESS_H__
#define __RKNN_PROCESS_H__

#include "super_enc_v3_test.h"
#include "mpp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

MPP_RET super_enc_rknn_init(SuperEncCtx *sec);
MPP_RET super_enc_rknn_process(SuperEncCtx *sec);
MPP_RET super_enc_rknn_release(SuperEncCtx *sec);

#ifdef __cplusplus
}
#endif

#endif // __RKNN_PROCESS_H__