#!/bin/bash

# This script copies the header files from the MPP library to the specified directory.


##                          source                                                   destination
cp -v /home/timkingh/code/h_3588/dev_mpp/mpp/inc/*.h                    /home/timkingh/code/a_rknn/d_yolov5_seg/npu_yolov5_seg/super_enc_v3_test/libs/mpp/include
cp -v /home/timkingh/code/h_3588/dev_mpp/mpp/mpp/inc/mpp_internal.h     /home/timkingh/code/a_rknn/d_yolov5_seg/npu_yolov5_seg/super_enc_v3_test/libs/mpp/include

cp -v /home/timkingh/code/h_3588/dev_mpp/mpp/mpp/base/inc/mpp_trie.h    /home/timkingh/code/a_rknn/d_yolov5_seg/npu_yolov5_seg/super_enc_v3_test/libs/mpp/include/base
cp -v /home/timkingh/code/h_3588/dev_mpp/mpp/osal/inc/*.h               /home/timkingh/code/a_rknn/d_yolov5_seg/npu_yolov5_seg/super_enc_v3_test/libs/mpp/include/osal

