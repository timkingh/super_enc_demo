# super_enc_v3_test README

## 简介

超级编码3.0是指引⼊AI感兴趣区域检测，编码器内部根据检测结果优化编码算法，提升编码质量。 ⽬前适配的AI模型包括检测模型yolov5和分割模型yolov5_seg，前者输出检测⽬标框，硬件开销较⼩，后者可以得到精准的内容分割结果，硬件开销较⼤。同等情况下，分割模型可以节省部分码率。根据AI检测结果，可以将图像内容区分为前景和背景。同时， AI检测的结果通过软件buffer的⽅式传递到编码器内部，因此现有接⼝可以兼容其他模型。 ⽀持该功能的平台为RK3588、 RK3576和RV1126B。  

## 编译

比如Linux 64位环境：

cd ~/build/linux/aarch64

./make-Makefiles.bash

make -j

## 命令选项

​    ./super_enc_v3_test -i input.yuv -rt 5 -rknn yolov5s_seg_for3588.rknn \

​        -dbe 1 -dbs 0 -atf 1 -atr_i 1 -atr_p 1 -atl 1 -o output.h265 \

​        -w 1920 -h 1080 -f 4 -t 16777220 -sm 1 -g 0:30:0 -fps 15 \

​        -fqc 24:40:25:41 -qc 33:20:45:21:46 -rc 5 -semode 2 -bps 0:1000000:1500000 -qpdd 0 -n 100 \

​        -soc 1 -segmap_calc_en 1 -smart_en 3