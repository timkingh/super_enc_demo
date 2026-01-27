# super_enc_v3_test README

## 简介

超级编码3.0是指引⼊AI感兴趣区域检测，编码器内部根据检测结果优化编码算法，提升编码质量。 ⽬前适配的AI模型包括检测模型yolov5和分割模型yolov5_seg，前者输出检测⽬标框，硬件开销较⼩，后者可以得到精准的内容分割结果，硬件开销较⼤。同等情况下，分割模型可以节省部分码率。根据AI检测结果，可以将图像内容区分为前景和背景。同时， AI检测的结果通过软件buffer的⽅式传递到编码器内部，因此现有接⼝可以兼容其他模型。 ⽀持该功能的平台为RK3588、 RK3576和RV1126B。  

## 编译

比如Linux 64位环境：

cd ~/build/linux/aarch64

./make-Makefiles.bash

make -j

## 命令选项

​    ./super_enc_v3_test -i input.yuv -rt 4 -rknn yolov5s_seg_for3588.rknn \

​        -dbe 1 -dbs 0 -atf 1 -atr_i 1 -atr_p 1 -atl 1 -o output.h265 \

​        -w 1920 -h 1080 -f 4 -t 16777220 -sm 1 -g 0:30:0 -fps 15 \

​        -fqc 24:40:25:41 -qc 33:20:45:21:46 -rc 5 -semode 2 -bps 0:1000000:1500000 -qpdd 0 -n 100 \

​        -soc 1 -segmap_calc_en 1



**-i：**输入文件路径，可以是JPEG图片或YUV序列。

**-w：**输入YUV宽度。（输入是JPEG时不需要）

**-h：**输入YUV高度。（输入是JPEG时不需要）

**-f：**输入YUV格式，0为YUV420sp，4为YUV420p。（输入是JPEG时不需要）

**-t：**输出格式，7表示H.264，16777220表示H.265。（输入是JPEG时不需要）

**-rc：**码控模式，0 - VBR， 1 - CBR， 2 - FIXQP，3 - AVBR， 4 - SMARTV1， 5 - SMARTV3

**-o：**输出文件路径，H.264/H.265码流。（输入是JPEG时暂不支持）

**-n：**运行的帧数

**-dbe：**去运动模糊使能，0或1

**-dbs：**去运动模糊强度，取值[0, 7]

**-atf：**去帧间闪烁使能，取值[0, 3]。

**-atr_i：**I帧去振铃效应使能，取值[0, 3]。

**-atr_p：**P帧去振铃效应使能，取值[0, 3]。

**-atr_l：**去条纹效应使能，取值[0, 3]。

**-sm：**场景模式，0 - CVR， 1- IPC

**-g：**GOP长度

**-fps：**帧率

**-fqc：**帧级QP约束，I帧最小帧级QP：I帧最大帧级QP：P帧最小帧级QP：P帧最大帧级QP

**-qc：**块级QP约束，初始帧级QP：I帧最小块级QP：I帧最大块级QP：P帧最小块级QP：P帧最大块级QP

**-bps：**码率约束，目标码率：最小码率：最大码率

**-qpdd：**H.265中CU qp划分深度

**-rt：**功能选择，demo执行不同的业务逻辑

​    0 - 输入JPEG图片，调用RKNN检测，输出检测框结果。主要用于确认NPU调用路径的正确性。

​    1 - 输入JPEG图片，调用RKNN检测和encoder编码。（暂时不支持）

​    2 - 输入YUV文件，调用RKNN检测，输出检测结果。主要用于确认YUV检测路径的正确性。

​    3 - 输入YUV文件，调用encoder编码。主要用于确认YUV调用encoder路径的正确性。

​    4 - 输入YUV文件，调用RKNN检测和encoder编码。

**-soc：** 0 - RK3576，1 - RK3588。不同芯片平台的RGA/encoder有部分差异，需要做不同处理。

**-rknn：**RKNN模型路径，根据芯片平台生成。

**-semode：**超级编码内部模式，0 - 关闭， 1 - 均衡模式， 2 - 质量优先， 3 - 码率优先， 4 - 用户自定义。

**-segmap_calc_en：**0 - 使用NN检测框结果； 1 - 使用NN分割映射结果。

**-smart_en：**0 - 关闭smart， 1 - smart v1， 3 - smart v3。（与-rc重叠，暂不使用）

**-nn_out：**NN分割映射结果的输出路径，用于功能调试，可以确认NPU检测的准确性。+

## 相关资料

MPP demo：https://github.com/HermanChen/mpp

NPU demo：https://github.com/airockchip/rknn_model_zoo

RGA demo：https://github.com/airockchip/librga

