set +e

SE_PWD=`pwd`
SE_TOP=${SE_PWD}/../../..
TOOLCHAIN=/home/timkingh/code/n_1126b/sdk/tools/linux/toolchain/arm-rockchip1240-linux-gnueabihf/bin/arm-rockchip1240-linux-gnueabihf-
# TODO: modify target soc by cmd args
TARGET_SOC="rv1126b"

cmake -DTOOLCHAIN=${TOOLCHAIN} \
      -DCMAKE_TOOLCHAIN_FILE=./arm.linux.cross.cmake \
      -DTARGET_SOC=${TARGET_SOC} \
      ${SE_TOP}