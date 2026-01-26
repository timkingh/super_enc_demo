set +e

SE_PWD=`pwd`
SE_TOP=${SE_PWD}/../../..
TOOLCHAIN=/home/pub/toolchain/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-
# TODO: modify target soc by cmd args
TARGET_SOC="rk3588"

cmake -DTOOLCHAIN=${TOOLCHAIN} \
      -DCMAKE_TOOLCHAIN_FILE=./arm.linux.cross.cmake \
      -DTARGET_SOC=${TARGET_SOC} \
      ${SE_TOP}