#!/bin/bash

set -e

rm -rfv CMakeCache.txt
rm -rfv CMakeFiles/

ANDROID_NDK_PATH=/home/pub/ndk/android-ndk-r19c
BUILD_TYPE="Release"
ANDROID_ABI="arm64-v8a"
TARGET_SOC="rk3576"

SE_PWD=`pwd`
SE_TOP=${SE_PWD}/../../..

#cmake  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake \
cmake  -DTARGET_SOC=${TARGET_SOC} \
       -DCMAKE_SYSTEM_NAME=Android \
       -DCMAKE_SYSTEM_VERSION=24 \
       -DCMAKE_ANDROID_ARCH_ABI=${ANDROID_ABI} \
       -DCMAKE_ANDROID_STL_TYPE=c++_static \
       -DCMAKE_ANDROID_NDK=${ANDROID_NDK_PATH} \
       -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        ${SE_TOP}