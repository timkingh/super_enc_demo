cmake_minimum_required(VERSION 3.10)

project(super_enc_v3_test)

message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
set(EXECUTABLE_OUTPUT_PATH ./)

include_directories(libs/librga/include)
include_directories(libs/rknpu2/include)
include_directories(libs/mpp/include)
include_directories(libs/mpp/include/osal)
include_directories(libs/mpp/include/base)
include_directories(libs/stb_image)
include_directories(libs/jpeg_turbo/include)
include_directories(libs/allocator/drm/)
include_directories(libs/allocator/dma/)
include_directories(postprocess)
include_directories(nn_utils)
include_directories(mpp_utils)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(RGA_LIB ${PROJECT_SOURCE_DIR}/libs/librga/Android/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
    set(RKNNRT_LIB ${PROJECT_SOURCE_DIR}/libs/rknpu2/Android/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
    set(MPP_LIB ${PROJECT_SOURCE_DIR}/libs/mpp/Android/${CMAKE_ANDROID_ARCH_ABI}/libmpp.so)
    #find_library(RGA_LIB NAMES rga PATHS libs/librga/Android/${CMAKE_ANDROID_ARCH_ABI})
    #find_library(RKNNRT_LIB rknnrt HINTS libs/rknpu2/Android/${CMAKE_ANDROID_ARCH_ABI})
    #find_library(MPP_LIB mpp HINTS libs/mpp/Android/${CMAKE_ANDROID_ARCH_ABI})
elseif (TARGET_SOC STREQUAL "rv1126b")
    add_definitions(-DRV1126B_ARMHF)
    find_library(RGA_LIB NAMES rga PATHS libs/librga/Linux/armhf)
    find_library(RKNNRT_LIB rknnrt HINTS libs/rknpu2/Linux/armhf)
    find_library(MPP_LIB rockchip_mpp HINTS libs/mpp/Linux/armhf)
else()
    find_library(RGA_LIB NAMES rga PATHS libs/librga/Linux/aarch64)
    find_library(RKNNRT_LIB rknnrt HINTS libs/rknpu2/Linux/aarch64)
    find_library(MPP_LIB rockchip_mpp HINTS libs/mpp/Linux/aarch64)
endif()

message("RGA_LIB: ${RGA_LIB}")
message("RKNNRT_LIB: ${RKNNRT_LIB}")
message("MPP_LIB: ${MPP_LIB}")

add_subdirectory(nn_utils)
add_subdirectory(postprocess)
add_subdirectory(mpp_utils)

add_executable(super_enc_v3_test super_enc_v3_test.cpp
               super_enc_common.c
               yolov5_seg.c
               rknn_process.cpp
               mpp_process.c)

target_link_libraries(super_enc_v3_test ${RKNNRT_LIB} ${RGA_LIB} ${MPP_LIB}
                      nn_utils postprocess mpp_utils)