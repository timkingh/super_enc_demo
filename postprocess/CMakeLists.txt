
include_directories(${PROJECT_SOURCE_DIR})

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/libs/opencv/opencv-android-sdk-build/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
elseif (TARGET_SOC STREQUAL "rv1126b")
    include_directories(${PROJECT_SOURCE_DIR}/libs/opencv/opencv-linux-armhf/include/)
    set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/libs/opencv/opencv-linux-armhf/share/OpenCV)
else()
    include_directories(${PROJECT_SOURCE_DIR}/libs/opencv/opencv-linux-aarch64/include/)
    set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/libs/opencv/opencv-linux-aarch64/share/OpenCV)
endif()

#set(TARGET_LIB_ARCH lib64)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

add_library(postprocess STATIC
    postproc_yolov5_seg_rknn2.cpp
    postproc_yolov6_rknn3.cpp
    )

target_link_libraries(postprocess -ldl ${OpenCV_LIBS})
