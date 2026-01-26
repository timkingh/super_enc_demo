
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(JPG_LIB ${PROJECT_SOURCE_DIR}/libs/jpeg_turbo/Android/${CMAKE_ANDROID_ARCH_ABI}/libturbojpeg.a)
    #find_library(JPG_LIB turbojpeg HINTS ${PROJECT_SOURCE_DIR}/libs/jpeg_turbo/Android/${CMAKE_ANDROID_ARCH_ABI})
elseif (TARGET_SOC STREQUAL "rv1126b")
    find_library(JPG_LIB turbojpeg HINTS ${PROJECT_SOURCE_DIR}/libs/jpeg_turbo/Linux/armhf)
else()
    find_library(JPG_LIB turbojpeg HINTS ${PROJECT_SOURCE_DIR}/libs/jpeg_turbo/Linux/aarch64)
endif()
message("JPG_LIB: ${JPG_LIB}")

# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()

add_library(nn_utils STATIC
    file_utils.c
    image_drawing.c
    image_utils.c
    )

target_link_libraries(nn_utils ${JPG_LIB})
