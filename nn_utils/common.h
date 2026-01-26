#ifndef _RKNN_MODEL_ZOO_COMMON_H_
#define _RKNN_MODEL_ZOO_COMMON_H_

/**
 * @brief Image pixel format
 *
 */
typedef enum {
    IMAGE_FORMAT_GRAY8,
    IMAGE_FORMAT_RGB888,
    IMAGE_FORMAT_RGBA8888,
    IMAGE_FORMAT_YUV420SP_NV21,
    IMAGE_FORMAT_YUV420SP_NV12,
    IMAGE_FORMAT_YUV420P
} image_format_t;

/**
 * @brief Image buffer
 *
 */
typedef struct {
    int width;
    int height;
    int width_stride;
    int height_stride;
    image_format_t format;
    unsigned char* virt_addr; /* used for jpeg input(2025.02.24) */
    int size;
    int fd;
    int use_dma32_buf; /* used on RK3588 platform(2025.03.13) */
} image_buffer_t;

/**
 * @brief Image rectangle
 *
 */
typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

#endif //_RKNN_MODEL_ZOO_COMMON_H_
