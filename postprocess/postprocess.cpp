// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "opencv2/opencv.hpp"

#include "im2d.hpp"
// #include "dma_alloc.hpp"
// #include "drm_alloc.hpp"
#include "Float16.h"
#include "easy_timer.h"
#include "postprocess.h"
#include "mpp_log.h"
#include "mpp_common.h"

#include <set>
#include <vector>
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

#ifndef RV1126B_ARMHF
#define ENABLE_NEON
#define ENABLE_MATMUL_OPT /* dynamic shape */
#define ENABLE_MASK_MERGE
#endif

#define POST_DBG_FUNCTION             (0x00000001)
#define POST_DBG_DETAIL               (0x00000002)
#define POST_DBG_DETECT_RECT          (0x00000004)
#define POST_DBG_TIME                 (0x00000008)
#define POST_DBG_MATRICES             (0x00000010)
#define POST_DBG_MASK                 (0x00000020)

#define post_log(cond, fmt, ...)   do { if (cond) mpp_log_f(fmt, ## __VA_ARGS__); } while (0)
#define post_dbg(flag, fmt, ...)   post_log((post_debug & flag), fmt, ## __VA_ARGS__)
#define post_dbg_func(fmt, ...)    post_dbg(POST_DBG_FUNCTION, fmt, ## __VA_ARGS__)
#define post_dbg_detail(fmt, ...)  post_dbg(POST_DBG_DETAIL, fmt, ## __VA_ARGS__)
#define post_dbg_detect_rect(fmt, ...)  post_dbg(POST_DBG_DETECT_RECT, fmt, ## __VA_ARGS__)
#define post_dbg_time(fmt, ...)    do { if (post_debug & POST_DBG_TIME) \
                                           { timer.tok(); timer.print_time(fmt, ## __VA_ARGS__); }} while (0)
#define post_dbg_matrix(fmt, ...)  post_dbg(POST_DBG_MATRICES, fmt, ## __VA_ARGS__)
#define post_dbg_mask(fmt, ...)    post_dbg(POST_DBG_MASK, fmt, ## __VA_ARGS__)

static RK_S32 post_debug = 0x0;

static char *labels[OBJ_CLASS_NUM];

const int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                          {30, 61, 62, 45, 59, 119},
                          {116, 90, 156, 198, 373, 326}};

/* 0 - person; 1 - bicycle; 2 - car */
const int yolo_to_rk_class[8] = { 0, 1, 2, 1, -1, 2, -1, 2 };

int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[])
{
    // printf("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] == -1 || classIds[i] != filterId)
        {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;

    post_dbg_func("enter\n");

    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }

    post_dbg_func("exit\n");

    return low;
}

static void resize_by_opencv_fp(float *input_image, int input_width, int input_height, int boxes_num, float *output_image, int target_width, int target_height)
{
    post_dbg_func("enter\n");

    for (int b = 0; b < boxes_num; b++)
    {
        cv::Mat src_image(input_height, input_width, CV_32F, &input_image[b * input_width * input_height]);
        cv::Mat dst_image;
        cv::resize(src_image, dst_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        memcpy(&output_image[b * target_width * target_height], dst_image.data, target_width * target_height * sizeof(float));
    }

    post_dbg_func("exit\n");
}

static void resize_by_opencv_uint8(uint8_t *input_image, int input_width, int input_height,
                            int boxes_num, uint8_t *output_image, int target_width, int target_height)
{
    post_dbg_func("enter\n");

    for (int b = 0; b < boxes_num; b++) {
        cv::Mat src_image(input_height, input_width, CV_8U, &input_image[b * input_width * input_height]);
        cv::Mat dst_image;
        cv::resize(src_image, dst_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        memcpy(&output_image[b * target_width * target_height],
               dst_image.data, target_width * target_height * sizeof(uint8_t));
    }

    post_dbg_func("exit\n");
}

// void resize_by_rga_rk3588(uint8_t *input_image, int input_width, int input_height,
//                           uint8_t *output_image, int target_width, int target_height)
// {
//     char *src_buf, *dst_buf;
//     int src_buf_size, dst_buf_size;
//     rga_buffer_handle_t src_handle, dst_handle;
//     int src_width = input_width;
//     int src_height = input_height;
//     int src_format = RK_FORMAT_YCbCr_400;
//     int dst_width = target_width;
//     int dst_height = target_height;
//     int dst_format = RK_FORMAT_YCbCr_400;
//     int dst_dma_fd, src_dma_fd;
//     rga_buffer_t dst = {};
//     rga_buffer_t src = {};

//     post_dbg_func("enter\n");

//     dst_buf_size = dst_width * dst_height * get_bpp_from_format(dst_format);
//     src_buf_size = src_width * src_height * get_bpp_from_format(src_format);

//     /*
//      * Allocate dma_buf within 4G from dma32_heap,
//      * return dma_fd and virtual address.
//      */
//     dma_buf_alloc(DMA_HEAP_DMA32_UNCACHE_PATCH, dst_buf_size, &dst_dma_fd, (void **)&dst_buf);
//     dma_buf_alloc(DMA_HEAP_DMA32_UNCACHE_PATCH, src_buf_size, &src_dma_fd, (void **)&src_buf);
//     memcpy(src_buf, input_image, src_buf_size);

//     dst_handle = importbuffer_fd(dst_dma_fd, dst_buf_size);
//     src_handle = importbuffer_fd(src_dma_fd, src_buf_size);

//     dst = wrapbuffer_handle(dst_handle, dst_width, dst_height, dst_format);
//     src = wrapbuffer_handle(src_handle, src_width, src_height, src_format);

//     int ret = imresize(src, dst);
//     if (ret == IM_STATUS_SUCCESS)
//     {
//         printf("%s running success!\n", "rga_resize");
//     }
//     else
//     {
//         printf("%s running failed, %s\n", "rga_resize", imStrError((IM_STATUS)ret));
//     }

//     memcpy(output_image, dst_buf, target_width * target_height);

//     releasebuffer_handle(src_handle);
//     releasebuffer_handle(dst_handle);
//     dma_buf_free(src_buf_size, &src_dma_fd, src_buf);
//     dma_buf_free(dst_buf_size, &dst_dma_fd, dst_buf);

//     post_dbg_func("exit\n");
// }

class DrmObject
{
public:
    int drm_buffer_fd;
    int drm_buffer_handle;
    size_t actual_size;
    uint8_t *drm_buf;
};


static void crop_mask_fp(float *seg_mask, uint8_t *all_mask_in_one, float *boxes, int boxes_num,
                         int *cls_id, int height, int width, int scene_mode)
{
    post_dbg_func("enter\n");

    for (int b = 0; b < boxes_num; b++) {
        float x1 = boxes[b * 4 + 0];
        float y1 = boxes[b * 4 + 1];
        float x2 = boxes[b * 4 + 2];
        float y2 = boxes[b * 4 + 3];

        // convert yolo class to rk class
        if (scene_mode == 0) {
            /* 0: person 1: book */
            cls_id[b] = cls_id[b] == LABEL_BOOK;
        } else {
            cls_id[b] = yolo_to_rk_class[cls_id[b]];
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (j >= x1 && j < x2 && i >= y1 && i < y2) {
                    if (all_mask_in_one[i * width + j] == 0) {
                        if (seg_mask[b * width * height + i * width + j] > 0) {
                            all_mask_in_one[i * width + j] = (cls_id[b] + 1);
                        } else {
                            all_mask_in_one[i * width + j] = 0;
                        }
                    }
                }
            }
        }
    }

    post_dbg_func("exit\n");
}

#ifdef ENABLE_MASK_MERGE
static void crop_mask_uint8_merge(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes,
                     int boxes_num, int *cls_id, int height, int width, int scene_mode)
{
    int i, b, j, k;

    post_dbg_func("enter\n");

    for (int b = 0; b < boxes_num; b++) {
        float x1 = boxes[b * 4 + 0];
        float y1 = boxes[b * 4 + 1];
        float x2 = boxes[b * 4 + 2];
        float y2 = boxes[b * 4 + 3];

        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                if (j >= x1 && j < x2 && i >= y1 && i < y2 &&
                    all_mask_in_one[i * width + j] == 0) {
                    all_mask_in_one[i * width + j] = seg_mask[i * width + j] > 0;
                }
            }
        }
    }

    post_dbg_func("exit\n");
}
#endif

static void crop_mask_uint8(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes,
                     int boxes_num, int *cls_id, int height, int width, int scene_mode)
{
    int i, b, j, k;

    post_dbg_func("enter\n");

    for (int b = 0; b < boxes_num; b++) {
        float x1 = boxes[b * 4 + 0];
        float y1 = boxes[b * 4 + 1];
        float x2 = boxes[b * 4 + 2];
        float y2 = boxes[b * 4 + 3];

        // convert yolo class to rk class
        if (scene_mode == 0) {
            /* 0: person 1: book */
            cls_id[b] = cls_id[b] == LABEL_BOOK;
        } else {
            cls_id[b] = yolo_to_rk_class[cls_id[b]];
        }

        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                if (j >= x1 && j < x2 && i >= y1 && i < y2) {
                    if (all_mask_in_one[i * width + j] == 0) {
                        if (seg_mask[b * width * height + i * width + j] > 0) {
                            // add one for each class
                            all_mask_in_one[i * width + j] = (cls_id[b] + 1);
                        }
                    }
                }
            }
        }
    }

    post_dbg_func("exit\n");
}

static void matmul_by_cpu_fp(std::vector<float> &A, float *B, float *C, int ROWS_A, int COLS_A, int COLS_B)
{
    float temp = 0;

    post_dbg_func("enter\n");

    for (int i = 0; i < ROWS_A; i++)
    {
        for (int j = 0; j < COLS_B; j++)
        {
            temp = 0;
            for (int k = 0; k < COLS_A; k++)
            {
                temp += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
            C[i * COLS_B + j] = temp;
        }
    }

    post_dbg_func("exit\n");
}

static void matmul_by_cpu_uint8(std::vector<float> &A, float *B, uint8_t *C, int ROWS_A, int COLS_A, int COLS_B)
{
    float temp = 0;
    int index_A, index_C;

    post_dbg_func("enter\n");

    for (int i = 0; i < ROWS_A; i++)
    {
        index_A = i * COLS_A;
        index_C = i * COLS_B;
        for (int j = 0; j < COLS_B; j++)
        {
            temp = 0;
            for (int k = 0; k < COLS_A; k++)
            {
                temp += A[index_A + k] * B[k * COLS_B + j];
            }
            if (temp > 0)
            {
                C[index_C + j] = 4;
            }
        }
    }

    post_dbg_func("exit\n");
}

static void matmul_by_npu_fp(std::vector<float> &A_input, float *B_input, float *C_input,
                             int ROWS_A, int COLS_A, int COLS_B, RknnCtx *nn_ctx)
{
    int B_layout = 0;
    int AC_layout = 0;
    int32_t M = ROWS_A;
    int32_t K = COLS_A;
    int32_t N = COLS_B;

    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    TIMER timer;

    post_dbg_func("enter\n");

    memset(&info, 0, sizeof(rknn_matmul_info));
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = B_layout;
    info.AC_layout = AC_layout;

    rknn_matmul_io_attr io_attr;
    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));

    timer.tik();
    rknpu2::float16 int8Vector_A[ROWS_A * COLS_A];
    for (int i = 0; i < ROWS_A * COLS_A; ++i)
        int8Vector_A[i] = (rknpu2::float16)A_input[i];
    post_dbg_time("int8Vector_A");

    timer.tik();
    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    // Create A/B/C
    rknn_tensor_mem *A = rknn_create_mem(ctx, io_attr.A.size);
    rknn_tensor_mem *B = rknn_create_mem(ctx, io_attr.B.size);
    rknn_tensor_mem *C = rknn_create_mem(ctx, io_attr.C.size);

    post_dbg_time("rknn_create_mem");
    timer.tik();
    memcpy(A->virt_addr, int8Vector_A, A->size);
    memcpy(B->virt_addr, nn_ctx->vector_b, B->size);
    post_dbg_time("int8Vector_AB memcpy");

    timer.tik();
    // Set A/B/C
    ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
    ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
    ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
    post_dbg_time("rknn_matmul_set_io_mem");

    timer.tik();
    ret = rknn_matmul_run(ctx);
    post_dbg_time("rknn_matmul_run");

    timer.tik();
    for (int i = 0; i < ROWS_A * COLS_B; ++i)
        C_input[i] = ((float *)C->virt_addr)[i];
    post_dbg_time("C_input");

    // destroy
    timer.tik();
    rknn_destroy_mem(ctx, A);
    rknn_destroy_mem(ctx, B);
    rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
    post_dbg_time("rknn_destroy_mem");

    post_dbg_func("exit\n");
}

static int matmul_by_npu_fp_optimized(std::vector<float> &A_input,
#ifdef USE_FP_MASK_MAP
                                      float *C_input,
#else
                                      uint8_t *C_input,
#endif
                                      int ROWS_A, int COLS_A, int COLS_B, RknnCtx *nn_ctx)
{
    rknn_matmul_ctx mat_ctx = nn_ctx->matmul_ctx;
    int tensor_idx = ROWS_A - 1;
    rknn_matmul_io_attr *io_attr = &nn_ctx->io_attr[tensor_idx];
    TIMER timer;
    int ret = 0;

    post_dbg_func("enter\n");

    assert(ROWS_A <= OBJ_NUMB_MAX_SIZE);
    timer.tik();
    ret = rknn_matmul_set_dynamic_shape(mat_ctx, &nn_ctx->shapes[tensor_idx]);
    if (ret != 0) {
        mpp_err_f("rknn_matmul_set_dynamic_shapes fail! ret %d\n", ret);
        return -1;
    }
    post_dbg_time("3 - rknn_matmul_set_dynamic_shape");

    timer.tik();
    rknpu2::float16 int8Vector_A[ROWS_A * COLS_A];
    for (int i = 0; i < ROWS_A * COLS_A; ++i)
        int8Vector_A[i] = (rknpu2::float16)A_input[i];
    post_dbg_time("3 - int8Vector_A init");

    // post_dbg_matrix("current tensor_idx %d matrix A size %d B size %d C size %d\n",
    //                 tensor_idx, io_attr->A.size, io_attr->B.size, io_attr->C.size);

    timer.tik();
    memcpy(nn_ctx->tensor_a->virt_addr, int8Vector_A, io_attr->A.size);
    memcpy(nn_ctx->tensor_b->virt_addr, nn_ctx->vector_b, io_attr->B.size);
    post_dbg_time("3 - int8Vector_AB memcpy");

    timer.tik();
    ret = rknn_matmul_set_io_mem(mat_ctx, nn_ctx->tensor_a, &io_attr->A);
    ret = rknn_matmul_set_io_mem(mat_ctx, nn_ctx->tensor_b, &io_attr->B);
    ret = rknn_matmul_set_io_mem(mat_ctx, nn_ctx->tensor_c, &io_attr->C);
    post_dbg_time("3 - rknn_matmul_set_io_mem");

    // Run
    timer.tik();
    ret = rknn_matmul_run(mat_ctx);
    post_dbg_time("3 - rknn_matmul_run");

    timer.tik();
#ifdef USE_FP_MASK_MAP
    memcpy(C_input, nn_ctx->tensor_c->virt_addr, io_attr->C.size);
#else
    int tensor_c_len = io_attr->C.size / sizeof(float);
    // mpp_log("matrix C len: %d\n", tensor_c_len);
#ifdef ENABLE_MASK_MERGE
    int boxes_num = ROWS_A;
    int tensor_merge_len = tensor_c_len / boxes_num;
    float merge_c_result = 0;

    // mpp_log("boxes_num: %d, tensor_merge_len: %d\n", boxes_num, tensor_merge_len);
    for (int i = 0; i < tensor_merge_len; ++i) {
        merge_c_result = 0;
        for (int j = 0; j < boxes_num; ++j) {
            merge_c_result += ((float *)nn_ctx->tensor_c->virt_addr)[j * tensor_merge_len + i] > 0;
            // mpp_log("merge_c_result: %f\n", merge_c_result);
        }
        C_input[i] = merge_c_result > 0 ? 4 : 0;
    }
#else
    for (int i = 0; i < tensor_c_len; ++i)
        C_input[i] = ((float *)nn_ctx->tensor_c->virt_addr)[i] > 0 ? 4 : 0;
#endif
#endif
    post_dbg_time("3 - C_input: matmul output copy");
    post_dbg_func("exit\n");

    return ret;
}

static void seg_reverse(uint8_t *seg_mask, uint8_t *cropped_seg, uint8_t *seg_mask_real,
                 int model_in_height, int model_in_width, int cropped_height,
                 int cropped_width, int ori_in_height, int ori_in_width, int y_pad, int x_pad)
{
    post_dbg_func("enter\n");

    if (y_pad == 0 && x_pad == 0 && ori_in_height == model_in_height && ori_in_width == model_in_width)
    {
        memcpy(seg_mask_real, seg_mask, ori_in_height * ori_in_width);
        return;
    }

    int cropped_index = 0;
    for (int i = 0; i < model_in_height; i++)
    {
        for (int j = 0; j < model_in_width; j++)
        {
            if (i >= y_pad && i < model_in_height - y_pad && j >= x_pad && j < model_in_width - x_pad)
            {
                int seg_index = i * model_in_width + j;
                cropped_seg[cropped_index] = seg_mask[seg_index];
                cropped_index++;
            }
        }
    }
    // Note: Here are different methods provided for implementing single-channel image scaling.
    //       The method of using rga to resize the image requires that the image size is 2 aligned.
    resize_by_opencv_uint8(cropped_seg, cropped_width, cropped_height, 1, seg_mask_real, ori_in_width, ori_in_height);
    // resize_by_rga_rk356x(cropped_seg, cropped_width, cropped_height, seg_mask_real, ori_in_width, ori_in_height);
    // resize_by_rga_rk3588(cropped_seg, cropped_width, cropped_height, seg_mask_real, ori_in_width, ori_in_height);

    post_dbg_func("exit\n");
}

static int box_reverse(int position, int boundary, int pad, float scale)
{
    return (int)((clamp(position, 0, boundary) - pad) / scale);
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

#ifdef ENABLE_NEON
#include <arm_neon.h>

void convert_neon(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; i += 4) {
        float32x4_t f32 = vld1q_f32(src + i);
        float16x4_t f16 = vcvt_f16_f32(f32);
        vst1_u16(dst + i, (uint16x4_t)f16);
    }
}
#endif

static int process_i8(rknn_output *all_input, int input_id, int *anchor,
                      int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &segments,
                      float *proto, std::vector<float> &objProbs, std::vector<int> &classId,
                      float threshold, RknnCtx *nn_ctx)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    post_dbg_func("enter\n");

    if (input_id % 2 == 1)
        return validCount;

    if (input_id == 6) { /* prototype masks */
        int8_t *input_proto = (int8_t *)all_input[input_id].buf;
        int32_t zp_proto = nn_ctx->output_attrs[input_id].zp;
        float scale_proto = nn_ctx->output_attrs[input_id].scale;
        int proto_num = PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT;
        TIMER timer;

        timer.tik();
        for (int i = 0; i < proto_num; i++)
            proto[i] = deqnt_affine_to_f32(input_proto[i], zp_proto, scale_proto);

#ifdef ENABLE_NEON
        /* convert fp32 to fp16 */
        convert_neon(proto, nn_ctx->vector_b, proto_num);
#endif
        post_dbg_time("0 - deqnt_affine_to_f32 + convert_neon");
        return validCount;
    }

    int8_t *input = (int8_t *)all_input[input_id].buf;
    int8_t *input_seg = (int8_t *)all_input[input_id + 1].buf;
    int32_t zp = nn_ctx->output_attrs[input_id].zp;
    float scale = nn_ctx->output_attrs[input_id].scale;
    int32_t zp_seg = nn_ctx->output_attrs[input_id + 1].zp;
    float scale_seg = nn_ctx->output_attrs[input_id + 1].scale;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    post_dbg_detail("thres_i8 %d\n", thres_i8);

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];

                if (box_confidence >= thres_i8) {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int offset_seg = (PROTO_CHANNEL * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    int8_t *in_ptr_seg = input_seg + offset_seg;

                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    post_dbg_detail("a %d i %d j %d box_confidence %d [%f %f %f %f]\n",
                                    a, i, j, box_confidence, box_x, box_y, box_w, box_h);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    post_dbg_detail("a %d i %d j %d maxClassProbs %d maxClassId %d\n",
                                    a, i, j, maxClassProbs, maxClassId);

                    float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                    float class_prob_f32 = deqnt_affine_to_f32(maxClassProbs, zp, scale);
                    float limit_score = box_conf_f32 * class_prob_f32;
                    // if (maxClassProbs > thres_i8)
                    if (limit_score > threshold)
                    {
                        post_dbg_detail("a %d i %d j %d box_conf %f class_prob %f limit_score %f threshold %f\n",
                                        a, i, j, box_conf_f32, class_prob_f32, limit_score, threshold);
                        for (int k = 0; k < PROTO_CHANNEL; k++)
                        {
                            float seg_element_fp = deqnt_affine_to_f32(in_ptr_seg[(k)*grid_len], zp_seg, scale_seg);
                            segments.push_back(seg_element_fp);
                        }

                        objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }

    post_dbg_func("exit\n");

    return validCount;
}

static int process_fp32(rknn_output *all_input, int input_id, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                        std::vector<float> &boxes, std::vector<float> &segments, float *proto, std::vector<float> &objProbs, std::vector<int> &classId, float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    post_dbg_func("enter\n");

    if (input_id % 2 == 1)
    {
        return validCount;
    }

    if (input_id == 6)
    {
        float *input_proto = (float *)all_input[input_id].buf;
        for (int i = 0; i < PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT; i++)
        {
            proto[i] = input_proto[i];
        }
        return validCount;
    }

    float *input = (float *)all_input[input_id].buf;
    float *input_seg = (float *)all_input[input_id + 1].buf;

    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                float box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= threshold)
                {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int offset_seg = (PROTO_CHANNEL * a) * grid_len + i * grid_w + j;
                    float *in_ptr = input + offset;
                    float *in_ptr_seg = input_seg + offset_seg;

                    float box_x = *in_ptr * 2.0 - 0.5;
                    float box_y = in_ptr[grid_len] * 2.0 - 0.5;
                    float box_w = in_ptr[2 * grid_len] * 2.0;
                    float box_h = in_ptr[3 * grid_len] * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    float maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        float prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float limit_score = maxClassProbs * box_confidence;
                    // if (maxClassProbs > threshold)
                    if (limit_score > threshold)
                    {
                        for (int k = 0; k < PROTO_CHANNEL; k++)
                        {
                            float seg_element_f32 = in_ptr_seg[(k)*grid_len];
                            segments.push_back(seg_element_f32);
                        }

                        objProbs.push_back(maxClassProbs * box_confidence);
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }

    post_dbg_func("exit\n");

    return validCount;
}

static int check_unwanted_class_id(int cls_id, int scene_mode)
{
    static LabelName label_names[2][5] = {
        { LABEL_PERSON, LABEL_BOOK, LABEL_BUTT, LABEL_BUTT, LABEL_BUTT },
        { LABEL_PERSON, LABEL_BICYCLE, LABEL_CAR, LABEL_MOTORCYCLE, LABEL_BUS }
    };
    int unwanted_flag = 1;

    for (int k = 0; k < 5; k++)
        unwanted_flag &= (cls_id != label_names[scene_mode][k]);

    return unwanted_flag;
}

static int calc_matrix_multiply(RknnCtx *nn_ctx, std::vector<float> &filterSegments_by_nms, int boxes_num)
{
    int ret = 0;
    TIMER timer;
    int ROWS_A = boxes_num;
    int COLS_A = PROTO_CHANNEL;
    int COLS_B = PROTO_HEIGHT * PROTO_WEIGHT;

    timer.tik();
#ifdef USE_FP_RESIZE
    memset(nn_ctx->matmul_out, 0, boxes_num * PROTO_HEIGHT * PROTO_WEIGHT * sizeof(float));

    if (boxes_num > 0) {
#ifdef ENABLE_MATMUL_OPT
        matmul_by_npu_fp_optimized(filterSegments_by_nms, nn_ctx->matmul_out, ROWS_A, COLS_A, COLS_B, nn_ctx);
#else
        matmul_by_npu_fp(filterSegments_by_nms, nn_ctx->proto, nn_ctx->matmul_out, ROWS_A, COLS_A, COLS_B, nn_ctx);
#endif
    } else
        mpp_log("boxes_num(%d) <= 0, no need to do matmul\n", boxes_num);
    post_dbg_time("3 - matmul_by_cpu_fp/optimized");
#else
    memset(nn_ctx->matmul_out, 0, boxes_num * PROTO_HEIGHT * PROTO_WEIGHT * sizeof(uint8_t));
    matmul_by_cpu_uint8(filterSegments_by_nms, nn_ctx->proto, nn_ctx->matmul_out, ROWS_A, COLS_A, COLS_B); //TODO:fix
    post_dbg_time("3 - matmul_by_cpu_uint8");
#endif

    return ret;
}

int calc_instance_mask(RknnCtx *nn_ctx, rknn_output *outputs,
                       letterbox_t *letter_box, float conf_threshold, float nms_threshold,
                       object_detect_result_list *od_results)
{
    int ret = 0;
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    std::vector<float> filterSegments;
    std::vector<float> filterSegments_by_nms;

    int model_in_width = nn_ctx->model_width;
    int model_in_height = nn_ctx->model_height;

    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    TIMER timer;

    post_dbg_func("enter\n");

    timer.tik();
    // process the outputs of rknn
    for (int i = 0; i < 7; i++) {
        grid_h = nn_ctx->output_attrs[i].dims[2];
        grid_w = nn_ctx->output_attrs[i].dims[3];
        stride = model_in_height / grid_h;

        post_dbg_detail("idx %d grid_h %d grid_w %d stride %d\n", i, grid_h, grid_w, stride);

        if (nn_ctx->is_quant) {
            validCount += process_i8(outputs, i, (int *)anchor[i / 2], grid_h, grid_w,
                                     model_in_height, model_in_width, stride, filterBoxes,
                                     filterSegments, nn_ctx->proto, objProbs,
                                     classId, conf_threshold, nn_ctx);
            post_dbg_detail("idx %d validCount %d\n", i, validCount);
        } else {
            validCount += process_fp32(outputs, i, (int *)anchor[i / 2], grid_h, grid_w,
                                       model_in_height, model_in_width, stride, filterBoxes,
                                       filterSegments, nn_ctx->proto, objProbs, classId, conf_threshold);
        }
    }
    post_dbg_time("0 - process_i8");

    // nms
    if (validCount <= 0) {
        mpp_log("no valid detection results\n");
        return 0;
    }

    timer.tik();
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
        indexArray.push_back(i);

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    post_dbg_time("1 - quick_sort_indice_inverse");

    timer.tik();
    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }
    post_dbg_time("2 - nms");

    int last_count = 0;
    od_results->count = 0;

    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
            continue;

        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        if (check_unwanted_class_id(id, nn_ctx->scene_mode))
            continue;

        for (int k = 0; k < PROTO_CHANNEL; k++)
            filterSegments_by_nms.push_back(filterSegments[n * PROTO_CHANNEL + k]);

        od_results->results[last_count].box.left = x1;
        od_results->results[last_count].box.top = y1;
        od_results->results[last_count].box.right = x2;
        od_results->results[last_count].box.bottom = y2;

        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    int boxes_num = od_results->count;

    for (int i = 0; i < boxes_num; i++) {
        // for crop_mask
        nn_ctx->filterBoxes_by_nms[i * 4 + 0] = od_results->results[i].box.left;   // x1;
        nn_ctx->filterBoxes_by_nms[i * 4 + 1] = od_results->results[i].box.top;    // y1;
        nn_ctx->filterBoxes_by_nms[i * 4 + 2] = od_results->results[i].box.right;  // x2;
        nn_ctx->filterBoxes_by_nms[i * 4 + 3] = od_results->results[i].box.bottom; // y2;
        nn_ctx->cls_id[i] = od_results->results[i].cls_id;

        // get real box
        od_results->results[i].box.left = box_reverse(od_results->results[i].box.left, model_in_width, letter_box->x_pad, letter_box->scale);
        od_results->results[i].box.top = box_reverse(od_results->results[i].box.top, model_in_height, letter_box->y_pad, letter_box->scale);
        od_results->results[i].box.right = box_reverse(od_results->results[i].box.right, model_in_width, letter_box->x_pad, letter_box->scale);
        od_results->results[i].box.bottom = box_reverse(od_results->results[i].box.bottom, model_in_height, letter_box->y_pad, letter_box->scale);
        post_dbg_detect_rect("idx %d box [%d %d %d %d] cls_id %d prop %f\n",
                             i, od_results->results[i].box.left, od_results->results[i].box.top,
                             od_results->results[i].box.right, od_results->results[i].box.bottom,
                             od_results->results[i].cls_id, od_results->results[i].prop);
    }

    if (nn_ctx->segmap_calc_en)
        ret = calc_matrix_multiply(nn_ctx, filterSegments_by_nms, boxes_num);

    post_dbg_func("exit\n");
    return ret;
}

int trans_detect_result(RknnCtx *nn_ctx, letterbox_t *letter_box, object_detect_result_list *od_results)
{
    int ret = 0;
    TIMER timer;
    int boxes_num = od_results->count;
    int model_in_width = nn_ctx->model_width;
    int model_in_height = nn_ctx->model_height;

    memset(nn_ctx->all_mask_in_one, 0, model_in_height * model_in_width * sizeof(uint8_t));

#if defined(USE_FP_RESIZE) && defined(USE_FP_MASK_MAP)
    timer.tik();
    // resize to (boxes_num, model_in_width, model_in_height)
    resize_by_opencv_fp(nn_ctx->matmul_out, PROTO_WEIGHT, PROTO_HEIGHT, boxes_num,
                        nn_ctx->seg_mask, model_in_width, model_in_height);
    post_dbg_time("4 - resize_by_opencv_fp");

    timer.tik();
    crop_mask_fp(nn_ctx->seg_mask, nn_ctx->all_mask_in_one,
                 nn_ctx->filterBoxes_by_nms,
                 boxes_num, nn_ctx->cls_id,
                 model_in_height, model_in_width, nn_ctx->scene_mode);
    post_dbg_time("4 - crop_mask_fp");
#else
    timer.tik();
#ifdef ENABLE_MASK_MERGE
    resize_by_opencv_uint8(nn_ctx->matmul_out, PROTO_WEIGHT, PROTO_HEIGHT, 1,
        nn_ctx->seg_mask, model_in_width, model_in_height);
#else
    resize_by_opencv_uint8(nn_ctx->matmul_out, PROTO_WEIGHT, PROTO_HEIGHT, boxes_num,
                           nn_ctx->seg_mask, model_in_width, model_in_height);
#endif
    post_dbg_time("4 - resize_by_opencv_uint8");

    timer.tik();

#ifdef ENABLE_MASK_MERGE
    crop_mask_uint8_merge(nn_ctx->seg_mask, nn_ctx->all_mask_in_one,
                          nn_ctx->filterBoxes_by_nms,
                          boxes_num, nn_ctx->cls_id,
                          model_in_height, model_in_width, nn_ctx->scene_mode);
#else
    crop_mask_uint8(nn_ctx->seg_mask, nn_ctx->all_mask_in_one,
                    nn_ctx->filterBoxes_by_nms,
                    boxes_num, nn_ctx->cls_id,
                    model_in_height, model_in_width, nn_ctx->scene_mode);
#endif
    post_dbg_time("4 - crop_mask_uint8");
#endif

    // get real mask
    int cropped_height = model_in_height - letter_box->y_pad * 2;
    int cropped_width = model_in_width - letter_box->x_pad * 2;
    int ori_in_height = nn_ctx->input_image_height;
    int ori_in_width = nn_ctx->input_image_width;
    int y_pad = letter_box->y_pad;
    int x_pad = letter_box->x_pad;
    uint8_t *real_seg_mask;

    post_dbg_mask("x_pad %d y_pad %d cropped_height %d cropped_width %d\n",
                  x_pad, y_pad, cropped_height, cropped_width);

    if (nn_ctx->pre_alloc_mask) {
        real_seg_mask = nn_ctx->real_seg_mask;
    } else {
        real_seg_mask = (uint8_t *)malloc(ori_in_height * ori_in_width * sizeof(uint8_t));
        if (real_seg_mask == NULL) {
            mpp_err_f("malloc real_seg_mask failed\n");
            return -1;
        }
    }
    od_results->results_seg[0].seg_mask = real_seg_mask;

    timer.tik();
    seg_reverse(nn_ctx->all_mask_in_one, nn_ctx->cropped_seg_mask, real_seg_mask,
                model_in_height, model_in_width, cropped_height, cropped_width,
                ori_in_height, ori_in_width, y_pad, x_pad);
    post_dbg_time("4 - seg_reverse");

    post_dbg_func("exit\n");
    return ret;
}

int init_post_process(RknnCtx *nn_ctx)
{
    int ret = 0;
    TIMER timer;

    if (nn_ctx->run_type <= 2) /* display rectangles for jpeg */
        post_debug = POST_DBG_DETECT_RECT;

    nn_ctx->proto = (float *)calloc(1, PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT * sizeof(float));
    if (!nn_ctx->proto) {
        mpp_err_f("malloc nn_ctx->proto failed!\n");
        return -1;
    }

    nn_ctx->vector_b = (uint16_t *)calloc(1, PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT * sizeof(uint16_t));
    if (!nn_ctx->vector_b) {
        mpp_err_f("malloc nn_ctx->vector_b failed!\n");
        return -1;
    }

#if defined(USE_FP_RESIZE) && defined(USE_FP_MASK_MAP)
    nn_ctx->matmul_out = (float *)calloc(1, OBJ_NUMB_MAX_SIZE * PROTO_HEIGHT * PROTO_WEIGHT * sizeof(float));
#else
    nn_ctx->matmul_out = (uint8_t *)calloc(1, OBJ_NUMB_MAX_SIZE * PROTO_HEIGHT * PROTO_WEIGHT * sizeof(uint8_t));
#endif
    if (!nn_ctx->matmul_out) {
        mpp_err_f("malloc nn_ctx->matmul_out failed!\n");
        return -1;
    }

#ifdef ENABLE_MATMUL_OPT
    {
        rknn_matmul_info info;
        int max_size_a = 0, max_size_b = 0, max_size_c = 0;

        memset(nn_ctx->io_attr, 0, sizeof(rknn_matmul_io_attr) * OBJ_NUMB_MAX_SIZE);
        memset(&info, 0, sizeof(rknn_matmul_info));

        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NORM;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;

        timer.tik();
        for (int i = 0; i < OBJ_NUMB_MAX_SIZE; ++i) {
            nn_ctx->shapes[i].M = i + 1;
            nn_ctx->shapes[i].K = PROTO_CHANNEL;
            nn_ctx->shapes[i].N = PROTO_HEIGHT * PROTO_WEIGHT;
        }

        ret = rknn_matmul_create_dynamic_shape(&nn_ctx->matmul_ctx, &info,
                                               OBJ_NUMB_MAX_SIZE, nn_ctx->shapes, nn_ctx->io_attr);
        if (ret < 0) {
            mpp_log("rknn_matmul_create_dynamic_shape fail! ret=%d\n", ret);
            return -1;
        }

        for (int i = 0; i < OBJ_NUMB_MAX_SIZE; ++i) {
            max_size_a = MPP_MAX(nn_ctx->io_attr[i].A.size, max_size_a);
            max_size_b = MPP_MAX(nn_ctx->io_attr[i].B.size, max_size_b);
            max_size_c = MPP_MAX(nn_ctx->io_attr[i].C.size, max_size_c);
        }
        post_dbg_matrix("tensor_a max size %d tensor_b max size %d tensor_c max size %d\n",
                        max_size_a, max_size_b, max_size_c);

        nn_ctx->tensor_a = rknn_create_mem(nn_ctx->matmul_ctx, max_size_a);
        if (!nn_ctx->tensor_a) {
            mpp_err_f("rknn_create_mem tensor_a fail!\n");
            return -1;
        }

        nn_ctx->tensor_b = rknn_create_mem(nn_ctx->matmul_ctx, max_size_b);
        if (!nn_ctx->tensor_b) {
            mpp_err_f("rknn_create_mem tensor_b fail!\n");
            return -1;
        }

        nn_ctx->tensor_c = rknn_create_mem(nn_ctx->matmul_ctx, max_size_c);
        if (!nn_ctx->tensor_c) {
            mpp_err_f("rknn_create_mem tensor_c fail!\n");
            return -1;
        }
        post_dbg_matrix("tensor_a size %d tensor_b size %d tensor_c size %d\n",
                        nn_ctx->tensor_a->size, nn_ctx->tensor_b->size, nn_ctx->tensor_c->size);
        post_dbg_time("rknn_matmul_create_dynamic_shape");
    }
#endif

#if defined(USE_FP_RESIZE) && defined(USE_FP_MASK_MAP)
    nn_ctx->seg_mask = (float *)calloc(1, OBJ_NUMB_MAX_SIZE * nn_ctx->model_width * nn_ctx->model_width * sizeof(float));
#else
    nn_ctx->seg_mask = (uint8_t *)calloc(1, OBJ_NUMB_MAX_SIZE * nn_ctx->model_width * nn_ctx->model_width * sizeof(uint8_t));
#endif
    if (!nn_ctx->seg_mask) {
        mpp_err_f("malloc nn_ctx->seg_mask failed!\n");
        return -1;
    }

    nn_ctx->all_mask_in_one = (uint8_t *)calloc(1, nn_ctx->model_height * nn_ctx->model_width * sizeof(uint8_t));
    if (!nn_ctx->all_mask_in_one) {
        mpp_err_f("malloc nn_ctx->all_mask_in_one failed!\n");
        return -1;
    }

    nn_ctx->cropped_seg_mask = (uint8_t *)calloc(1, nn_ctx->model_height * nn_ctx->model_width * sizeof(uint8_t));
    if (!nn_ctx->cropped_seg_mask) {
        mpp_err_f("malloc nn_ctx->cropped_seg_mask failed!\n");
        return -1;
    }

    if (nn_ctx->pre_alloc_mask) {
        nn_ctx->real_seg_mask = (uint8_t *)calloc(1, nn_ctx->input_image_height * nn_ctx->input_image_width * sizeof(uint8_t));
        if (!nn_ctx->real_seg_mask) {
            mpp_err_f("malloc nn_ctx->real_seg_mask failed!\n");
            return -1;
        }
    }

    return 0;
}

void deinit_post_process(RknnCtx *nn_ctx)
{
    TIMER timer;
    // for (int i = 0; i < OBJ_CLASS_NUM; i++)
    // {
    //     {
    //         free(labels[i]);
    //         labels[i] = nullptr;
    //     }
    // }

    if (nn_ctx->proto) {
        free(nn_ctx->proto);
        nn_ctx->proto = nullptr;
    }

    if (nn_ctx->vector_b) {
        free(nn_ctx->vector_b);
        nn_ctx->vector_b = nullptr;
    }

    if (nn_ctx->matmul_out) {
        free(nn_ctx->matmul_out);
        nn_ctx->matmul_out = nullptr;
    }

#ifdef ENABLE_MATMUL_OPT
    timer.tik();
    if (nn_ctx->tensor_a) {
        rknn_destroy_mem(nn_ctx->matmul_ctx, nn_ctx->tensor_a);
        nn_ctx->tensor_a = nullptr;
    }

    if (nn_ctx->tensor_b) {
        rknn_destroy_mem(nn_ctx->matmul_ctx, nn_ctx->tensor_b);
        nn_ctx->tensor_b = nullptr;
    }

    if (nn_ctx->tensor_c) {
        rknn_destroy_mem(nn_ctx->matmul_ctx, nn_ctx->tensor_c);
        nn_ctx->tensor_c = nullptr;
    }

    if (nn_ctx->matmul_ctx)
        rknn_matmul_destroy(nn_ctx->matmul_ctx);
    post_dbg_time("rknn_matmul_destroy");
#endif

    if (!nn_ctx->seg_mask) {
        free(nn_ctx->seg_mask);
        nn_ctx->seg_mask = nullptr;
    }

    if (!nn_ctx->all_mask_in_one) {
        free(nn_ctx->all_mask_in_one);
        nn_ctx->all_mask_in_one = nullptr;
    }

    if (!nn_ctx->cropped_seg_mask) {
        free(nn_ctx->cropped_seg_mask);
        nn_ctx->cropped_seg_mask = nullptr;
    }

    if (nn_ctx->pre_alloc_mask && nn_ctx->real_seg_mask) {
        free(nn_ctx->real_seg_mask);
        nn_ctx->real_seg_mask = nullptr;
    }
}
