#ifdef RKNN_V3

#include "postproc_yolov6_rknn3.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <set>
#include <vector>
#include "mpp_log.h"
#include "mpp_time.h"

/* Yolov6 postprocess function for RKNN v3 */
#define Y6R3P_DBG_FUNCTION             (0x00000001)
#define Y6R3P_DBG_DETAIL               (0x00000002)
#define Y6R3P_DBG_DETECT_RECT          (0x00000004)
#define Y6R3P_DBG_TIME                 (0x00000008)
#define Y6R3P_DBG_MATRICES             (0x00000010)
#define Y6R3P_DBG_MASK                 (0x00000020)

#define y6r3p_log(cond, fmt, ...)   do { if (cond) mpp_log_f(fmt, ## __VA_ARGS__); } while (0)
#define y6r3p_dbg(flag, fmt, ...)   y6r3p_log((y6r3p_debug & flag), fmt, ## __VA_ARGS__)
#define y6r3p_dbg_func(fmt, ...)    y6r3p_dbg(Y6R3P_DBG_FUNCTION, fmt, ## __VA_ARGS__)
#define y6r3p_dbg_detail(fmt, ...)   y6r3p_dbg(Y6R3P_DBG_DETAIL, fmt, ## __VA_ARGS__)
#define y6r3p_dbg_detect_rect(fmt, ...)   y6r3p_dbg(Y6R3P_DBG_DETECT_RECT, fmt, ## __VA_ARGS__)
#define y6r3p_dbg_time(fmt, ...)    y6r3p_dbg(Y6R3P_DBG_TIME, fmt, ## __VA_ARGS__)

RK_S32 y6r3p_debug = 0;

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }


static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    y6r3p_dbg_func("enter\n");
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    y6r3p_dbg_func("enter\n");

    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
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
    y6r3p_dbg_func("exit\n");
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    y6r3p_dbg_func("enter\n");

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
    y6r3p_dbg_func("exit\n");
    return low;
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

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl(float* tensor, int dfl_len, float* box)
{
    y6r3p_dbg_func("enter\n");
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }

        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
    y6r3p_dbg_func("exit\n");
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int grid_len = grid_h * grid_w;
    int validCount = 0;

    y6r3p_dbg_func("enter\n");
    y6r3p_dbg_detail("grid_len %d sum_thres %d, score_thres %d\n", grid_len, score_sum_thres_i8, score_thres_i8);

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            int base_ofst = offset;
            int cur_ofst;
            for (int c = 0; c < sizeof(label_names[0]) / sizeof(label_names[0][0]); c++) {
                int ci = label_names[1][c]; //TODO: scene mode
                cur_ofst = base_ofst + grid_len * ci;

                if ((score_tensor[cur_ofst] > score_thres_i8) && (score_tensor[cur_ofst] > max_score)) {
                    max_score = score_tensor[cur_ofst];
                    max_class_id = ci;
                }
            }

            // compute box
            if (max_score> score_thres_i8) {
                offset = i* grid_w + j;
                float box[4];
                if (dfl_len > 1) {
                    float before_dfl[dfl_len * 4];
                    for (int k = 0; k < dfl_len * 4; k++) {
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);
                } else {
                    for (int k = 0; k < 4; k++) {
                        box[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                }

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    y6r3p_dbg_func("exit\n");
    return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    y6r3p_dbg_func("enter\n");

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                if (dfl_len>1){
                    /// dfl
                    float before_dfl[dfl_len*4];
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = box_tensor[offset];
                        offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);
                }
                else {
                    for (int k=0; k< 4; k++){
                        box[k] = box_tensor[offset];
                        offset += grid_len;
                    }
                }
                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    y6r3p_dbg_func("exit\n");
    return validCount;
}

static int convert_fp16_to_fp32(const float16 *src, float *dst, int n_elems)
{
    y6r3p_dbg_func("enter\n");
    for (int i = 0; i < n_elems; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
    y6r3p_dbg_func("exit\n");
    return 0;
}

int post_process_after_exYoloPostProcess(Rknn3Ctx *nn3_ctx, void *outputs,
                                         rknn3_tensor_attr output_attrs,
                                         letterbox_t *letter_box,
                                         ObjDetRetLst *od_results)
{
    rknn3_tensor *_outputs = (rknn3_tensor *)outputs;
    int K = output_attrs.shape[1];
    int size = K*6;
    int model_in_w = nn3_ctx->model_width;
    int model_in_h = nn3_ctx->model_height;

    rknn3_tensor_type dtype = output_attrs.dtype;
    float outputfp32[size];
    y6r3p_dbg_func("enter\n");

    if (dtype == RKNN3_TENSOR_FLOAT16) {
        float16 *output_tensor = (float16 *)_outputs[0].mem->virt_addr;
        convert_fp16_to_fp32(output_tensor, outputfp32, size);
    } else if(dtype == RKNN3_TENSOR_FLOAT32) {
        memcpy(outputfp32, _outputs[0].mem->virt_addr, size*sizeof(float));
    } else {
        return -1;
    }

    int last_count = 0;
    for(int i = 0; i < K; i++) {
        float score = outputfp32[i*6];
        if(score > 0) {
            if (last_count >= OBJ_NUM_MAX) {
                break;
            }
            float x1 = outputfp32[i*6 + 2] - letter_box->x_pad;
            float y1 = outputfp32[i*6 + 3] - letter_box->y_pad;
            float x2 = outputfp32[i*6 + 4] - letter_box->x_pad;
            float y2 = outputfp32[i*6 + 5] - letter_box->y_pad;


            int id = outputfp32[i*6 + 1];
            float obj_conf = score;

            od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].prop = obj_conf;
            od_results->results[last_count].cls_id = id;
            last_count++;
        }
    }
    od_results->count = last_count;
    y6r3p_dbg_func("exit\n");
    return 0;
}

static int getTensorSize(rknn3_tensor tensor)
{
    y6r3p_dbg_func("enter\n");
    int size = 0;
    for (int i = 0; i < tensor.attr->n_dims; i++) {
        if (i == 0)
            size = tensor.attr->shape[i];
        else
            size *= tensor.attr->shape[i];
    }
    y6r3p_dbg_func("exit\n");
    return size;
}

int postproc_yolov6_rknn3(Rknn3Ctx *nn3_ctx, void *outputs, letterbox_t *letter_box,
                          float conf_threshold, float nms_threshold,
                          ObjDetRetLst *od_results)
{
    rknn3_tensor *_outputs = (rknn3_tensor *)outputs;
    int model_in_w = nn3_ctx->model_width;
    int model_in_h = nn3_ctx->model_height;
    RK_S64 time_start, time_end;
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;

    memset(od_results, 0, sizeof(ObjDetRetLst));
    y6r3p_dbg_func("enter\n");

    time_start = mpp_time();
    // default 3 branch
    int dfl_len = nn3_ctx->outputs[0].attr->shape[1] / 4;
    int output_per_branch = nn3_ctx->io_num.n_output / 3;
    for (int i = 0; i < 3; i++) {
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;
        int sum_idx = i * output_per_branch + 2;

        rknn3_tensor_type dtype = _outputs[box_idx].attr->dtype;
        grid_h = nn3_ctx->outputs[box_idx].attr->shape[2];
        grid_w = nn3_ctx->outputs[box_idx].attr->shape[3];
        stride = model_in_h / grid_h;

        y6r3p_dbg_detail("i %d grid_h %d grid_w %d stride %d dfl_len %d\n",
                         i, grid_h, grid_w, stride, dfl_len);

        if (nn3_ctx->is_quant) {
            void *score_sum = nullptr;
            int32_t score_sum_zp = 0;
            float score_sum_scale = 1.0;
            if (output_per_branch == 3) {
                score_sum = _outputs[sum_idx].mem->virt_addr;
                score_sum_zp = nn3_ctx->outputs[sum_idx].attr->qnt_info.zero_point;
                score_sum_scale = nn3_ctx->outputs[sum_idx].attr->qnt_info.scale;
            }

            validCount += process_i8((int8_t *)_outputs[box_idx].mem->virt_addr,
                                     nn3_ctx->outputs[box_idx].attr->qnt_info.zero_point,
                                     nn3_ctx->outputs[box_idx].attr->qnt_info.scale,
                                     (int8_t *)_outputs[score_idx].mem->virt_addr,
                                     nn3_ctx->outputs[score_idx].attr->qnt_info.zero_point,
                                     nn3_ctx->outputs[score_idx].attr->qnt_info.scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, conf_threshold);
        } else {
            float *box_tensor = nullptr;
            float *score_tensor = nullptr;
            float *score_sum_tensor = nullptr;
            bool box_tensor_allocated = false;
            bool score_tensor_allocated = false;
            bool score_sum_tensor_allocated = false;

            //convert box from fp16 to fp32
            if(_outputs[box_idx].attr->dtype == RKNN3_TENSOR_FLOAT16) {
                int size = getTensorSize(_outputs[box_idx]);
                box_tensor = (float*) malloc(size * sizeof(float));
                box_tensor_allocated = true;
                convert_fp16_to_fp32((float16*)_outputs[box_idx].mem->virt_addr, box_tensor, size);
            } else {
                box_tensor = (float *)_outputs[box_idx].mem->virt_addr;
            }

            //convert score from fp16 to fp32
            if(_outputs[score_idx].attr->dtype == RKNN3_TENSOR_FLOAT16) {
                int size = getTensorSize(_outputs[score_idx]);
                score_tensor = (float*) malloc(size * sizeof(float));
                score_tensor_allocated = true;
                convert_fp16_to_fp32((float16*)_outputs[score_idx].mem->virt_addr, score_tensor, size);
            } else {
                score_tensor = (float *)_outputs[score_idx].mem->virt_addr;
            }

            //convert score_sum from fp16 to fp32
            if (output_per_branch == 3) {
                if(_outputs[sum_idx].attr->dtype == RKNN3_TENSOR_FLOAT16) {
                    int size = getTensorSize(_outputs[sum_idx]);
                    score_sum_tensor = (float*) malloc(size * sizeof(float));
                    score_sum_tensor_allocated = true;
                    convert_fp16_to_fp32((float16*)_outputs[sum_idx].mem->virt_addr, score_sum_tensor, size);
                } else {
                    score_sum_tensor = (float *)_outputs[sum_idx].mem->virt_addr;
                }
            }

            validCount += process_fp32(box_tensor, score_tensor, score_sum_tensor,
                                       grid_h, grid_w, stride, dfl_len,
                                       filterBoxes, objProbs, classId, conf_threshold);
            if (box_tensor_allocated)
                free(box_tensor);
            if (score_tensor_allocated)
                free(score_tensor);
            if (score_sum_tensor_allocated)
                free(score_sum_tensor);
        }
    }

    time_end = mpp_time();
    y6r3p_dbg_time("process_i8 elapsed %0.2f ms\n", (float)(time_end - time_start) / 1000);

    // no object detect
    if (validCount <= 0) {
        mpp_log("no valid detection results\n");
        od_results->count = 0;
        return 0;
    }

    time_start = mpp_time();
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }
    time_end = mpp_time();
    y6r3p_dbg_time("sort and nms elapsed %0.2f ms\n", (float)(time_end - time_start) / 1000);

    int last_count = 0;
    od_results->count = 0;

    time_start = mpp_time();
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUM_MAX) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;

        y6r3p_dbg_detect_rect("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(id),
                od_results->results[last_count].box.left, od_results->results[last_count].box.top,
                od_results->results[last_count].box.right, od_results->results[last_count].box.bottom,
                od_results->results[last_count].prop);
        last_count++;
    }
    od_results->count = last_count;

    time_end = mpp_time();
    y6r3p_dbg_time("valid box detection elapsed %0.2f ms\n", (float)(time_end - time_start) / 1000);
    y6r3p_dbg_func("exit\n");
    return 0;
}

int init_postproc_yolov6_rknn3(Rknn3Ctx *nn3_ctx)
{
    y6r3p_dbg_func("enter\n");
    int ret = 0;
    ret = loadLabelName(LABEL_NAME_TXT_PATH, labels);
    if (ret < 0) {
        mpp_err_f("Load %s failed!\n", LABEL_NAME_TXT_PATH);
        return -1;
    }
    y6r3p_dbg_func("exit\n");
    return 0;
}

void deinit_postproc_yolov6_rknn3(Rknn3Ctx *nn3_ctx)
{
    y6r3p_dbg_func("enter\n");
    for (int i = 0; i < OBJ_CLASS_NUM; i++) {
        if (labels[i] != nullptr) {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
    y6r3p_dbg_func("exit\n");
}
#endif