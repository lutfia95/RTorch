#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  RT_STATUS_OK = 0,
  RT_STATUS_END = 1,
  RT_STATUS_ERR = 2,
};

enum {
  RT_SCALAR_BOOL = 0,
  RT_SCALAR_U8 = 1,
  RT_SCALAR_I8 = 2,
  RT_SCALAR_I16 = 3,
  RT_SCALAR_I32 = 4,
  RT_SCALAR_I64 = 5,
  RT_SCALAR_U16 = 6,
  RT_SCALAR_U32 = 7,
  RT_SCALAR_U64 = 8,
  RT_SCALAR_F16 = 9,
  RT_SCALAR_BF16 = 10,
  RT_SCALAR_F32 = 11,
  RT_SCALAR_F64 = 12,
};

typedef struct RtTransformPipeline RtTransformPipeline;
typedef struct RtTensorDataset RtTensorDataset;
typedef struct RtDataLoader RtDataLoader;
typedef struct RtOwnedBatch RtOwnedBatch;
typedef struct RtTensor RtTensor;

typedef struct RtStatistics {
  size_t count;
  double sum;
  double mean;
  double population_std;
  double sample_std;
  double min;
  double max;
} RtStatistics;

char* rt_last_error_message(void);
void rt_last_error_free(char* message);
bool rt_statistics(
    const void* data,
    size_t len,
    int scalar_type,
    RtStatistics* out);

RtTensor* rt_tensor_from_data(
    const void* data,
    size_t len,
    int scalar_type,
    const int64_t* shape,
    size_t shape_len,
    bool requires_grad);
RtTensor* rt_tensor_zeros(
    const int64_t* shape,
    size_t shape_len,
    int scalar_type,
    bool requires_grad);
RtTensor* rt_tensor_ones(
    const int64_t* shape,
    size_t shape_len,
    int scalar_type,
    bool requires_grad);
RtTensor* rt_tensor_clone(const RtTensor* tensor);
void rt_tensor_free(RtTensor* tensor);
size_t rt_tensor_numel(const RtTensor* tensor);
size_t rt_tensor_shape_len(const RtTensor* tensor);
bool rt_tensor_copy_shape(
    const RtTensor* tensor,
    int64_t* out,
    size_t out_len);
int rt_tensor_scalar_type(const RtTensor* tensor);
bool rt_tensor_requires_grad(const RtTensor* tensor);
bool rt_tensor_set_requires_grad(RtTensor* tensor, bool requires_grad);
bool rt_tensor_copy_f32(
    const RtTensor* tensor,
    float* out,
    size_t out_len);
RtTensor* rt_tensor_add(const RtTensor* left, const RtTensor* right);
RtTensor* rt_tensor_matmul(const RtTensor* left, const RtTensor* right);
RtTensor* rt_tensor_reshape(
    const RtTensor* tensor,
    const int64_t* shape,
    size_t shape_len);
RtTensor* rt_tensor_sum(const RtTensor* tensor);
bool rt_tensor_backward(RtTensor* tensor);
RtTensor* rt_tensor_grad(const RtTensor* tensor);

RtTransformPipeline* rt_transform_pipeline_new(
    bool to_float,
    int64_t resize_h,
    int64_t resize_w,
    int64_t crop_h,
    int64_t crop_w,
    const float* mean,
    size_t mean_len,
    const float* std_values,
    size_t std_values_len);
void rt_transform_pipeline_free(RtTransformPipeline* pipeline);

RtTensorDataset* rt_tensor_dataset_new(const char* root);
void rt_tensor_dataset_free(RtTensorDataset* dataset);
size_t rt_tensor_dataset_len(const RtTensorDataset* dataset);
size_t rt_tensor_dataset_class_count(const RtTensorDataset* dataset);
const char* rt_tensor_dataset_class_name(const RtTensorDataset* dataset, size_t class_index);

RtDataLoader* rt_data_loader_new(
    RtTensorDataset* dataset,
    RtTransformPipeline* pipeline,
    size_t batch_size,
    bool shuffle);
void rt_data_loader_free(RtDataLoader* loader);
bool rt_data_loader_reset(RtDataLoader* loader);
int rt_data_loader_next(RtDataLoader* loader, RtOwnedBatch** out);

const float* rt_owned_batch_data(const RtOwnedBatch* batch);
size_t rt_owned_batch_data_len(const RtOwnedBatch* batch);
const int64_t* rt_owned_batch_shape(const RtOwnedBatch* batch);
size_t rt_owned_batch_shape_len(const RtOwnedBatch* batch);
const int64_t* rt_owned_batch_labels(const RtOwnedBatch* batch);
size_t rt_owned_batch_labels_len(const RtOwnedBatch* batch);
bool rt_owned_batch_statistics(const RtOwnedBatch* batch, RtStatistics* out);
RtTensor* rt_owned_batch_tensor(const RtOwnedBatch* batch);
void rt_owned_batch_free(RtOwnedBatch* batch);

#ifdef __cplusplus
}
#endif
