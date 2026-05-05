#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  PTL_STATUS_OK = 0,
  PTL_STATUS_END = 1,
  PTL_STATUS_ERR = 2,
};

typedef struct PtlTransformPipeline PtlTransformPipeline;
typedef struct PtlTensorDataset PtlTensorDataset;
typedef struct PtlDataLoader PtlDataLoader;
typedef struct PtlOwnedBatch PtlOwnedBatch;

const char* ptl_last_error_message(void);
void ptl_last_error_free(char* message);

PtlTransformPipeline* ptl_transform_pipeline_new(
    bool to_float,
    int64_t resize_h,
    int64_t resize_w,
    int64_t crop_h,
    int64_t crop_w,
    const float* mean,
    size_t mean_len,
    const float* std,
    size_t std_len);
void ptl_transform_pipeline_free(PtlTransformPipeline* pipeline);

PtlTensorDataset* ptl_tensor_dataset_new(const char* root);
void ptl_tensor_dataset_free(PtlTensorDataset* dataset);
size_t ptl_tensor_dataset_len(const PtlTensorDataset* dataset);
size_t ptl_tensor_dataset_class_count(const PtlTensorDataset* dataset);
const char* ptl_tensor_dataset_class_name(const PtlTensorDataset* dataset, size_t class_index);

PtlDataLoader* ptl_data_loader_new(
    PtlTensorDataset* dataset,
    PtlTransformPipeline* pipeline,
    size_t batch_size,
    bool shuffle);
void ptl_data_loader_free(PtlDataLoader* loader);
bool ptl_data_loader_reset(PtlDataLoader* loader);
int ptl_data_loader_next(PtlDataLoader* loader, PtlOwnedBatch** out);

const float* ptl_owned_batch_data(const PtlOwnedBatch* batch);
size_t ptl_owned_batch_data_len(const PtlOwnedBatch* batch);
const int64_t* ptl_owned_batch_shape(const PtlOwnedBatch* batch);
size_t ptl_owned_batch_shape_len(const PtlOwnedBatch* batch);
const int64_t* ptl_owned_batch_labels(const PtlOwnedBatch* batch);
size_t ptl_owned_batch_labels_len(const PtlOwnedBatch* batch);
void ptl_owned_batch_free(PtlOwnedBatch* batch);

#ifdef __cplusplus
}
#endif
