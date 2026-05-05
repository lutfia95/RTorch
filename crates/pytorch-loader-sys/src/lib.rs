use std::os::raw::{c_char, c_int};

pub const PTL_STATUS_OK: c_int = 0;
pub const PTL_STATUS_END: c_int = 1;
pub const PTL_STATUS_ERR: c_int = 2;

#[repr(C)]
pub struct PtlTransformPipeline {
    _private: [u8; 0],
}

#[repr(C)]
pub struct PtlTensorDataset {
    _private: [u8; 0],
}

#[repr(C)]
pub struct PtlDataLoader {
    _private: [u8; 0],
}

#[repr(C)]
pub struct PtlOwnedBatch {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn ptl_last_error_message() -> *const c_char;
    pub fn ptl_last_error_free(message: *mut c_char);

    pub fn ptl_transform_pipeline_new(
        to_float: bool,
        resize_h: i64,
        resize_w: i64,
        crop_h: i64,
        crop_w: i64,
        mean: *const f32,
        mean_len: usize,
        std: *const f32,
        std_len: usize,
    ) -> *mut PtlTransformPipeline;
    pub fn ptl_transform_pipeline_free(pipeline: *mut PtlTransformPipeline);

    pub fn ptl_tensor_dataset_new(root: *const c_char) -> *mut PtlTensorDataset;
    pub fn ptl_tensor_dataset_free(dataset: *mut PtlTensorDataset);
    pub fn ptl_tensor_dataset_len(dataset: *const PtlTensorDataset) -> usize;
    pub fn ptl_tensor_dataset_class_count(dataset: *const PtlTensorDataset) -> usize;
    pub fn ptl_tensor_dataset_class_name(
        dataset: *const PtlTensorDataset,
        class_index: usize,
    ) -> *const c_char;

    pub fn ptl_data_loader_new(
        dataset: *mut PtlTensorDataset,
        pipeline: *mut PtlTransformPipeline,
        batch_size: usize,
        shuffle: bool,
    ) -> *mut PtlDataLoader;
    pub fn ptl_data_loader_free(loader: *mut PtlDataLoader);
    pub fn ptl_data_loader_reset(loader: *mut PtlDataLoader) -> bool;
    pub fn ptl_data_loader_next(loader: *mut PtlDataLoader, out: *mut *mut PtlOwnedBatch) -> c_int;

    pub fn ptl_owned_batch_data(batch: *const PtlOwnedBatch) -> *const f32;
    pub fn ptl_owned_batch_data_len(batch: *const PtlOwnedBatch) -> usize;
    pub fn ptl_owned_batch_shape(batch: *const PtlOwnedBatch) -> *const i64;
    pub fn ptl_owned_batch_shape_len(batch: *const PtlOwnedBatch) -> usize;
    pub fn ptl_owned_batch_labels(batch: *const PtlOwnedBatch) -> *const i64;
    pub fn ptl_owned_batch_labels_len(batch: *const PtlOwnedBatch) -> usize;
    pub fn ptl_owned_batch_free(batch: *mut PtlOwnedBatch);
}
