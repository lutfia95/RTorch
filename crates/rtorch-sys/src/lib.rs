use std::os::raw::{c_char, c_int};

pub const RT_STATUS_OK: c_int = 0;
pub const RT_STATUS_END: c_int = 1;
pub const RT_STATUS_ERR: c_int = 2;

#[repr(C)]
pub struct RtTransformPipeline {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RtTensorDataset {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RtDataLoader {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RtOwnedBatch {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RtStatistics {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub population_std: f64,
    pub sample_std: f64,
    pub min: f64,
    pub max: f64,
}

unsafe extern "C" {
    pub fn rt_last_error_message() -> *mut c_char;
    pub fn rt_last_error_free(message: *mut c_char);
    pub fn rt_statistics_f32(
        data: *const f32,
        len: usize,
        out: *mut RtStatistics,
    ) -> bool;

    pub fn rt_transform_pipeline_new(
        to_float: bool,
        resize_h: i64,
        resize_w: i64,
        crop_h: i64,
        crop_w: i64,
        mean: *const f32,
        mean_len: usize,
        std_values: *const f32,
        std_values_len: usize,
    ) -> *mut RtTransformPipeline;
    pub fn rt_transform_pipeline_free(pipeline: *mut RtTransformPipeline);

    pub fn rt_tensor_dataset_new(root: *const c_char) -> *mut RtTensorDataset;
    pub fn rt_tensor_dataset_free(dataset: *mut RtTensorDataset);
    pub fn rt_tensor_dataset_len(dataset: *const RtTensorDataset) -> usize;
    pub fn rt_tensor_dataset_class_count(dataset: *const RtTensorDataset) -> usize;
    pub fn rt_tensor_dataset_class_name(
        dataset: *const RtTensorDataset,
        class_index: usize,
    ) -> *const c_char;

    pub fn rt_data_loader_new(
        dataset: *mut RtTensorDataset,
        pipeline: *mut RtTransformPipeline,
        batch_size: usize,
        shuffle: bool,
    ) -> *mut RtDataLoader;
    pub fn rt_data_loader_free(loader: *mut RtDataLoader);
    pub fn rt_data_loader_reset(loader: *mut RtDataLoader) -> bool;
    pub fn rt_data_loader_next(loader: *mut RtDataLoader, out: *mut *mut RtOwnedBatch) -> c_int;

    pub fn rt_owned_batch_data(batch: *const RtOwnedBatch) -> *const f32;
    pub fn rt_owned_batch_data_len(batch: *const RtOwnedBatch) -> usize;
    pub fn rt_owned_batch_shape(batch: *const RtOwnedBatch) -> *const i64;
    pub fn rt_owned_batch_shape_len(batch: *const RtOwnedBatch) -> usize;
    pub fn rt_owned_batch_labels(batch: *const RtOwnedBatch) -> *const i64;
    pub fn rt_owned_batch_labels_len(batch: *const RtOwnedBatch) -> usize;
    pub fn rt_owned_batch_statistics(
        batch: *const RtOwnedBatch,
        out: *mut RtStatistics,
    ) -> bool;
    pub fn rt_owned_batch_free(batch: *mut RtOwnedBatch);
}
