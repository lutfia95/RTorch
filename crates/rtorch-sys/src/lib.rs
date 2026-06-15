use std::os::raw::{c_char, c_int};

pub const RT_STATUS_OK: c_int = 0;
pub const RT_STATUS_END: c_int = 1;
pub const RT_STATUS_ERR: c_int = 2;

pub const RT_SCALAR_BOOL: c_int = 0;
pub const RT_SCALAR_U8: c_int = 1;
pub const RT_SCALAR_I8: c_int = 2;
pub const RT_SCALAR_I16: c_int = 3;
pub const RT_SCALAR_I32: c_int = 4;
pub const RT_SCALAR_I64: c_int = 5;
pub const RT_SCALAR_U16: c_int = 6;
pub const RT_SCALAR_U32: c_int = 7;
pub const RT_SCALAR_U64: c_int = 8;
pub const RT_SCALAR_F16: c_int = 9;
pub const RT_SCALAR_BF16: c_int = 10;
pub const RT_SCALAR_F32: c_int = 11;
pub const RT_SCALAR_F64: c_int = 12;

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
pub struct RtTensor {
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
    pub fn rt_statistics(
        data: *const std::ffi::c_void,
        len: usize,
        scalar_type: c_int,
        out: *mut RtStatistics,
    ) -> bool;

    pub fn rt_tensor_from_data(
        data: *const std::ffi::c_void,
        len: usize,
        scalar_type: c_int,
        shape: *const i64,
        shape_len: usize,
        requires_grad: bool,
    ) -> *mut RtTensor;
    pub fn rt_tensor_zeros(
        shape: *const i64,
        shape_len: usize,
        scalar_type: c_int,
        requires_grad: bool,
    ) -> *mut RtTensor;
    pub fn rt_tensor_ones(
        shape: *const i64,
        shape_len: usize,
        scalar_type: c_int,
        requires_grad: bool,
    ) -> *mut RtTensor;
    pub fn rt_tensor_clone(tensor: *const RtTensor) -> *mut RtTensor;
    pub fn rt_tensor_free(tensor: *mut RtTensor);
    pub fn rt_tensor_numel(tensor: *const RtTensor) -> usize;
    pub fn rt_tensor_shape_len(tensor: *const RtTensor) -> usize;
    pub fn rt_tensor_copy_shape(tensor: *const RtTensor, out: *mut i64, out_len: usize) -> bool;
    pub fn rt_tensor_scalar_type(tensor: *const RtTensor) -> c_int;
    pub fn rt_tensor_requires_grad(tensor: *const RtTensor) -> bool;
    pub fn rt_tensor_set_requires_grad(tensor: *mut RtTensor, requires_grad: bool) -> bool;
    pub fn rt_tensor_copy_f32(tensor: *const RtTensor, out: *mut f32, out_len: usize) -> bool;
    pub fn rt_tensor_add(left: *const RtTensor, right: *const RtTensor) -> *mut RtTensor;
    pub fn rt_tensor_matmul(left: *const RtTensor, right: *const RtTensor) -> *mut RtTensor;
    pub fn rt_tensor_reshape(
        tensor: *const RtTensor,
        shape: *const i64,
        shape_len: usize,
    ) -> *mut RtTensor;
    pub fn rt_tensor_sum(tensor: *const RtTensor) -> *mut RtTensor;
    pub fn rt_tensor_backward(tensor: *mut RtTensor) -> bool;
    pub fn rt_tensor_grad(tensor: *const RtTensor) -> *mut RtTensor;

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
    pub fn rt_owned_batch_statistics(batch: *const RtOwnedBatch, out: *mut RtStatistics) -> bool;
    pub fn rt_owned_batch_tensor(batch: *const RtOwnedBatch) -> *mut RtTensor;
    pub fn rt_owned_batch_free(batch: *mut RtOwnedBatch);
}
