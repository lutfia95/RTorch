use std::ffi::{c_void, CString, NulError};
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::path::Path;

pub use half::{bf16, f16};
use rtorch_sys as sys;

#[derive(Debug)]
pub enum Error {
    InteriorNul(NulError),
    Backend(String),
    InvalidUtf8Path,
    ShapeOverflow,
    ShapeElementMismatch { expected: usize, actual: usize },
    UnsupportedScalarType(i32),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InteriorNul(err) => write!(f, "string contains interior NUL: {err}"),
            Self::Backend(msg) => write!(f, "{msg}"),
            Self::InvalidUtf8Path => write!(f, "path is not valid UTF-8"),
            Self::ShapeOverflow => write!(f, "tensor shape value overflowed usize"),
            Self::ShapeElementMismatch { expected, actual } => write!(
                f,
                "tensor shape requires {expected} elements but received {actual}"
            ),
            Self::UnsupportedScalarType(value) => {
                write!(f, "backend returned unsupported scalar type {value}")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<NulError> for Error {
    fn from(value: NulError) -> Self {
        Self::InteriorNul(value)
    }
}

fn path_to_cstring(path: &Path) -> Result<CString, Error> {
    let value = path.to_str().ok_or(Error::InvalidUtf8Path)?;
    Ok(CString::new(value)?)
}

fn take_last_error() -> Error {
    unsafe {
        let ptr = sys::rt_last_error_message();
        if ptr.is_null() {
            return Error::Backend("unknown backend error".to_string());
        }

        let message = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        sys::rt_last_error_free(ptr);
        Error::Backend(message)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TransformConfig {
    pub to_float: bool,
    pub resize: Option<(i64, i64)>,
    pub center_crop: Option<(i64, i64)>,
    pub normalize_mean: Option<Vec<f32>>,
    pub normalize_std: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Statistics {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub population_std: f64,
    pub sample_std: f64,
    pub min: f64,
    pub max: f64,
}

fn statistics_from_raw(raw: sys::RtStatistics) -> Statistics {
    Statistics {
        count: raw.count,
        sum: raw.sum,
        mean: raw.mean,
        population_std: raw.population_std,
        sample_std: raw.sample_std,
        min: raw.min,
        max: raw.max,
    }
}

mod private {
    pub trait Sealed {}
}

/// A Rust scalar type that RTorch can pass to LibTorch for statistics.
pub trait StatisticsElement: private::Sealed {
    #[doc(hidden)]
    const SCALAR_TYPE: i32;
}

/// A Rust scalar type that can be copied into an owned LibTorch tensor.
pub trait TensorElement: StatisticsElement {}

macro_rules! impl_statistics_element {
    ($type:ty, $scalar_type:expr) => {
        impl private::Sealed for $type {}

        impl StatisticsElement for $type {
            const SCALAR_TYPE: i32 = $scalar_type;
        }
    };
}

impl_statistics_element!(bool, sys::RT_SCALAR_BOOL);
impl_statistics_element!(u8, sys::RT_SCALAR_U8);
impl_statistics_element!(i8, sys::RT_SCALAR_I8);
impl_statistics_element!(i16, sys::RT_SCALAR_I16);
impl_statistics_element!(i32, sys::RT_SCALAR_I32);
impl_statistics_element!(i64, sys::RT_SCALAR_I64);
impl_statistics_element!(u16, sys::RT_SCALAR_U16);
impl_statistics_element!(u32, sys::RT_SCALAR_U32);
impl_statistics_element!(u64, sys::RT_SCALAR_U64);
impl_statistics_element!(f16, sys::RT_SCALAR_F16);
impl_statistics_element!(bf16, sys::RT_SCALAR_BF16);
impl_statistics_element!(f32, sys::RT_SCALAR_F32);
impl_statistics_element!(f64, sys::RT_SCALAR_F64);

#[cfg(target_pointer_width = "32")]
impl_statistics_element!(usize, sys::RT_SCALAR_U32);
#[cfg(target_pointer_width = "64")]
impl_statistics_element!(usize, sys::RT_SCALAR_U64);
#[cfg(target_pointer_width = "32")]
impl_statistics_element!(isize, sys::RT_SCALAR_I32);
#[cfg(target_pointer_width = "64")]
impl_statistics_element!(isize, sys::RT_SCALAR_I64);

impl<T: StatisticsElement> TensorElement for T {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum ScalarType {
    Bool = sys::RT_SCALAR_BOOL,
    U8 = sys::RT_SCALAR_U8,
    I8 = sys::RT_SCALAR_I8,
    I16 = sys::RT_SCALAR_I16,
    I32 = sys::RT_SCALAR_I32,
    I64 = sys::RT_SCALAR_I64,
    U16 = sys::RT_SCALAR_U16,
    U32 = sys::RT_SCALAR_U32,
    U64 = sys::RT_SCALAR_U64,
    F16 = sys::RT_SCALAR_F16,
    BF16 = sys::RT_SCALAR_BF16,
    F32 = sys::RT_SCALAR_F32,
    F64 = sys::RT_SCALAR_F64,
}

impl TryFrom<i32> for ScalarType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            sys::RT_SCALAR_BOOL => Ok(Self::Bool),
            sys::RT_SCALAR_U8 => Ok(Self::U8),
            sys::RT_SCALAR_I8 => Ok(Self::I8),
            sys::RT_SCALAR_I16 => Ok(Self::I16),
            sys::RT_SCALAR_I32 => Ok(Self::I32),
            sys::RT_SCALAR_I64 => Ok(Self::I64),
            sys::RT_SCALAR_U16 => Ok(Self::U16),
            sys::RT_SCALAR_U32 => Ok(Self::U32),
            sys::RT_SCALAR_U64 => Ok(Self::U64),
            sys::RT_SCALAR_F16 => Ok(Self::F16),
            sys::RT_SCALAR_BF16 => Ok(Self::BF16),
            sys::RT_SCALAR_F32 => Ok(Self::F32),
            sys::RT_SCALAR_F64 => Ok(Self::F64),
            _ => Err(Error::UnsupportedScalarType(value)),
        }
    }
}

fn checked_shape(shape: &[usize]) -> Result<(Vec<i64>, usize), Error> {
    let mut raw = Vec::with_capacity(shape.len());
    let mut count = 1usize;
    for dimension in shape {
        raw.push((*dimension).try_into().map_err(|_| Error::ShapeOverflow)?);
        count = count.checked_mul(*dimension).ok_or(Error::ShapeOverflow)?;
    }
    Ok((raw, count))
}

/// An owned LibTorch tensor that preserves dtype, shape, and autograd state.
pub struct Tensor {
    raw: *mut sys::RtTensor,
}

impl Tensor {
    fn from_raw(raw: *mut sys::RtTensor) -> Result<Self, Error> {
        if raw.is_null() {
            Err(take_last_error())
        } else {
            Ok(Self { raw })
        }
    }

    pub fn from_slice<T: TensorElement>(
        values: &[T],
        shape: &[usize],
        requires_grad: bool,
    ) -> Result<Self, Error> {
        let (raw_shape, expected) = checked_shape(shape)?;
        if expected != values.len() {
            return Err(Error::ShapeElementMismatch {
                expected,
                actual: values.len(),
            });
        }
        let data = if values.is_empty() {
            std::ptr::null()
        } else {
            values.as_ptr().cast()
        };
        Self::from_raw(unsafe {
            sys::rt_tensor_from_data(
                data,
                values.len(),
                T::SCALAR_TYPE,
                raw_shape.as_ptr(),
                raw_shape.len(),
                requires_grad,
            )
        })
    }

    /// Transfer a vector into tensor storage without copying its elements.
    /// LibTorch/DLPack views keep the allocation alive until their last owner
    /// is dropped. Storage cannot be resized by LibTorch.
    pub fn from_vec<T: TensorElement + Send + 'static>(
        mut values: Vec<T>,
        shape: &[usize],
    ) -> Result<Self, Error> {
        let (raw_shape, expected) = checked_shape(shape)?;
        if expected != values.len() {
            return Err(Error::ShapeElementMismatch {
                expected,
                actual: values.len(),
            });
        }
        // Builders grow geometrically while streaming. Do not retain unused
        // capacity for the lifetime of a returned tensor or DLPack view.
        values.shrink_to_fit();
        unsafe extern "C" fn release<T>(owner: *mut c_void) {
            // The backend calls this exactly once, including on construction errors.
            drop(unsafe { Box::from_raw(owner.cast::<Vec<T>>()) });
        }
        let mut owner = Box::new(values);
        let data = owner.as_mut_ptr().cast();
        let len = owner.len();
        let owner = Box::into_raw(owner).cast();
        Self::from_raw(unsafe {
            sys::rt_tensor_from_owned_data(
                data,
                len,
                T::SCALAR_TYPE,
                raw_shape.as_ptr(),
                raw_shape.len(),
                owner,
                release::<T>,
            )
        })
    }

    pub fn zeros(
        shape: &[usize],
        scalar_type: ScalarType,
        requires_grad: bool,
    ) -> Result<Self, Error> {
        let (raw_shape, _) = checked_shape(shape)?;
        Self::from_raw(unsafe {
            sys::rt_tensor_zeros(
                raw_shape.as_ptr(),
                raw_shape.len(),
                scalar_type as i32,
                requires_grad,
            )
        })
    }

    pub fn ones(
        shape: &[usize],
        scalar_type: ScalarType,
        requires_grad: bool,
    ) -> Result<Self, Error> {
        let (raw_shape, _) = checked_shape(shape)?;
        Self::from_raw(unsafe {
            sys::rt_tensor_ones(
                raw_shape.as_ptr(),
                raw_shape.len(),
                scalar_type as i32,
                requires_grad,
            )
        })
    }

    pub fn try_clone(&self) -> Result<Self, Error> {
        Self::from_raw(unsafe { sys::rt_tensor_clone(self.raw) })
    }

    pub fn numel(&self) -> usize {
        unsafe { sys::rt_tensor_numel(self.raw) }
    }

    pub fn shape(&self) -> Result<Vec<usize>, Error> {
        let len = unsafe { sys::rt_tensor_shape_len(self.raw) };
        let mut raw = vec![0_i64; len];
        let ok = unsafe { sys::rt_tensor_copy_shape(self.raw, raw.as_mut_ptr(), raw.len()) };
        if !ok {
            return Err(take_last_error());
        }
        raw.into_iter()
            .map(|value| value.try_into().map_err(|_| Error::ShapeOverflow))
            .collect()
    }

    pub fn scalar_type(&self) -> Result<ScalarType, Error> {
        ScalarType::try_from(unsafe { sys::rt_tensor_scalar_type(self.raw) })
    }

    pub fn requires_grad(&self) -> bool {
        unsafe { sys::rt_tensor_requires_grad(self.raw) }
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), Error> {
        if unsafe { sys::rt_tensor_set_requires_grad(self.raw, requires_grad) } {
            Ok(())
        } else {
            Err(take_last_error())
        }
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>, Error> {
        let mut values = vec![0.0_f32; self.numel()];
        let ok = unsafe { sys::rt_tensor_copy_f32(self.raw, values.as_mut_ptr(), values.len()) };
        if ok {
            Ok(values)
        } else {
            Err(take_last_error())
        }
    }

    pub fn to_vec_f64(&self) -> Result<Vec<f64>, Error> {
        let mut values = vec![0.0_f64; self.numel()];
        let ok = unsafe { sys::rt_tensor_copy_f64(self.raw, values.as_mut_ptr(), values.len()) };
        if ok {
            Ok(values)
        } else {
            Err(take_last_error())
        }
    }

    pub fn to_vec_i64(&self) -> Result<Vec<i64>, Error> {
        let mut values = vec![0_i64; self.numel()];
        let ok = unsafe { sys::rt_tensor_copy_i64(self.raw, values.as_mut_ptr(), values.len()) };
        if ok {
            Ok(values)
        } else {
            Err(take_last_error())
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        Self::from_raw(unsafe { sys::rt_tensor_add(self.raw, other.raw) })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, Error> {
        Self::from_raw(unsafe { sys::rt_tensor_matmul(self.raw, other.raw) })
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self, Error> {
        let (raw_shape, expected) = checked_shape(shape)?;
        if expected != self.numel() {
            return Err(Error::ShapeElementMismatch {
                expected,
                actual: self.numel(),
            });
        }
        Self::from_raw(unsafe {
            sys::rt_tensor_reshape(self.raw, raw_shape.as_ptr(), raw_shape.len())
        })
    }

    pub fn sum(&self) -> Result<Self, Error> {
        Self::from_raw(unsafe { sys::rt_tensor_sum(self.raw) })
    }

    pub fn backward(&mut self) -> Result<(), Error> {
        if unsafe { sys::rt_tensor_backward(self.raw) } {
            Ok(())
        } else {
            Err(take_last_error())
        }
    }

    pub fn grad(&self) -> Result<Self, Error> {
        Self::from_raw(unsafe { sys::rt_tensor_grad(self.raw) })
    }

    /// Exports this tensor as a DLPack `DLManagedTensor`.
    ///
    /// The returned pointer follows the DLPack ownership convention: exactly one
    /// consumer must either pass it to a DLPack consumer or release it with
    /// `free_dlpack_raw`.
    pub fn to_dlpack_raw(&self) -> Result<*mut c_void, Error> {
        let raw = unsafe { sys::rt_tensor_to_dlpack(self.raw) };
        if raw.is_null() {
            Err(take_last_error())
        } else {
            Ok(raw)
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { sys::rt_tensor_free(self.raw) }
    }
}

/// Releases an unconsumed DLPack pointer returned by `Tensor::to_dlpack_raw`.
///
/// # Safety
///
/// `raw` must be a live `DLManagedTensor` pointer produced by RTorch and must
/// not already have been consumed by another DLPack consumer.
pub unsafe fn free_dlpack_raw(raw: *mut c_void) {
    unsafe { sys::rt_dlpack_free(raw) }
}

/// Computes aggregate statistics with LibTorch over a supported Rust slice.
pub fn statistics<T: StatisticsElement>(values: &[T]) -> Result<Statistics, Error> {
    let mut raw = sys::RtStatistics::default();
    let data = if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr().cast()
    };
    let ok = unsafe { sys::rt_statistics(data, values.len(), T::SCALAR_TYPE, &mut raw) };
    if !ok {
        return Err(take_last_error());
    }
    Ok(statistics_from_raw(raw))
}

pub struct TransformPipeline {
    raw: *mut sys::RtTransformPipeline,
}

impl TransformPipeline {
    pub fn new(config: &TransformConfig) -> Result<Self, Error> {
        let mean = config.normalize_mean.as_deref().unwrap_or(&[]);
        let std = config.normalize_std.as_deref().unwrap_or(&[]);

        let raw = unsafe {
            sys::rt_transform_pipeline_new(
                config.to_float,
                config.resize.map(|v| v.0).unwrap_or(-1),
                config.resize.map(|v| v.1).unwrap_or(-1),
                config.center_crop.map(|v| v.0).unwrap_or(-1),
                config.center_crop.map(|v| v.1).unwrap_or(-1),
                if mean.is_empty() {
                    std::ptr::null()
                } else {
                    mean.as_ptr()
                },
                mean.len(),
                if std.is_empty() {
                    std::ptr::null()
                } else {
                    std.as_ptr()
                },
                std.len(),
            )
        };

        if raw.is_null() {
            return Err(take_last_error());
        }

        Ok(Self { raw })
    }

    fn raw(&self) -> *mut sys::RtTransformPipeline {
        self.raw
    }
}

impl Drop for TransformPipeline {
    fn drop(&mut self) {
        unsafe { sys::rt_transform_pipeline_free(self.raw) }
    }
}

pub struct TensorDataset {
    raw: *mut sys::RtTensorDataset,
}

impl TensorDataset {
    pub fn from_class_folders(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = path_to_cstring(root.as_ref())?;
        let raw = unsafe { sys::rt_tensor_dataset_new(root.as_ptr()) };
        if raw.is_null() {
            return Err(take_last_error());
        }

        Ok(Self { raw })
    }

    pub fn from_hdf5_file(
        path: impl AsRef<Path>,
        dataset_name: impl AsRef<str>,
    ) -> Result<Self, Error> {
        let path = path_to_cstring(path.as_ref())?;
        let dataset_name = CString::new(dataset_name.as_ref())?;
        let raw =
            unsafe { sys::rt_tensor_dataset_hdf5_file_new(path.as_ptr(), dataset_name.as_ptr()) };
        if raw.is_null() {
            return Err(take_last_error());
        }

        Ok(Self { raw })
    }

    pub fn len(&self) -> usize {
        unsafe { sys::rt_tensor_dataset_len(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn class_names(&self) -> Result<Vec<String>, Error> {
        let count = unsafe { sys::rt_tensor_dataset_class_count(self.raw) };
        let mut names = Vec::with_capacity(count);

        for index in 0..count {
            let ptr = unsafe { sys::rt_tensor_dataset_class_name(self.raw, index) };
            if ptr.is_null() {
                return Err(take_last_error());
            }

            let name = unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            names.push(name);
        }

        Ok(names)
    }

    pub fn get(&self, index: usize) -> Result<Sample, Error> {
        self.get_inner(index, None)
    }

    /// Read an atom span directly when its offset/length is already known.
    /// Avoids re-reading the structures table separately for each atom field.
    pub fn read_hdf5_slice(&self, offset: u64, length: u64) -> Result<Tensor, Error> {
        Tensor::from_raw(unsafe {
            sys::rt_tensor_dataset_read_hdf5_slice(self.raw, offset, length)
        })
    }

    pub fn get_with_transform(
        &self,
        index: usize,
        transform: &TransformPipeline,
    ) -> Result<Sample, Error> {
        self.get_inner(index, Some(transform))
    }

    fn get_inner(
        &self,
        index: usize,
        transform: Option<&TransformPipeline>,
    ) -> Result<Sample, Error> {
        let raw = unsafe {
            sys::rt_tensor_dataset_get(
                self.raw,
                index,
                transform.map(|t| t.raw()).unwrap_or(std::ptr::null_mut()),
            )
        };
        if raw.is_null() {
            return Err(take_last_error());
        }

        Ok(Sample {
            shape: owned_shape(raw)?,
            raw,
        })
    }

    fn raw(&self) -> *mut sys::RtTensorDataset {
        self.raw
    }
}

impl Drop for TensorDataset {
    fn drop(&mut self) {
        unsafe { sys::rt_tensor_dataset_free(self.raw) }
    }
}

#[derive(Clone, Debug)]
pub struct HDF5GraphStructure {
    pub offset: i64,
    pub length: i32,
    pub cell: [f64; 9],
    pub pbc: [bool; 3],
    pub energy: f64,
    pub stress: [f64; 6],
    pub virials: [f64; 9],
    pub dipole: [f64; 3],
    pub energy_weight: f32,
    pub forces_weight: f32,
    pub stress_weight: f32,
    pub virials_weight: f32,
    pub dipole_weight: f32,
}

impl From<sys::RtHdf5GraphStructure> for HDF5GraphStructure {
    fn from(value: sys::RtHdf5GraphStructure) -> Self {
        Self {
            offset: value.offset,
            length: value.length,
            cell: value.cell,
            pbc: [value.pbc[0] != 0, value.pbc[1] != 0, value.pbc[2] != 0],
            energy: value.energy,
            stress: value.stress,
            virials: value.virials,
            dipole: value.dipole,
            energy_weight: value.energy_weight,
            forces_weight: value.forces_weight,
            stress_weight: value.stress_weight,
            virials_weight: value.virials_weight,
            dipole_weight: value.dipole_weight,
        }
    }
}

pub struct HDF5GraphMetadataDataset {
    raw: *mut sys::RtHdf5GraphDataset,
}

impl HDF5GraphMetadataDataset {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path_to_cstring(path.as_ref())?;
        let raw = unsafe { sys::rt_hdf5_graph_dataset_new(path.as_ptr()) };
        if raw.is_null() {
            return Err(take_last_error());
        }
        Ok(Self { raw })
    }

    pub fn len(&self) -> usize {
        unsafe { sys::rt_hdf5_graph_dataset_len(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Result<HDF5GraphStructure, Error> {
        let mut raw = sys::RtHdf5GraphStructure::default();
        let ok = unsafe { sys::rt_hdf5_graph_dataset_get(self.raw, index, &mut raw) };
        if ok {
            Ok(raw.into())
        } else {
            Err(take_last_error())
        }
    }
}

impl Drop for HDF5GraphMetadataDataset {
    fn drop(&mut self) {
        unsafe { sys::rt_hdf5_graph_dataset_free(self.raw) }
    }
}

fn owned_shape(raw: *mut sys::RtOwnedBatch) -> Result<Vec<usize>, Error> {
    let len = unsafe { sys::rt_owned_batch_shape_len(raw) };
    let ptr = unsafe { sys::rt_owned_batch_shape(raw) };
    if len > 0 && ptr.is_null() {
        unsafe { sys::rt_owned_batch_free(raw) };
        return Err(Error::Backend(
            "backend returned a null shape pointer".to_string(),
        ));
    }
    let raw_shape = if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    };
    let mut shape = Vec::with_capacity(raw_shape.len());
    for value in raw_shape {
        let converted = match (*value).try_into() {
            Ok(converted) => converted,
            Err(_) => {
                unsafe { sys::rt_owned_batch_free(raw) };
                return Err(Error::ShapeOverflow);
            }
        };
        shape.push(converted);
    }
    Ok(shape)
}

#[derive(Debug)]
pub struct Sample {
    raw: *mut sys::RtOwnedBatch,
    shape: Vec<usize>,
}

impl Sample {
    /// Materializes a contiguous CPU float32 view on first use.
    pub fn data(&self) -> Result<&[f32], Error> {
        owned_data(self.raw, &self.raw)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn label(&self) -> i64 {
        let ptr = unsafe { sys::rt_owned_batch_labels(self.raw) };
        let len = unsafe { sys::rt_owned_batch_labels_len(self.raw) };
        debug_assert_eq!(len, 1);
        if len == 0 {
            return 0;
        }
        assert!(!ptr.is_null(), "backend returned a null label pointer");
        unsafe { *ptr }
    }

    pub fn statistics(&self) -> Result<Statistics, Error> {
        let mut raw = sys::RtStatistics::default();
        let ok = unsafe { sys::rt_owned_batch_statistics(self.raw, &mut raw) };
        if !ok {
            return Err(take_last_error());
        }

        Ok(statistics_from_raw(raw))
    }

    pub fn tensor(&self) -> Result<Tensor, Error> {
        Tensor::from_raw(unsafe { sys::rt_owned_batch_tensor(self.raw) })
    }
}

impl Drop for Sample {
    fn drop(&mut self) {
        unsafe { sys::rt_owned_batch_free(self.raw) }
    }
}

#[derive(Debug)]
pub struct Batch {
    raw: *mut sys::RtOwnedBatch,
    shape: Vec<usize>,
}

impl Batch {
    /// Materializes a contiguous CPU float32 view on first use.
    pub fn data(&self) -> Result<&[f32], Error> {
        owned_data(self.raw, &self.raw)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn labels(&self) -> &[i64] {
        let ptr = unsafe { sys::rt_owned_batch_labels(self.raw) };
        let len = unsafe { sys::rt_owned_batch_labels_len(self.raw) };
        if len == 0 {
            &[]
        } else {
            assert!(!ptr.is_null(), "backend returned a null labels pointer");
            unsafe { std::slice::from_raw_parts(ptr, len) }
        }
    }

    /// Returns aggregate statistics computed by LibTorch over all batch elements.
    pub fn statistics(&self) -> Result<Statistics, Error> {
        let mut raw = sys::RtStatistics::default();
        let ok = unsafe { sys::rt_owned_batch_statistics(self.raw, &mut raw) };
        if !ok {
            return Err(take_last_error());
        }

        Ok(statistics_from_raw(raw))
    }

    /// Returns an owned LibTorch tensor containing the batch data.
    pub fn tensor(&self) -> Result<Tensor, Error> {
        Tensor::from_raw(unsafe { sys::rt_owned_batch_tensor(self.raw) })
    }
}

fn owned_data(
    raw: *mut sys::RtOwnedBatch,
    _owner: &*mut sys::RtOwnedBatch,
) -> Result<&[f32], Error> {
    let mut ptr = std::ptr::null();
    let mut len = 0;
    if !unsafe { sys::rt_owned_batch_data_view(raw, &mut ptr, &mut len) } {
        return Err(take_last_error());
    }
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(Error::Backend(
            "backend returned a null data pointer".to_string(),
        ));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { sys::rt_owned_batch_free(self.raw) }
    }
}

pub struct DataLoader<'a> {
    raw: *mut sys::RtDataLoader,
    _dataset: PhantomData<&'a TensorDataset>,
    _transforms: PhantomData<&'a TransformPipeline>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            shuffle: false,
            drop_last: false,
        }
    }
}

impl<'a> DataLoader<'a> {
    pub fn new(
        dataset: &'a TensorDataset,
        transforms: Option<&'a TransformPipeline>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self, Error> {
        Self::with_config(
            dataset,
            transforms,
            DataLoaderConfig {
                batch_size,
                shuffle,
                drop_last: false,
            },
        )
    }

    pub fn with_config(
        dataset: &'a TensorDataset,
        transforms: Option<&'a TransformPipeline>,
        config: DataLoaderConfig,
    ) -> Result<Self, Error> {
        let raw = unsafe {
            sys::rt_data_loader_new_with_options(
                dataset.raw(),
                transforms.map(|t| t.raw()).unwrap_or(std::ptr::null_mut()),
                config.batch_size,
                config.shuffle,
                config.drop_last,
            )
        };

        if raw.is_null() {
            return Err(take_last_error());
        }

        Ok(Self {
            raw,
            _dataset: PhantomData,
            _transforms: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        unsafe { sys::rt_data_loader_len(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        let ok = unsafe { sys::rt_data_loader_reset(self.raw) };
        if ok {
            Ok(())
        } else {
            Err(take_last_error())
        }
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch>, Error> {
        let mut batch = std::ptr::null_mut();

        let status = unsafe { sys::rt_data_loader_next(self.raw, &mut batch) };
        match status {
            sys::RT_STATUS_OK => {
                if batch.is_null() {
                    return Err(Error::Backend(
                        "backend returned success without a batch".to_string(),
                    ));
                }
                Ok(Some(Batch {
                    shape: owned_shape(batch)?,
                    raw: batch,
                }))
            }
            sys::RT_STATUS_END => Ok(None),
            _ => Err(take_last_error()),
        }
    }
}

impl Iterator for DataLoader<'_> {
    type Item = Result<Batch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    }
}

impl Drop for DataLoader<'_> {
    fn drop(&mut self) {
        unsafe { sys::rt_data_loader_free(self.raw) }
    }
}

pub struct OwnedDataLoader {
    raw: *mut sys::RtDataLoader,
    _dataset: TensorDataset,
    _transforms: Option<TransformPipeline>,
}

impl OwnedDataLoader {
    pub fn with_config(
        dataset: TensorDataset,
        transforms: Option<TransformPipeline>,
        config: DataLoaderConfig,
    ) -> Result<Self, Error> {
        let raw = unsafe {
            sys::rt_data_loader_new_with_options(
                dataset.raw(),
                transforms
                    .as_ref()
                    .map(|transform| transform.raw())
                    .unwrap_or(std::ptr::null_mut()),
                config.batch_size,
                config.shuffle,
                config.drop_last,
            )
        };

        if raw.is_null() {
            return Err(take_last_error());
        }

        Ok(Self {
            raw,
            _dataset: dataset,
            _transforms: transforms,
        })
    }

    pub fn from_dataset(
        dataset: TensorDataset,
        transforms: Option<TransformPipeline>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self, Error> {
        Self::with_config(
            dataset,
            transforms,
            DataLoaderConfig {
                batch_size,
                shuffle,
                drop_last: false,
            },
        )
    }

    pub fn len(&self) -> usize {
        unsafe { sys::rt_data_loader_len(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        let ok = unsafe { sys::rt_data_loader_reset(self.raw) };
        if ok {
            Ok(())
        } else {
            Err(take_last_error())
        }
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch>, Error> {
        let mut batch = std::ptr::null_mut();

        let status = unsafe { sys::rt_data_loader_next(self.raw, &mut batch) };
        match status {
            sys::RT_STATUS_OK => {
                if batch.is_null() {
                    return Err(Error::Backend(
                        "backend returned success without a batch".to_string(),
                    ));
                }
                Ok(Some(Batch {
                    shape: owned_shape(batch)?,
                    raw: batch,
                }))
            }
            sys::RT_STATUS_END => Ok(None),
            _ => Err(take_last_error()),
        }
    }
}

impl Iterator for OwnedDataLoader {
    type Item = Result<Batch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    }
}

impl Drop for OwnedDataLoader {
    fn drop(&mut self) {
        unsafe { sys::rt_data_loader_free(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{
        bf16, f16, statistics, DataLoader, ScalarType, Tensor, TensorDataset, TransformConfig,
        TransformPipeline,
    };

    static TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("rtorch-{name}-{}-{id}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn decode_base64(input: &str) -> Vec<u8> {
        let mut output = Vec::new();
        let mut buffer = 0_u32;
        let mut bits = 0_u8;
        for byte in input.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
            if byte == b'=' {
                break;
            }
            let value = match byte {
                b'A'..=b'Z' => byte - b'A',
                b'a'..=b'z' => byte - b'a' + 26,
                b'0'..=b'9' => byte - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => panic!("invalid base64 test fixture"),
            };
            buffer = (buffer << 6) | u32::from(value);
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                output.push((buffer >> bits) as u8);
                buffer &= (1_u32 << bits) - 1;
            }
        }
        output
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    #[test]
    fn computes_f32_statistics_with_libtorch() {
        let result = statistics(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(result.count, 4);
        assert_close(result.sum, 10.0);
        assert_close(result.mean, 2.5);
        assert_close(result.population_std, 1.25_f64.sqrt());
        assert_close(result.sample_std, (5.0_f64 / 3.0).sqrt());
        assert_close(result.min, 1.0);
        assert_close(result.max, 4.0);
    }

    #[test]
    fn computes_f64_statistics_with_libtorch() {
        let result = statistics(&[1.5_f64, 2.5, 3.5]).unwrap();

        assert_eq!(result.count, 3);
        assert_close(result.mean, 2.5);
        assert_close(result.min, 1.5);
        assert_close(result.max, 3.5);
    }

    #[test]
    fn computes_integer_statistics_with_libtorch() {
        let signed = statistics(&[-4_i64, 0, 8]).unwrap();
        let unsigned = statistics(&[2_u16, 4, 6]).unwrap();

        assert_close(signed.mean, 4.0 / 3.0);
        assert_close(signed.min, -4.0);
        assert_close(unsigned.mean, 4.0);
        assert_close(unsigned.max, 6.0);
    }

    #[test]
    fn computes_reduced_float_statistics_with_libtorch() {
        let float16 =
            statistics(&[f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)]).unwrap();
        let bfloat16 = statistics(&[
            bf16::from_f32(2.0),
            bf16::from_f32(4.0),
            bf16::from_f32(6.0),
        ])
        .unwrap();

        assert_close(float16.mean, 2.0);
        assert_close(bfloat16.mean, 4.0);
    }

    #[test]
    fn computes_boolean_statistics_with_libtorch() {
        let result = statistics(&[true, false, true, true]).unwrap();

        assert_close(result.mean, 0.75);
        assert_close(result.min, 0.0);
        assert_close(result.max, 1.0);
    }

    #[test]
    fn rejects_empty_statistics_input() {
        assert!(statistics::<f32>(&[]).is_err());
    }

    #[test]
    fn transferred_vector_storage_survives_reshape_and_owner_drop() {
        let tensor = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let view = tensor.reshape(&[4]).unwrap();
        drop(tensor);
        assert_eq!(view.to_vec_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(Tensor::from_vec(vec![1_i64], &[2]).is_err());
        let empty = Tensor::from_vec(Vec::<f32>::new(), &[0, 3]).unwrap();
        assert_eq!(empty.shape().unwrap(), vec![0, 3]);
        assert_eq!(empty.numel(), 0);
    }

    #[test]
    fn creates_and_inspects_owned_tensor() {
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], false).unwrap();

        assert_eq!(tensor.shape().unwrap(), vec![2, 2]);
        assert_eq!(tensor.scalar_type().unwrap(), ScalarType::F32);
        assert_eq!(tensor.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn performs_tensor_algebra() {
        let left = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], false).unwrap();
        let right = Tensor::ones(&[2, 2], ScalarType::F32, false).unwrap();

        let added = left.add(&right).unwrap();
        assert_eq!(added.to_vec_f32().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);

        let multiplied = left.matmul(&right).unwrap();
        assert_eq!(multiplied.to_vec_f32().unwrap(), vec![3.0, 3.0, 7.0, 7.0]);
    }

    #[test]
    fn computes_tensor_gradients() {
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0, 3.0], &[3], true).unwrap();
        let doubled = tensor.add(&tensor).unwrap();
        let mut total = doubled.sum().unwrap();
        total.backward().unwrap();

        assert_eq!(tensor.grad().unwrap().to_vec_f32().unwrap(), vec![2.0; 3]);
    }

    #[test]
    fn loads_python_torch_save_tensor_and_batches_it() {
        let root = TestDirectory::new("python-pt");
        let class_dir = root.0.join("example");
        fs::create_dir(&class_dir).unwrap();
        fs::write(
            class_dir.join("sample.pt"),
            decode_base64(include_str!("../../../test_data/python_tensor.pt.base64")),
        )
        .unwrap();

        let dataset = TensorDataset::from_class_folders(&root.0).unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.class_names().unwrap(), ["example"]);

        let sample = dataset.get(0).unwrap();
        assert_eq!(sample.shape(), [2, 3]);
        assert_eq!(sample.data().unwrap(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(
            sample.tensor().unwrap().scalar_type().unwrap(),
            ScalarType::F32
        );

        let mut loader = DataLoader::new(&dataset, None, 1, false).unwrap();
        let batch = loader.next().unwrap().unwrap();
        assert_eq!(batch.shape(), [1, 2, 3]);
        assert_eq!(batch.labels(), [0]);
        assert_eq!(batch.data().unwrap(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(loader.next().is_none());
    }

    #[test]
    fn loads_python_torch_save_dict_samples_tensor() {
        let root = TestDirectory::new("python-pt-dict");
        let class_dir = root.0.join("example");
        fs::create_dir(&class_dir).unwrap();
        fs::write(
            class_dir.join("sample.pt"),
            decode_base64(include_str!(
                "../../../test_data/python_samples_dict.pt.base64"
            )),
        )
        .unwrap();

        let dataset = TensorDataset::from_class_folders(&root.0).unwrap();
        assert_eq!(dataset.len(), 1);

        let sample = dataset.get(0).unwrap();
        assert_eq!(sample.shape(), [2, 3]);
        assert_eq!(sample.data().unwrap(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn loader_stops_after_a_sample_error_and_can_be_reset() {
        let root = TestDirectory::new("loader-error");
        let class_dir = root.0.join("broken");
        fs::create_dir(&class_dir).unwrap();
        fs::write(class_dir.join("sample.pt"), b"not a tensor archive").unwrap();

        let dataset = TensorDataset::from_class_folders(&root.0).unwrap();
        let mut loader = DataLoader::new(&dataset, None, 1, false).unwrap();
        assert!(loader.next().unwrap().is_err());
        assert!(loader.next().is_none());
        loader.reset().unwrap();
        assert!(loader.next().unwrap().is_err());
    }

    #[test]
    fn rejects_invalid_transform_configuration_at_construction() {
        assert!(TransformPipeline::new(&TransformConfig {
            resize: Some((0, 224)),
            ..TransformConfig::default()
        })
        .is_err());
        assert!(TransformPipeline::new(&TransformConfig {
            normalize_mean: Some(vec![0.5]),
            normalize_std: None,
            ..TransformConfig::default()
        })
        .is_err());
        assert!(TransformPipeline::new(&TransformConfig {
            normalize_mean: Some(vec![0.5]),
            normalize_std: Some(vec![0.0]),
            ..TransformConfig::default()
        })
        .is_err());
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn loads_structured_hdf5_samples() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/train.h5");
        let dataset = TensorDataset::from_hdf5_file(path, "positions").unwrap();
        assert!(!dataset.is_empty());

        let sample = dataset.get(0).unwrap();
        assert_eq!(sample.shape().last(), Some(&3));
        assert!(!sample.data().unwrap().is_empty());
    }
}
