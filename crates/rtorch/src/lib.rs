use std::ffi::{CString, NulError};
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
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InteriorNul(err) => write!(f, "string contains interior NUL: {err}"),
            Self::Backend(msg) => write!(f, "{msg}"),
            Self::InvalidUtf8Path => write!(f, "path is not valid UTF-8"),
            Self::ShapeOverflow => write!(f, "tensor shape value overflowed usize"),
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

        let message = std::ffi::CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned();
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

    fn raw(&self) -> *mut sys::RtTensorDataset {
        self.raw
    }
}

impl Drop for TensorDataset {
    fn drop(&mut self) {
        unsafe { sys::rt_tensor_dataset_free(self.raw) }
    }
}

#[derive(Debug)]
pub struct Batch {
    raw: *mut sys::RtOwnedBatch,
    shape: Vec<usize>,
}

impl Batch {
    pub fn data(&self) -> &[f32] {
        let ptr = unsafe { sys::rt_owned_batch_data(self.raw) };
        let len = unsafe { sys::rt_owned_batch_data_len(self.raw) };
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn labels(&self) -> &[i64] {
        let ptr = unsafe { sys::rt_owned_batch_labels(self.raw) };
        let len = unsafe { sys::rt_owned_batch_labels_len(self.raw) };
        unsafe { std::slice::from_raw_parts(ptr, len) }
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

impl<'a> DataLoader<'a> {
    pub fn new(
        dataset: &'a TensorDataset,
        transforms: Option<&'a TransformPipeline>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self, Error> {
        let raw = unsafe {
            sys::rt_data_loader_new(
                dataset.raw(),
                transforms.map(|t| t.raw()).unwrap_or(std::ptr::null_mut()),
                batch_size,
                shuffle,
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
                let raw_shape = unsafe {
                    std::slice::from_raw_parts(
                        sys::rt_owned_batch_shape(batch),
                        sys::rt_owned_batch_shape_len(batch),
                    )
                };
                let mut shape = Vec::with_capacity(raw_shape.len());
                for value in raw_shape {
                    let converted = match (*value).try_into() {
                        Ok(converted) => converted,
                        Err(_) => {
                            unsafe { sys::rt_owned_batch_free(batch) };
                            return Err(Error::ShapeOverflow);
                        }
                    };
                    shape.push(converted);
                }
                Ok(Some(Batch { raw: batch, shape }))
            }
            sys::RT_STATUS_END => Ok(None),
            _ => Err(take_last_error()),
        }
    }
}

impl Drop for DataLoader<'_> {
    fn drop(&mut self) {
        unsafe { sys::rt_data_loader_free(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::{bf16, f16, statistics};

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
        let float16 = statistics(&[
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ])
        .unwrap();
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
}
