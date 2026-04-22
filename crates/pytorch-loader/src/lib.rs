use std::ffi::{CString, NulError};
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::path::Path;

use pytorch_loader_sys as sys;

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
        let ptr = sys::ptl_last_error_message();
        if ptr.is_null() {
            return Error::Backend("unknown backend error".to_string());
        }

        let message = std::ffi::CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned();
        sys::ptl_last_error_free(ptr as *mut c_char);
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

pub struct TransformPipeline {
    raw: *mut sys::PtlTransformPipeline,
}

impl TransformPipeline {
    pub fn new(config: &TransformConfig) -> Result<Self, Error> {
        let mean = config.normalize_mean.as_deref().unwrap_or(&[]);
        let std = config.normalize_std.as_deref().unwrap_or(&[]);

        let raw = unsafe {
            sys::ptl_transform_pipeline_new(
                config.to_float,
                config.resize.map(|v| v.0).unwrap_or(-1),
                config.resize.map(|v| v.1).unwrap_or(-1),
                config.center_crop.map(|v| v.0).unwrap_or(-1),
                config.center_crop.map(|v| v.1).unwrap_or(-1),
                mean.as_ptr(),
                mean.len(),
                std.as_ptr(),
                std.len(),
            )
        };

        if raw.is_null() {
            return Err(take_last_error());
        }

        Ok(Self { raw })
    }

    fn raw(&self) -> *mut sys::PtlTransformPipeline {
        self.raw
    }
}

impl Drop for TransformPipeline {
    fn drop(&mut self) {
        unsafe { sys::ptl_transform_pipeline_free(self.raw) }
    }
}

pub struct TensorDataset {
    raw: *mut sys::PtlTensorDataset,
}

impl TensorDataset {
    pub fn from_class_folders(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = path_to_cstring(root.as_ref())?;
        let raw = unsafe { sys::ptl_tensor_dataset_new(root.as_ptr()) };
        if raw.is_null() {
            return Err(take_last_error());
        }

        Ok(Self { raw })
    }

    fn raw(&self) -> *mut sys::PtlTensorDataset {
        self.raw
    }
}

impl Drop for TensorDataset {
    fn drop(&mut self) {
        unsafe { sys::ptl_tensor_dataset_free(self.raw) }
    }
}

#[derive(Debug)]
pub struct Batch {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub labels: Vec<i64>,
}

pub struct DataLoader<'a> {
    raw: *mut sys::PtlDataLoader,
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
            sys::ptl_data_loader_new(
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
        let ok = unsafe { sys::ptl_data_loader_reset(self.raw) };
        if ok {
            Ok(())
        } else {
            Err(take_last_error())
        }
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch>, Error> {
        let mut batch = sys::PtlBatchView {
            data: std::ptr::null(),
            data_len: 0,
            shape: std::ptr::null(),
            shape_len: 0,
            labels: std::ptr::null(),
            labels_len: 0,
        };

        let status = unsafe { sys::ptl_data_loader_next(self.raw, &mut batch) };
        match status {
            sys::PTL_STATUS_OK => {
                let data = unsafe { std::slice::from_raw_parts(batch.data, batch.data_len) }.to_vec();
                let raw_shape = unsafe { std::slice::from_raw_parts(batch.shape, batch.shape_len) };
                let mut shape = Vec::with_capacity(raw_shape.len());
                for value in raw_shape {
                    shape.push((*value).try_into().map_err(|_| Error::ShapeOverflow)?);
                }
                let labels =
                    unsafe { std::slice::from_raw_parts(batch.labels, batch.labels_len) }.to_vec();
                unsafe { sys::ptl_batch_view_free(&mut batch) };
                Ok(Some(Batch { data, shape, labels }))
            }
            sys::PTL_STATUS_END => Ok(None),
            _ => Err(take_last_error()),
        }
    }
}

impl Drop for DataLoader<'_> {
    fn drop(&mut self) {
        unsafe { sys::ptl_data_loader_free(self.raw) }
    }
}
