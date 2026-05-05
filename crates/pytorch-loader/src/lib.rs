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

    pub fn len(&self) -> usize {
        unsafe { sys::ptl_tensor_dataset_len(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn class_names(&self) -> Result<Vec<String>, Error> {
        let count = unsafe { sys::ptl_tensor_dataset_class_count(self.raw) };
        let mut names = Vec::with_capacity(count);

        for index in 0..count {
            let ptr = unsafe { sys::ptl_tensor_dataset_class_name(self.raw, index) };
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
    raw: *mut sys::PtlOwnedBatch,
    shape: Vec<usize>,
}

impl Batch {
    pub fn data(&self) -> &[f32] {
        let ptr = unsafe { sys::ptl_owned_batch_data(self.raw) };
        let len = unsafe { sys::ptl_owned_batch_data_len(self.raw) };
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn labels(&self) -> &[i64] {
        let ptr = unsafe { sys::ptl_owned_batch_labels(self.raw) };
        let len = unsafe { sys::ptl_owned_batch_labels_len(self.raw) };
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { sys::ptl_owned_batch_free(self.raw) }
    }
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
        let mut batch = std::ptr::null_mut();

        let status = unsafe { sys::ptl_data_loader_next(self.raw, &mut batch) };
        match status {
            sys::PTL_STATUS_OK => {
                let raw_shape = unsafe {
                    std::slice::from_raw_parts(
                        sys::ptl_owned_batch_shape(batch),
                        sys::ptl_owned_batch_shape_len(batch),
                    )
                };
                let mut shape = Vec::with_capacity(raw_shape.len());
                for value in raw_shape {
                    let converted = match (*value).try_into() {
                        Ok(converted) => converted,
                        Err(_) => {
                            unsafe { sys::ptl_owned_batch_free(batch) };
                            return Err(Error::ShapeOverflow);
                        }
                    };
                    shape.push(converted);
                }
                Ok(Some(Batch { raw: batch, shape }))
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
