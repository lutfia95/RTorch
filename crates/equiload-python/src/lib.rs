#![allow(clippy::too_many_arguments, clippy::useless_conversion)]

use std::ffi::c_char;

use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

mod graph;

use equiload_core::{
    free_dlpack_raw, Batch as RtBatch, DataLoaderConfig, Error as RtError,
    OwnedDataLoader as RtOwnedDataLoader, Sample as RtSample, Tensor as RtTensor,
    TensorDataset as RtTensorDataset,
};

const DLPACK_CAPSULE_NAME: &[u8] = b"dltensor\0";
static FROM_DLPACK: GILOnceCell<PyObject> = GILOnceCell::new();

pub(crate) fn to_py_err(err: RtError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut ffi::PyObject) {
    let name = DLPACK_CAPSULE_NAME.as_ptr().cast::<c_char>();
    if unsafe { ffi::PyCapsule_IsValid(capsule, name) } == 0 {
        return;
    }
    let raw = unsafe { ffi::PyCapsule_GetPointer(capsule, name) };
    if !raw.is_null() {
        unsafe { free_dlpack_raw(raw.cast()) };
    }
}

pub(crate) fn tensor_to_torch(py: Python<'_>, tensor: &RtTensor) -> PyResult<PyObject> {
    let raw = tensor.to_dlpack_raw().map_err(to_py_err)?;
    let capsule = unsafe {
        ffi::PyCapsule_New(
            raw.cast(),
            DLPACK_CAPSULE_NAME.as_ptr().cast::<c_char>(),
            Some(dlpack_capsule_destructor),
        )
    };
    if capsule.is_null() {
        unsafe { free_dlpack_raw(raw) };
        return Err(PyErr::fetch(py));
    }

    let capsule = unsafe { PyObject::from_owned_ptr(py, capsule) };
    let from_dlpack = FROM_DLPACK.get_or_try_init(py, || -> PyResult<PyObject> {
        Ok(py
            .import_bound("torch.utils.dlpack")?
            .getattr("from_dlpack")?
            .into_py(py))
    })?;
    let torch_tensor = from_dlpack.bind(py).call1((capsule,))?;
    Ok(torch_tensor.into_py(py))
}

#[pyclass(name = "Tensor", unsendable)]
struct PyTensor {
    inner: RtTensor,
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        self.inner.shape().map_err(to_py_err)
    }

    #[getter]
    fn scalar_type(&self) -> PyResult<String> {
        self.inner
            .scalar_type()
            .map(|value| format!("{value:?}"))
            .map_err(to_py_err)
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    fn to_torch(&self, py: Python<'_>) -> PyResult<PyObject> {
        tensor_to_torch(py, &self.inner)
    }

    fn tolist_f32(&self) -> PyResult<Vec<f32>> {
        self.inner.to_vec_f32().map_err(to_py_err)
    }
}

#[pyclass(name = "Sample", unsendable)]
struct PySample {
    inner: RtSample,
}

#[pymethods]
impl PySample {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn label(&self) -> i64 {
        self.inner.label()
    }

    fn tensor(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.tensor().map_err(to_py_err)?,
        })
    }

    fn to_torch(&self, py: Python<'_>) -> PyResult<PyObject> {
        let tensor = self.inner.tensor().map_err(to_py_err)?;
        tensor_to_torch(py, &tensor)
    }

    fn data_f32(&self) -> PyResult<Vec<f32>> {
        self.inner
            .data()
            .map(|values| values.to_vec())
            .map_err(to_py_err)
    }
}

#[pyclass(name = "Batch", unsendable)]
struct PyBatch {
    inner: RtBatch,
}

#[pymethods]
impl PyBatch {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn labels(&self) -> Vec<i64> {
        self.inner.labels().to_vec()
    }

    fn tensor(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.tensor().map_err(to_py_err)?,
        })
    }

    fn to_torch(&self, py: Python<'_>) -> PyResult<PyObject> {
        let tensor = self.inner.tensor().map_err(to_py_err)?;
        tensor_to_torch(py, &tensor)
    }

    fn data_f32(&self) -> PyResult<Vec<f32>> {
        self.inner
            .data()
            .map(|values| values.to_vec())
            .map_err(to_py_err)
    }
}

#[pyclass(name = "TensorDataset", unsendable)]
struct PyTensorDataset {
    inner: RtTensorDataset,
}

#[pymethods]
impl PyTensorDataset {
    #[staticmethod]
    fn from_class_folders(root: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RtTensorDataset::from_class_folders(root).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, dataset_name = "positions"))]
    fn from_hdf5_file(path: &str, dataset_name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RtTensorDataset::from_hdf5_file(path, dataset_name).map_err(to_py_err)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn class_names(&self) -> PyResult<Vec<String>> {
        self.inner.class_names().map_err(to_py_err)
    }

    fn get(&self, index: usize) -> PyResult<PySample> {
        Ok(PySample {
            inner: self.inner.get(index).map_err(to_py_err)?,
        })
    }

    fn __getitem__(&self, index: usize) -> PyResult<PySample> {
        self.get(index)
    }
}

#[pyclass(name = "DataLoader", unsendable)]
struct PyDataLoader {
    inner: RtOwnedDataLoader,
}

#[pymethods]
impl PyDataLoader {
    #[staticmethod]
    #[pyo3(signature = (root, batch_size = 1, shuffle = false, drop_last = false))]
    fn from_class_folders(
        root: &str,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
    ) -> PyResult<Self> {
        let dataset = RtTensorDataset::from_class_folders(root).map_err(to_py_err)?;
        Self::from_dataset(dataset, batch_size, shuffle, drop_last)
    }

    #[staticmethod]
    #[pyo3(signature = (path, dataset_name = "positions", batch_size = 1, shuffle = false, drop_last = false))]
    fn from_hdf5_file(
        path: &str,
        dataset_name: &str,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
    ) -> PyResult<Self> {
        let dataset = RtTensorDataset::from_hdf5_file(path, dataset_name).map_err(to_py_err)?;
        Self::from_dataset(dataset, batch_size, shuffle, drop_last)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn reset(&mut self) -> PyResult<()> {
        self.inner.reset().map_err(to_py_err)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyBatch>> {
        self.inner
            .next_batch()
            .map(|batch| batch.map(|inner| PyBatch { inner }))
            .map_err(to_py_err)
    }
}

impl PyDataLoader {
    fn from_dataset(
        dataset: RtTensorDataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RtOwnedDataLoader::with_config(
                dataset,
                None,
                DataLoaderConfig {
                    batch_size,
                    shuffle,
                    drop_last,
                },
            )
            .map_err(to_py_err)?,
        })
    }
}

#[pymodule]
fn equiload(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<PyTensor>()?;
    module.add_class::<PySample>()?;
    module.add_class::<PyBatch>()?;
    module.add_class::<PyTensorDataset>()?;
    module.add_class::<PyDataLoader>()?;
    graph::register(module)?;
    Ok(())
}
