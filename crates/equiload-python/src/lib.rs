//! Python bindings for EquiLoad's native HDF5 graph loader.
//!
//! The public Python API intentionally exposes only the graph-loader classes
//! consumed by Equitrain. Tensor conversion stays internal to this module.
#![allow(clippy::too_many_arguments, clippy::useless_conversion)]

use std::ffi::c_char;

use equiload_core::{free_dlpack_raw, Error as RtError, Tensor as RtTensor};
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

mod graph;

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
    Ok(from_dlpack.bind(py).call1((capsule,))?.into_py(py))
}

#[pymodule]
fn equiload(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    graph::register(module)?;
    Ok(())
}
