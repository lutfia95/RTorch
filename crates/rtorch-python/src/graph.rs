use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rtorch_core::{
    HDF5GraphMetadataDataset as RtHDF5GraphMetadataDataset, Tensor as RtTensor,
    TensorDataset as RtTensorDataset,
};

use crate::{tensor_to_torch, to_py_err};

struct GraphBuildResult {
    graph: PyObject,
    num_nodes: usize,
    num_edges: usize,
}

struct Neighborhood {
    edge_index: Vec<i64>,
    shifts: Vec<f64>,
    unit_shifts: Vec<f64>,
    cell: [f64; 9],
}

fn path_to_string(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<String> {
    let os = py.import_bound("os")?;
    os.getattr("fspath")?.call1((path,))?.extract::<String>()
}

fn determinant(cell: &[f64; 9]) -> f64 {
    cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6])
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn cell_row(cell: &[f64; 9], index: usize) -> [f64; 3] {
    [cell[3 * index], cell[3 * index + 1], cell[3 * index + 2]]
}

fn translation_ranges(cutoff: f64, pbc: [bool; 3], cell: &[f64; 9]) -> [i32; 3] {
    let volume = determinant(cell).abs();
    let rows = [cell_row(cell, 0), cell_row(cell, 1), cell_row(cell, 2)];
    let mut ranges = [0_i32; 3];

    for axis in 0..3 {
        if !pbc[axis] {
            continue;
        }
        let other_a = rows[(axis + 1) % 3];
        let other_b = rows[(axis + 2) % 3];
        let area = vector_norm(cross(other_a, other_b));
        let height = if volume > 1.0e-12 && area > 1.0e-12 {
            volume / area
        } else {
            vector_norm(rows[axis])
        };
        ranges[axis] = if height > 1.0e-12 {
            (cutoff / height).ceil() as i32 + 1
        } else {
            0
        };
    }

    ranges
}

fn get_neighborhood(
    positions: &[f64],
    cutoff: f64,
    pbc: [bool; 3],
    mut cell: [f64; 9],
) -> PyResult<Neighborhood> {
    if positions.is_empty() || !positions.len().is_multiple_of(3) {
        return Err(PyValueError::new_err("positions must have shape [N, 3]."));
    }
    if cutoff <= 0.0 {
        return Err(PyValueError::new_err("r_max must be positive."));
    }

    if cell.iter().all(|value| *value == 0.0) {
        cell = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }

    let max_positions = positions
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        + 1.0;
    let extension = max_positions * 5.0 * cutoff;
    if !pbc[0] {
        cell[0] = extension;
        cell[1] = 0.0;
        cell[2] = 0.0;
    }
    if !pbc[1] {
        cell[3] = 0.0;
        cell[4] = extension;
        cell[5] = 0.0;
    }
    if !pbc[2] {
        cell[6] = 0.0;
        cell[7] = 0.0;
        cell[8] = extension;
    }

    let ranges = translation_ranges(cutoff, pbc, &cell);
    let cutoff2 = cutoff * cutoff;
    let natoms = positions.len() / 3;
    let mut senders = Vec::new();
    let mut receivers = Vec::new();
    let mut shifts = Vec::new();
    let mut unit_shifts = Vec::new();

    for sender in 0..natoms {
        let sender_pos = [
            positions[3 * sender],
            positions[3 * sender + 1],
            positions[3 * sender + 2],
        ];
        for receiver in 0..natoms {
            let receiver_pos = [
                positions[3 * receiver],
                positions[3 * receiver + 1],
                positions[3 * receiver + 2],
            ];
            for sx in -ranges[0]..=ranges[0] {
                for sy in -ranges[1]..=ranges[1] {
                    for sz in -ranges[2]..=ranges[2] {
                        if sender == receiver && sx == 0 && sy == 0 && sz == 0 {
                            continue;
                        }

                        let shift = [
                            sx as f64 * cell[0] + sy as f64 * cell[3] + sz as f64 * cell[6],
                            sx as f64 * cell[1] + sy as f64 * cell[4] + sz as f64 * cell[7],
                            sx as f64 * cell[2] + sy as f64 * cell[5] + sz as f64 * cell[8],
                        ];
                        let displacement = [
                            receiver_pos[0] - sender_pos[0] + shift[0],
                            receiver_pos[1] - sender_pos[1] + shift[1],
                            receiver_pos[2] - sender_pos[2] + shift[2],
                        ];
                        let dist2 = displacement[0] * displacement[0]
                            + displacement[1] * displacement[1]
                            + displacement[2] * displacement[2];
                        if dist2 <= cutoff2 {
                            senders.push(sender as i64);
                            receivers.push(receiver as i64);
                            shifts.extend_from_slice(&shift);
                            unit_shifts.extend_from_slice(&[sx as f64, sy as f64, sz as f64]);
                        }
                    }
                }
            }
        }
    }

    let mut edge_index = Vec::with_capacity(senders.len() + receivers.len());
    edge_index.extend_from_slice(&senders);
    edge_index.extend_from_slice(&receivers);

    Ok(Neighborhood {
        edge_index,
        shifts,
        unit_shifts,
        cell,
    })
}

fn voigt_to_matrix(stress: [f64; 6]) -> [f64; 9] {
    [
        stress[0], stress[5], stress[4], stress[5], stress[1], stress[3], stress[4], stress[3],
        stress[2],
    ]
}

fn torch_default_dtype(py: Python<'_>) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyAny>)> {
    let torch = py.import_bound("torch")?;
    let default_dtype = torch.getattr("get_default_dtype")?.call0()?;
    let long_dtype = torch.getattr("long")?;
    Ok((default_dtype, long_dtype))
}

fn rt_to_torch(py: Python<'_>, tensor: &RtTensor) -> PyResult<PyObject> {
    tensor_to_torch(py, tensor)
}

fn rt_to_torch_dtype(
    py: Python<'_>,
    tensor: &RtTensor,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let object = rt_to_torch(py, tensor)?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", dtype)?;
    Ok(object
        .bind(py)
        .call_method("to", (), Some(&kwargs))?
        .into_py(py))
}

fn reshape_contiguous(py: Python<'_>, object: PyObject, shape: Vec<i64>) -> PyResult<PyObject> {
    let reshaped = object.bind(py).call_method1("reshape", (shape,))?;
    Ok(reshaped.call_method0("contiguous")?.into_py(py))
}

fn float_tensor(
    py: Python<'_>,
    values: &[f64],
    shape: &[usize],
    dtype: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let tensor = RtTensor::from_slice(values, shape, false).map_err(to_py_err)?;
    rt_to_torch_dtype(py, &tensor, dtype)
}

fn i64_tensor(py: Python<'_>, values: &[i64], shape: &[usize]) -> PyResult<PyObject> {
    let tensor = RtTensor::from_slice(values, shape, false).map_err(to_py_err)?;
    rt_to_torch(py, &tensor)
}

fn bool_tensor(py: Python<'_>, values: &[bool], shape: &[usize]) -> PyResult<PyObject> {
    let tensor = RtTensor::from_slice(values, shape, false).map_err(to_py_err)?;
    rt_to_torch(py, &tensor)
}

fn scalar_float_tensor(py: Python<'_>, value: f64, dtype: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    float_tensor(py, &[value], &[], dtype)
}

#[pyclass(name = "HDF5GraphDataset", unsendable)]
pub struct PyHDF5GraphDataset {
    path: String,
    r_max: f64,
    atomic_numbers: Vec<i64>,
    z_to_index: HashMap<i64, usize>,
    positions: RtTensorDataset,
    forces: RtTensorDataset,
    atomic_numbers_dataset: RtTensorDataset,
    metadata: RtHDF5GraphMetadataDataset,
}

impl PyHDF5GraphDataset {
    fn from_path_string(
        path: String,
        r_max: f64,
        atomic_numbers: Vec<i64>,
        niggli_reduce: bool,
    ) -> PyResult<Self> {
        if niggli_reduce {
            return Err(PyNotImplementedError::new_err(
                "rtorch.HDF5GraphDataset does not support niggli_reduce yet.",
            ));
        }
        if atomic_numbers.is_empty() {
            return Err(PyValueError::new_err("atomic_numbers must not be empty."));
        }

        let z_to_index = atomic_numbers
            .iter()
            .enumerate()
            .map(|(index, atomic_number)| (*atomic_number, index))
            .collect::<HashMap<_, _>>();
        let metadata =
            RtHDF5GraphMetadataDataset::from_file(PathBuf::from(&path)).map_err(to_py_err)?;
        let positions = RtTensorDataset::from_hdf5_file(PathBuf::from(&path), "positions")
            .map_err(to_py_err)?;
        let forces =
            RtTensorDataset::from_hdf5_file(PathBuf::from(&path), "forces").map_err(to_py_err)?;
        let atomic_numbers_dataset =
            RtTensorDataset::from_hdf5_file(PathBuf::from(&path), "atomic_numbers")
                .map_err(to_py_err)?;

        let length = metadata.len();
        if positions.len() != length
            || forces.len() != length
            || atomic_numbers_dataset.len() != length
        {
            return Err(PyValueError::new_err(
                "RTorch tensor datasets do not match structures length.",
            ));
        }

        Ok(Self {
            path,
            r_max,
            atomic_numbers,
            z_to_index,
            positions,
            forces,
            atomic_numbers_dataset,
            metadata,
        })
    }

    fn normalized_index(&self, index: isize) -> PyResult<usize> {
        let length = self.metadata.len() as isize;
        let normalized = if index < 0 { length + index } else { index };
        if normalized < 0 || normalized >= length {
            return Err(PyIndexError::new_err(index));
        }
        Ok(normalized as usize)
    }

    fn graph_at(&self, py: Python<'_>, index: usize) -> PyResult<GraphBuildResult> {
        let entry = self.metadata.get(index).map_err(to_py_err)?;
        let positions_sample = self.positions.get(index).map_err(to_py_err)?;
        let forces_sample = self.forces.get(index).map_err(to_py_err)?;
        let atomic_numbers_sample = self.atomic_numbers_dataset.get(index).map_err(to_py_err)?;

        let positions_tensor = positions_sample.tensor().map_err(to_py_err)?;
        let forces_tensor = forces_sample.tensor().map_err(to_py_err)?;
        let atomic_numbers_tensor = atomic_numbers_sample.tensor().map_err(to_py_err)?;

        let positions_values = positions_tensor.to_vec_f64().map_err(to_py_err)?;
        let atomic_number_values = atomic_numbers_tensor.to_vec_i64().map_err(to_py_err)?;
        if !positions_values.len().is_multiple_of(3) {
            return Err(PyValueError::new_err("positions must have shape [N, 3]."));
        }
        let natoms = positions_values.len() / 3;
        if forces_tensor.numel() != positions_values.len() || atomic_number_values.len() != natoms {
            return Err(PyValueError::new_err(format!(
                "Inconsistent atom count at index {index}."
            )));
        }
        if entry.length >= 0 && entry.length as usize != natoms {
            return Err(PyValueError::new_err(format!(
                "Structure metadata length does not match positions at index {index}."
            )));
        }

        let (default_dtype, long_dtype) = torch_default_dtype(py)?;
        let positions = reshape_contiguous(
            py,
            rt_to_torch_dtype(py, &positions_tensor, &default_dtype)?,
            vec![natoms as i64, 3],
        )?;
        let forces = reshape_contiguous(
            py,
            rt_to_torch_dtype(py, &forces_tensor, &default_dtype)?,
            vec![natoms as i64, 3],
        )?;
        let atomic_numbers = reshape_contiguous(
            py,
            rt_to_torch_dtype(py, &atomic_numbers_tensor, &long_dtype)?,
            vec![natoms as i64],
        )?;

        let mut node_attrs = vec![0.0_f64; natoms * self.atomic_numbers.len()];
        for (atom_index, atomic_number) in atomic_number_values.iter().enumerate() {
            let Some(species_index) = self.z_to_index.get(atomic_number) else {
                return Err(PyValueError::new_err(format!(
                    "Observed atom type {atomic_number} that is not listed in the atomic numbers table."
                )));
            };
            node_attrs[atom_index * self.atomic_numbers.len() + species_index] = 1.0;
        }

        let neighborhood = get_neighborhood(&positions_values, self.r_max, entry.pbc, entry.cell)?;
        let edge_count = neighborhood.edge_index.len() / 2;
        let stress = voigt_to_matrix(entry.stress);
        let pbc_values = entry.pbc;
        let tags = vec![0.0_f64; natoms];

        let data_module = py.import_bound("torch_geometric.data")?;
        let data_cls = data_module.getattr("Data")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item(
            "cell",
            float_tensor(py, &neighborhood.cell, &[3, 3], &default_dtype)?,
        )?;
        kwargs.set_item("cell_volume", determinant(&entry.cell).abs())?;
        kwargs.set_item("pos", positions.clone_ref(py))?;
        kwargs.set_item("positions", positions)?;
        kwargs.set_item(
            "node_attrs",
            float_tensor(
                py,
                &node_attrs,
                &[natoms, self.atomic_numbers.len()],
                &default_dtype,
            )?,
        )?;
        kwargs.set_item("atomic_numbers", atomic_numbers)?;
        kwargs.set_item("natoms", natoms)?;
        kwargs.set_item("tags", float_tensor(py, &tags, &[natoms], &default_dtype)?)?;
        kwargs.set_item(
            "edge_index",
            i64_tensor(py, &neighborhood.edge_index, &[2, edge_count])?,
        )?;
        kwargs.set_item(
            "shifts",
            float_tensor(py, &neighborhood.shifts, &[edge_count, 3], &default_dtype)?,
        )?;
        kwargs.set_item(
            "unit_shifts",
            float_tensor(
                py,
                &neighborhood.unit_shifts,
                &[edge_count, 3],
                &default_dtype,
            )?,
        )?;
        kwargs.set_item("y", scalar_float_tensor(py, entry.energy, &default_dtype)?)?;
        kwargs.set_item("force", forces)?;
        kwargs.set_item(
            "stress",
            float_tensor(py, &stress, &[1, 3, 3], &default_dtype)?,
        )?;
        kwargs.set_item("pbc", bool_tensor(py, &pbc_values, &[3])?)?;

        let graph = data_cls.call((), Some(&kwargs))?;
        graph.setattr("idx", index)?;

        Ok(GraphBuildResult {
            graph: graph.into_py(py),
            num_nodes: natoms,
            num_edges: edge_count,
        })
    }
}

#[pymethods]
impl PyHDF5GraphDataset {
    #[new]
    #[pyo3(signature = (path, r_max, atomic_numbers, *, niggli_reduce = false))]
    fn new(
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        r_max: f64,
        atomic_numbers: Vec<i64>,
        niggli_reduce: bool,
    ) -> PyResult<Self> {
        Self::from_path_string(
            path_to_string(py, path)?,
            r_max,
            atomic_numbers,
            niggli_reduce,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (path, r_max, atomic_numbers, *, niggli_reduce = false))]
    fn from_hdf5_file(
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        r_max: f64,
        atomic_numbers: Vec<i64>,
        niggli_reduce: bool,
    ) -> PyResult<Self> {
        Self::new(py, path, r_max, atomic_numbers, niggli_reduce)
    }

    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    fn close(&mut self) {}

    fn __len__(&self) -> usize {
        self.metadata.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<PyObject> {
        let index = self.normalized_index(index)?;
        self.graph_at(py, index).map(|result| result.graph)
    }
}

#[pyclass(name = "HDF5GraphDatasetCollection", unsendable)]
pub struct PyHDF5GraphDatasetCollection {
    datasets: Vec<PyHDF5GraphDataset>,
    offsets: Vec<usize>,
    length: usize,
}

impl PyHDF5GraphDatasetCollection {
    fn from_paths(
        paths: Vec<String>,
        r_max: f64,
        atomic_numbers: Vec<i64>,
        niggli_reduce: bool,
    ) -> PyResult<Self> {
        if paths.is_empty() {
            return Err(PyValueError::new_err("expected at least one HDF5 path."));
        }
        let mut datasets = Vec::with_capacity(paths.len());
        let mut offsets = Vec::with_capacity(paths.len());
        let mut length = 0_usize;
        for path in paths {
            offsets.push(length);
            let dataset = PyHDF5GraphDataset::from_path_string(
                path,
                r_max,
                atomic_numbers.clone(),
                niggli_reduce,
            )?;
            length += dataset.metadata.len();
            datasets.push(dataset);
        }
        Ok(Self {
            datasets,
            offsets,
            length,
        })
    }

    fn normalized_index(&self, index: isize) -> PyResult<usize> {
        let length = self.length as isize;
        let normalized = if index < 0 { length + index } else { index };
        if normalized < 0 || normalized >= length {
            return Err(PyIndexError::new_err(index));
        }
        Ok(normalized as usize)
    }

    fn dataset_for_index(&self, index: usize) -> (usize, usize) {
        let insertion = match self.offsets.binary_search(&index) {
            Ok(value) => value,
            Err(value) => value - 1,
        };
        (insertion, index - self.offsets[insertion])
    }

    fn graph_at(&self, py: Python<'_>, index: usize) -> PyResult<GraphBuildResult> {
        let (dataset_index, local_index) = self.dataset_for_index(index);
        self.datasets[dataset_index].graph_at(py, local_index)
    }
}

#[pymethods]
impl PyHDF5GraphDatasetCollection {
    #[staticmethod]
    #[pyo3(signature = (paths, r_max, atomic_numbers, *, niggli_reduce = false))]
    fn from_hdf5_files(
        paths: Vec<String>,
        r_max: f64,
        atomic_numbers: Vec<i64>,
        niggli_reduce: bool,
    ) -> PyResult<Self> {
        Self::from_paths(paths, r_max, atomic_numbers, niggli_reduce)
    }

    fn close(&mut self) {}

    fn __len__(&self) -> usize {
        self.length
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<PyObject> {
        let index = self.normalized_index(index)?;
        self.graph_at(py, index).map(|result| result.graph)
    }
}

#[pyclass(name = "HDF5GraphLoader", unsendable)]
pub struct PyHDF5GraphLoader {
    dataset: Py<PyHDF5GraphDatasetCollection>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    max_nodes: Option<usize>,
    max_edges: Option<usize>,
    drop: bool,
    seed: Option<u64>,
    indices: Vec<usize>,
    cursor: usize,
}

impl PyHDF5GraphLoader {
    fn new_from_collection(
        py: Python<'_>,
        dataset: PyHDF5GraphDatasetCollection,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        max_nodes: Option<usize>,
        max_edges: Option<usize>,
        drop: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive."));
        }
        let length = dataset.length;
        let dataset = Py::new(py, dataset)?;
        let mut loader = Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            max_nodes,
            max_edges,
            drop,
            seed,
            indices: Vec::new(),
            cursor: 0,
        };
        loader.reset_indices(length);
        Ok(loader)
    }

    fn dataset_len(&self, py: Python<'_>) -> usize {
        self.dataset.bind(py).borrow().length
    }

    fn reset_indices(&mut self, length: usize) {
        self.indices = (0..length).collect();
        if self.shuffle {
            shuffle_indices(&mut self.indices, self.seed);
        }
        self.cursor = 0;
    }
}

#[pymethods]
impl PyHDF5GraphLoader {
    #[staticmethod]
    #[pyo3(signature = (path, r_max, atomic_numbers, *, batch_size = 1, shuffle = false, drop_last = false, max_nodes = None, max_edges = None, drop = false, seed = None, niggli_reduce = false))]
    fn from_hdf5_file(
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        r_max: f64,
        atomic_numbers: Vec<i64>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        max_nodes: Option<usize>,
        max_edges: Option<usize>,
        drop: bool,
        seed: Option<u64>,
        niggli_reduce: bool,
    ) -> PyResult<Self> {
        let collection = PyHDF5GraphDatasetCollection::from_paths(
            vec![path_to_string(py, path)?],
            r_max,
            atomic_numbers,
            niggli_reduce,
        )?;
        Self::new_from_collection(
            py, collection, batch_size, shuffle, drop_last, max_nodes, max_edges, drop, seed,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (paths, r_max, atomic_numbers, *, batch_size = 1, shuffle = false, drop_last = false, max_nodes = None, max_edges = None, drop = false, seed = None, niggli_reduce = false))]
    fn from_hdf5_files(
        py: Python<'_>,
        paths: Vec<String>,
        r_max: f64,
        atomic_numbers: Vec<i64>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        max_nodes: Option<usize>,
        max_edges: Option<usize>,
        drop: bool,
        seed: Option<u64>,
        niggli_reduce: bool,
    ) -> PyResult<Self> {
        let collection =
            PyHDF5GraphDatasetCollection::from_paths(paths, r_max, atomic_numbers, niggli_reduce)?;
        Self::new_from_collection(
            py, collection, batch_size, shuffle, drop_last, max_nodes, max_edges, drop, seed,
        )
    }

    #[getter]
    fn dataset(&self, py: Python<'_>) -> PyObject {
        self.dataset.clone_ref(py).into_py(py)
    }

    #[getter]
    fn _uses_rtorch(&self) -> bool {
        true
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        let length = self.dataset_len(py);
        if self.drop_last {
            length / self.batch_size
        } else {
            length.div_ceil(self.batch_size)
        }
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        let length = Python::with_gil(|py| slf.dataset_len(py));
        slf.reset_indices(length);
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.cursor >= self.indices.len() {
            return Ok(None);
        }

        let end = (self.cursor + self.batch_size).min(self.indices.len());
        if self.drop_last && end - self.cursor < self.batch_size {
            self.cursor = self.indices.len();
            return Ok(None);
        }

        let dataset = self.dataset.bind(py).borrow();
        let mut graphs = Vec::with_capacity(end - self.cursor);
        for index in &self.indices[self.cursor..end] {
            graphs.push(dataset.graph_at(py, *index)?);
        }
        self.cursor = end;
        pack_dynamic_batches(py, graphs, self.max_nodes, self.max_edges, self.drop).map(Some)
    }
}

fn pack_dynamic_batches(
    py: Python<'_>,
    graphs: Vec<GraphBuildResult>,
    max_nodes: Option<usize>,
    max_edges: Option<usize>,
    drop: bool,
) -> PyResult<PyObject> {
    let batch_cls = py.import_bound("torch_geometric.data")?.getattr("Batch")?;
    let batches = PyList::empty_bound(py);
    let mut current_batch = Vec::new();
    let mut current_node_sum = 0_usize;
    let mut current_edge_sum = 0_usize;

    for item in graphs {
        if max_edges.is_some_and(|limit| drop && item.num_edges > limit) {
            continue;
        }
        if max_nodes.is_some_and(|limit| drop && item.num_nodes > limit) {
            continue;
        }

        if !current_batch.is_empty() {
            if max_nodes.is_some_and(|limit| current_node_sum + item.num_nodes > limit) {
                append_pyg_batch(py, &batch_cls, &batches, &mut current_batch)?;
                current_node_sum = 0;
                current_edge_sum = 0;
            }
            if max_edges.is_some_and(|limit| current_edge_sum + item.num_edges > limit) {
                append_pyg_batch(py, &batch_cls, &batches, &mut current_batch)?;
                current_node_sum = 0;
                current_edge_sum = 0;
            }
        }

        current_node_sum += item.num_nodes;
        current_edge_sum += item.num_edges;
        current_batch.push(item.graph);
    }

    if !current_batch.is_empty() {
        append_pyg_batch(py, &batch_cls, &batches, &mut current_batch)?;
    }

    Ok(batches.into_py(py))
}

fn append_pyg_batch(
    py: Python<'_>,
    batch_cls: &Bound<'_, PyAny>,
    batches: &Bound<'_, PyList>,
    current_batch: &mut Vec<PyObject>,
) -> PyResult<()> {
    let data_list = PyList::empty_bound(py);
    for graph in current_batch.iter() {
        data_list.append(graph.bind(py))?;
    }
    let batch = batch_cls.getattr("from_data_list")?.call1((data_list,))?;
    batches.append(batch)?;
    current_batch.clear();
    Ok(())
}

fn shuffle_indices(indices: &mut [usize], seed: Option<u64>) {
    let mut state = seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0x9e37_79b9_7f4a_7c15)
    });
    if state == 0 {
        state = 0x9e37_79b9_7f4a_7c15;
    }

    for index in (1..indices.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let swap_index = (state % (index as u64 + 1)) as usize;
        indices.swap(index, swap_index);
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHDF5GraphDataset>()?;
    module.add_class::<PyHDF5GraphDatasetCollection>()?;
    module.add_class::<PyHDF5GraphLoader>()?;
    Ok(())
}
