use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyList};

use equiload_core::{
    HDF5GraphMetadataDataset as RtHDF5GraphMetadataDataset, HDF5GraphStructure, Tensor as RtTensor,
    TensorDataset as RtTensorDataset,
};

use crate::{tensor_to_torch, to_py_err};

static DATA_BATCH_CLASS: GILOnceCell<PyObject> = GILOnceCell::new();

struct PreparedGraph {
    entry: HDF5GraphStructure,
    positions: Vec<f64>,
    forces: Vec<f64>,
    atomic_numbers: Vec<i64>,
    species: Vec<usize>,
    species_count: usize,
    neighborhood: Neighborhood,
    index: usize,
}

impl PreparedGraph {
    fn num_nodes(&self) -> usize {
        self.atomic_numbers.len()
    }
    fn num_edges(&self) -> usize {
        self.neighborhood.edge_index.len() / 2
    }
}

struct Neighborhood {
    edge_index: Vec<i64>,
    shifts: Vec<f64>,
    unit_shifts: Vec<f64>,
    cell: [f64; 9],
}

#[derive(Clone, Copy)]
struct Translation {
    unit: [i64; 3],
    cartesian: [f64; 3],
}

#[derive(Clone, Copy)]
struct ImageAtom {
    receiver: usize,
    translation: usize,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct BinKey(i64, i64, i64);

// Integer spatial keys do not need a string-oriented cryptographic hash.
// Keep the species table on the standard hasher; specialize only bin lookups.
#[derive(Default)]
struct SpatialHasher(u64);
impl Hasher for SpatialHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.write_u64(*byte as u64);
        }
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }
}
type SpatialMap<V> = HashMap<BinKey, V, BuildHasherDefault<SpatialHasher>>;

#[derive(Clone, Copy)]
struct EdgeRecord {
    sender: usize,
    receiver: usize,
    unit_shift: [i64; 3],
}

struct PreparedNeighborhood {
    positions: Vec<[f64; 3]>,
    wrapped_positions: Vec<[f64; 3]>,
    wrap_offsets: Vec<[i64; 3]>,
    translations: Vec<Translation>,
    cutoff: f64,
    cutoff2: f64,
    cell: [f64; 9],
}

// A spatial hash has higher setup cost than direct comparisons for tiny graphs.
// Keep the reference algorithm for those cases and use the cell list once the
// number of candidate pair/image checks becomes meaningful.
const BRUTE_FORCE_CHECK_LIMIT: usize = 8_192;
const MAX_EXPANDED_IMAGE_ATOMS: usize = 2_000_000;

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

fn inverse(cell: &[f64; 9]) -> Option<[f64; 9]> {
    let det = determinant(cell);
    if !det.is_finite() || det.abs() <= 1.0e-12 {
        return None;
    }

    Some([
        (cell[4] * cell[8] - cell[5] * cell[7]) / det,
        (cell[2] * cell[7] - cell[1] * cell[8]) / det,
        (cell[1] * cell[5] - cell[2] * cell[4]) / det,
        (cell[5] * cell[6] - cell[3] * cell[8]) / det,
        (cell[0] * cell[8] - cell[2] * cell[6]) / det,
        (cell[2] * cell[3] - cell[0] * cell[5]) / det,
        (cell[3] * cell[7] - cell[4] * cell[6]) / det,
        (cell[1] * cell[6] - cell[0] * cell[7]) / det,
        (cell[0] * cell[4] - cell[1] * cell[3]) / det,
    ])
}

fn row_vector_matrix(vector: [f64; 3], matrix: &[f64; 9]) -> [f64; 3] {
    [
        vector[0] * matrix[0] + vector[1] * matrix[3] + vector[2] * matrix[6],
        vector[0] * matrix[1] + vector[1] * matrix[4] + vector[2] * matrix[7],
        vector[0] * matrix[2] + vector[1] * matrix[5] + vector[2] * matrix[8],
    ]
}

fn lattice_shift(unit: [i64; 3], cell: &[f64; 9]) -> [f64; 3] {
    [
        unit[0] as f64 * cell[0] + unit[1] as f64 * cell[3] + unit[2] as f64 * cell[6],
        unit[0] as f64 * cell[1] + unit[1] as f64 * cell[4] + unit[2] as f64 * cell[7],
        unit[0] as f64 * cell[2] + unit[1] as f64 * cell[5] + unit[2] as f64 * cell[8],
    ]
}

fn translation_ranges(cutoff: f64, pbc: [bool; 3], cell: &[f64; 9]) -> PyResult<[i32; 3]> {
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
        if !height.is_finite() || height <= 1.0e-12 {
            return Err(PyValueError::new_err(
                "periodic cell height must be finite and positive.",
            ));
        }
        let range = (cutoff / height).ceil();
        if range > f64::from(i32::MAX) {
            return Err(PyValueError::new_err(
                "periodic translation range exceeds the supported integer range.",
            ));
        }
        ranges[axis] = range as i32;
    }

    Ok(ranges)
}

fn get_neighborhood(
    positions: &[f64],
    cutoff: f64,
    pbc: [bool; 3],
    cell: [f64; 9],
) -> PyResult<Neighborhood> {
    let prepared = prepare_neighborhood(positions, cutoff, pbc, cell)?;
    let natoms = prepared.positions.len();
    let direct_checks = natoms
        .checked_mul(natoms)
        .and_then(|value| value.checked_mul(prepared.translations.len()))
        .unwrap_or(usize::MAX);

    if direct_checks <= BRUTE_FORCE_CHECK_LIMIT {
        get_neighborhood_brute_force(&prepared)
    } else {
        get_neighborhood_cell_list(&prepared)
    }
}

fn prepare_neighborhood(
    positions: &[f64],
    cutoff: f64,
    pbc: [bool; 3],
    mut cell: [f64; 9],
) -> PyResult<PreparedNeighborhood> {
    if positions.is_empty() || !positions.len().is_multiple_of(3) {
        return Err(PyValueError::new_err("positions must have shape [N, 3]."));
    }
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(PyValueError::new_err("r_max must be finite and positive."));
    }
    let cutoff2 = cutoff * cutoff;
    if !cutoff2.is_finite() {
        return Err(PyValueError::new_err("r_max is too large."));
    }
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "positions must contain finite values.",
        ));
    }
    if cell.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("cell must contain finite values."));
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

    if cell.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "extended cell contains non-finite values.",
        ));
    }

    let natoms = positions.len() / 3;
    let atom_positions = positions.as_chunks::<3>().0.to_vec();
    let mut wrapped_positions = atom_positions.clone();
    let mut wrap_offsets = vec![[0_i64; 3]; natoms];

    if pbc.iter().any(|periodic| *periodic) {
        let inverse_cell = inverse(&cell).ok_or_else(|| {
            PyValueError::new_err("cell must be invertible when periodic boundaries are enabled.")
        })?;

        for atom in 0..natoms {
            let fractional = row_vector_matrix(atom_positions[atom], &inverse_cell);
            for axis in 0..3 {
                if !pbc[axis] {
                    continue;
                }
                let image = fractional[axis].floor();
                if image < i64::MIN as f64 || image > i64::MAX as f64 {
                    return Err(PyValueError::new_err(
                        "periodic image offset exceeds the supported integer range.",
                    ));
                }
                wrap_offsets[atom][axis] = image as i64;
            }

            let wrapping_shift = lattice_shift(wrap_offsets[atom], &cell);
            for axis in 0..3 {
                wrapped_positions[atom][axis] -= wrapping_shift[axis];
            }
        }
    }

    // Once periodic positions are wrapped into the primary cell, ceil(r/h)
    // images along a lattice direction are sufficient, where h is the
    // corresponding cell height. The previous unwrapped search needed an
    // extra image and often expanded 3-D searches from 27 to 125 cells.
    let ranges = translation_ranges(cutoff, pbc, &cell)?;
    let translation_count = ranges.iter().try_fold(1_usize, |count, range| {
        let width = 2_i64 * i64::from(*range) + 1;
        usize::try_from(width)
            .ok()
            .and_then(|width| count.checked_mul(width))
    });
    let translation_count = translation_count.ok_or_else(|| {
        PyValueError::new_err("periodic translation count exceeds the supported range.")
    })?;
    let mut translations = Vec::with_capacity(translation_count);
    for sx in -ranges[0]..=ranges[0] {
        for sy in -ranges[1]..=ranges[1] {
            for sz in -ranges[2]..=ranges[2] {
                let unit = [i64::from(sx), i64::from(sy), i64::from(sz)];
                translations.push(Translation {
                    unit,
                    cartesian: lattice_shift(unit, &cell),
                });
            }
        }
    }

    Ok(PreparedNeighborhood {
        positions: atom_positions,
        wrapped_positions,
        wrap_offsets,
        translations,
        cutoff,
        cutoff2,
        cell,
    })
}

fn edge_record(
    prepared: &PreparedNeighborhood,
    sender: usize,
    receiver: usize,
    translation: Translation,
) -> PyResult<Option<EdgeRecord>> {
    if sender == receiver && translation.unit == [0, 0, 0] {
        return Ok(None);
    }

    let displacement = [
        prepared.wrapped_positions[receiver][0] - prepared.wrapped_positions[sender][0]
            + translation.cartesian[0],
        prepared.wrapped_positions[receiver][1] - prepared.wrapped_positions[sender][1]
            + translation.cartesian[1],
        prepared.wrapped_positions[receiver][2] - prepared.wrapped_positions[sender][2]
            + translation.cartesian[2],
    ];
    let dist2 = displacement[0] * displacement[0]
        + displacement[1] * displacement[1]
        + displacement[2] * displacement[2];
    // Match the Python/matscipy neighbor list's strict cutoff.
    if dist2 >= prepared.cutoff2 {
        return Ok(None);
    }

    let mut unit_shift = [0_i64; 3];
    for (axis, unit) in unit_shift.iter_mut().enumerate() {
        *unit = translation.unit[axis]
            .checked_add(prepared.wrap_offsets[sender][axis])
            .and_then(|value| value.checked_sub(prepared.wrap_offsets[receiver][axis]))
            .ok_or_else(|| PyValueError::new_err("periodic image offset overflowed."))?;
    }

    Ok(Some(EdgeRecord {
        sender,
        receiver,
        unit_shift,
    }))
}

fn get_neighborhood_brute_force(prepared: &PreparedNeighborhood) -> PyResult<Neighborhood> {
    let natoms = prepared.positions.len();
    let mut records = Vec::new();
    for sender in 0..natoms {
        for receiver in 0..natoms {
            for translation in &prepared.translations {
                if let Some(record) = edge_record(prepared, sender, receiver, *translation)? {
                    records.push(record);
                }
            }
        }
    }
    records_to_neighborhood(records, prepared.cell)
}

fn bin_key(position: [f64; 3], inverse_bin_size: f64) -> BinKey {
    BinKey(
        (position[0] * inverse_bin_size).floor() as i64,
        (position[1] * inverse_bin_size).floor() as i64,
        (position[2] * inverse_bin_size).floor() as i64,
    )
}

fn get_neighborhood_cell_list(prepared: &PreparedNeighborhood) -> PyResult<Neighborhood> {
    let expanded_atoms = prepared
        .positions
        .len()
        .saturating_mul(prepared.translations.len());
    if expanded_atoms > MAX_EXPANDED_IMAGE_ATOMS {
        return get_neighborhood_translation_queries(prepared);
    }

    let natoms = prepared.positions.len();
    let inverse_bin_size = prepared.cutoff.recip();
    let mut bins: SpatialMap<Vec<ImageAtom>> = SpatialMap::default();

    for receiver in 0..natoms {
        for (translation_index, translation) in prepared.translations.iter().enumerate() {
            let image_position = [
                prepared.wrapped_positions[receiver][0] + translation.cartesian[0],
                prepared.wrapped_positions[receiver][1] + translation.cartesian[1],
                prepared.wrapped_positions[receiver][2] + translation.cartesian[2],
            ];
            bins.entry(bin_key(image_position, inverse_bin_size))
                .or_default()
                .push(ImageAtom {
                    receiver,
                    translation: translation_index,
                });
        }
    }

    let mut records = Vec::new();
    for sender in 0..natoms {
        let sender_bin = bin_key(prepared.wrapped_positions[sender], inverse_bin_size);
        for dx in -1_i64..=1 {
            let Some(bin_x) = sender_bin.0.checked_add(dx) else {
                continue;
            };
            for dy in -1_i64..=1 {
                let Some(bin_y) = sender_bin.1.checked_add(dy) else {
                    continue;
                };
                for dz in -1_i64..=1 {
                    let Some(bin_z) = sender_bin.2.checked_add(dz) else {
                        continue;
                    };
                    let Some(images) = bins.get(&BinKey(bin_x, bin_y, bin_z)) else {
                        continue;
                    };
                    for image in images {
                        let translation = prepared.translations[image.translation];
                        if let Some(record) =
                            edge_record(prepared, sender, image.receiver, translation)?
                        {
                            records.push(record);
                        }
                    }
                }
            }
        }
    }

    records_to_neighborhood(records, prepared.cell)
}

fn get_neighborhood_translation_queries(prepared: &PreparedNeighborhood) -> PyResult<Neighborhood> {
    let natoms = prepared.positions.len();
    let inverse_bin_size = prepared.cutoff.recip();
    let mut bins: SpatialMap<Vec<usize>> = SpatialMap::default();
    for receiver in 0..natoms {
        bins.entry(bin_key(
            prepared.wrapped_positions[receiver],
            inverse_bin_size,
        ))
        .or_default()
        .push(receiver);
    }

    let mut records = Vec::new();
    for sender in 0..natoms {
        for translation in &prepared.translations {
            // receiver + translation must be close to sender, so query the
            // unexpanded receiver bins around sender - translation.
            let query_position = [
                prepared.wrapped_positions[sender][0] - translation.cartesian[0],
                prepared.wrapped_positions[sender][1] - translation.cartesian[1],
                prepared.wrapped_positions[sender][2] - translation.cartesian[2],
            ];
            let query_bin = bin_key(query_position, inverse_bin_size);
            for dx in -1_i64..=1 {
                let Some(bin_x) = query_bin.0.checked_add(dx) else {
                    continue;
                };
                for dy in -1_i64..=1 {
                    let Some(bin_y) = query_bin.1.checked_add(dy) else {
                        continue;
                    };
                    for dz in -1_i64..=1 {
                        let Some(bin_z) = query_bin.2.checked_add(dz) else {
                            continue;
                        };
                        let Some(receivers) = bins.get(&BinKey(bin_x, bin_y, bin_z)) else {
                            continue;
                        };
                        for receiver in receivers {
                            if let Some(record) =
                                edge_record(prepared, sender, *receiver, *translation)?
                            {
                                records.push(record);
                            }
                        }
                    }
                }
            }
        }
    }

    records_to_neighborhood(records, prepared.cell)
}

fn records_to_neighborhood(records: Vec<EdgeRecord>, cell: [f64; 9]) -> PyResult<Neighborhood> {
    // Every search iterates senders in ascending order, matching Matscipy's
    // ordering guarantee. It deliberately does not sort receivers: sorting is
    // not part of that contract and adds O(E log E) work to every structure.
    let mut edge_index = Vec::with_capacity(2 * records.len());
    edge_index.extend(records.iter().map(|record| record.sender as i64));
    edge_index.extend(records.iter().map(|record| record.receiver as i64));

    let mut shifts = Vec::with_capacity(3 * records.len());
    let mut unit_shifts = Vec::with_capacity(3 * records.len());
    for record in records {
        let shift = lattice_shift(record.unit_shift, &cell);
        if shift.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "periodic Cartesian shift contains non-finite values.",
            ));
        }
        shifts.extend_from_slice(&shift);
        unit_shifts.extend(record.unit_shift.map(|value| value as f64));
    }

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

fn float_tensor(
    py: Python<'_>,
    values: Vec<f64>,
    shape: &[usize],
    dtype: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    // Allocate the final dtype directly for the common float32/float64 paths.
    // Geometry and cutoff decisions remain double precision.
    match dtype.str()?.to_str()? {
        "torch.float32" => {
            let values: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
            let tensor = RtTensor::from_vec(values, shape).map_err(to_py_err)?;
            rt_to_torch(py, &tensor)
        }
        "torch.float64" => {
            let tensor = RtTensor::from_vec(values, shape).map_err(to_py_err)?;
            rt_to_torch(py, &tensor)
        }
        _ => {
            let tensor = RtTensor::from_vec(values, shape).map_err(to_py_err)?;
            rt_to_torch_dtype(py, &tensor, dtype)
        }
    }
}

fn i64_tensor(py: Python<'_>, values: Vec<i64>, shape: &[usize]) -> PyResult<PyObject> {
    let tensor = RtTensor::from_vec(values, shape).map_err(to_py_err)?;
    rt_to_torch(py, &tensor)
}

fn bool_tensor(py: Python<'_>, values: Vec<bool>, shape: &[usize]) -> PyResult<PyObject> {
    let tensor = RtTensor::from_vec(values, shape).map_err(to_py_err)?;
    rt_to_torch(py, &tensor)
}

fn scalar_float_tensor(py: Python<'_>, value: f64, dtype: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    float_tensor(py, vec![value], &[], dtype)
}

#[pyclass(name = "HDF5GraphDataset", unsendable)]
pub struct PyHDF5GraphDataset {
    path: String,
    r_max: f64,
    atomic_numbers: Vec<i64>,
    z_to_index: HashMap<i64, usize>,
    handles: Option<GraphHandles>,
    length: usize,
}

struct GraphHandles {
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
                "equiload.HDF5GraphDataset does not support niggli_reduce yet.",
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
                "EquiLoad tensor datasets do not match structures length.",
            ));
        }

        Ok(Self {
            path,
            r_max,
            atomic_numbers,
            z_to_index,
            handles: Some(GraphHandles {
                positions,
                forces,
                atomic_numbers_dataset,
                metadata,
            }),
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

    // This phase creates no Python objects. Its buffers live for one outer batch.
    fn prepare_graph(&self, py: Python<'_>, index: usize) -> PyResult<PreparedGraph> {
        let handles = self.handles.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("HDF5 graph dataset is closed")
        })?;
        let entry = handles.metadata.get(index).map_err(to_py_err)?;
        if entry.offset < 0 || entry.length < 0 {
            return Err(PyValueError::new_err(
                "HDF5 structures contain negative offset or length",
            ));
        }
        let offset = entry.offset as u64;
        let length = entry.length as u64;
        let positions = handles
            .positions
            .read_hdf5_slice(offset, length)
            .map_err(to_py_err)?
            .to_vec_f64()
            .map_err(to_py_err)?;
        let forces = handles
            .forces
            .read_hdf5_slice(offset, length)
            .map_err(to_py_err)?
            .to_vec_f64()
            .map_err(to_py_err)?;
        let atomic_numbers = handles
            .atomic_numbers_dataset
            .read_hdf5_slice(offset, length)
            .map_err(to_py_err)?
            .to_vec_i64()
            .map_err(to_py_err)?;
        if !positions.len().is_multiple_of(3) {
            return Err(PyValueError::new_err("positions must have shape [N, 3]."));
        }
        let natoms = positions.len() / 3;
        if forces.len() != positions.len() || atomic_numbers.len() != natoms {
            return Err(PyValueError::new_err(format!(
                "Inconsistent atom count at index {index}."
            )));
        }
        if entry.length >= 0 && entry.length as usize != natoms {
            return Err(PyValueError::new_err(format!(
                "Structure metadata length does not match positions at index {index}."
            )));
        }
        let species = atomic_numbers
            .iter()
            .map(|z| {
                self.z_to_index.get(z).copied().ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Observed atom type {z} that is not listed in the atomic numbers table."
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let r_max = self.r_max;
        let pbc = entry.pbc;
        let cell = entry.cell;
        let neighborhood = py.allow_threads(|| get_neighborhood(&positions, r_max, pbc, cell))?;
        Ok(PreparedGraph {
            entry,
            positions,
            forces,
            atomic_numbers,
            species,
            species_count: self.atomic_numbers.len(),
            neighborhood,
            index,
        })
    }

    fn graph_at(&self, py: Python<'_>, index: usize) -> PyResult<PyObject> {
        let graph = self.prepare_graph(py, index)?;
        materialize_graphs(py, vec![graph], false)
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

    fn close(&mut self) {
        self.handles = None;
    }

    fn __len__(&self) -> usize {
        self.length
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<PyObject> {
        let index = self.normalized_index(index)?;
        self.graph_at(py, index)
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
            length += dataset.length;
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
        let insertion = self.offsets.partition_point(|offset| *offset <= index) - 1;
        (insertion, index - self.offsets[insertion])
    }

    fn graph_at(&self, py: Python<'_>, index: usize) -> PyResult<PyObject> {
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

    fn close(&mut self) {
        for dataset in &mut self.datasets {
            dataset.close();
        }
    }

    fn __len__(&self) -> usize {
        self.length
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<PyObject> {
        let index = self.normalized_index(index)?;
        self.graph_at(py, index)
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
        if self.shuffle {
            self.indices.clear();
            self.indices.extend(0..length);
            shuffle_indices(&mut self.indices, self.seed);
        } else {
            self.indices.clear();
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
    fn _uses_equiload(&self) -> bool {
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
        let length = self.dataset_len(py);
        if self.cursor >= length {
            return Ok(None);
        }

        let end = self.cursor.saturating_add(self.batch_size).min(length);
        if self.drop_last && end - self.cursor < self.batch_size {
            self.cursor = length;
            return Ok(None);
        }

        let dataset = self.dataset.bind(py).borrow();
        let graphs = (self.cursor..end).map(|position| {
            let index = if self.shuffle {
                self.indices[position]
            } else {
                position
            };
            let (dataset_index, local_index) = dataset.dataset_for_index(index);
            dataset.datasets[dataset_index].prepare_graph(py, local_index)
        });
        let result = pack_dynamic_batches(py, graphs, self.max_nodes, self.max_edges, self.drop)?;
        self.cursor = end;
        Ok(Some(result))
    }
}

fn pack_dynamic_batches<I>(
    py: Python<'_>,
    graphs: I,
    max_nodes: Option<usize>,
    max_edges: Option<usize>,
    drop: bool,
) -> PyResult<PyObject>
where
    I: IntoIterator<Item = PyResult<PreparedGraph>>,
{
    let (dtype, _) = torch_default_dtype(py)?;
    if dtype.str()?.to_str()? == "torch.float32" {
        pack_dynamic_batches_typed::<f32, I>(py, graphs, max_nodes, max_edges, drop, &dtype)
    } else {
        pack_dynamic_batches_typed::<f64, I>(py, graphs, max_nodes, max_edges, drop, &dtype)
    }
}

fn pack_dynamic_batches_typed<T: BatchFloat, I>(
    py: Python<'_>,
    graphs: I,
    max_nodes: Option<usize>,
    max_edges: Option<usize>,
    drop: bool,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<PyObject>
where
    I: IntoIterator<Item = PyResult<PreparedGraph>>,
{
    let batches = PyList::empty_bound(py);
    let graphs = graphs.into_iter();
    let estimated_count = graphs.size_hint().0;
    let mut current: Option<BatchBuffers<T>> = None;
    for graph in graphs {
        let graph = graph?;
        let n = graph.num_nodes();
        let e = graph.num_edges();
        if drop
            && (max_nodes.is_some_and(|limit| n > limit)
                || max_edges.is_some_and(|limit| e > limit))
        {
            continue;
        }
        if current.as_ref().is_some_and(|b| {
            max_nodes.is_some_and(|limit| b.atomic_numbers.len() + n > limit)
                || max_edges.is_some_and(|limit| b.receivers.len() + e > limit)
        }) {
            batches.append(finish_graph_buffers(
                py,
                current.take().unwrap(),
                true,
                dtype,
            )?)?;
        }
        let buffers = current.get_or_insert_with(|| {
            let count = estimated_count.max(1);
            let count = max_nodes.map_or(count, |limit| count.min((limit / n.max(1)).max(1)));
            let count = max_edges.map_or(count, |limit| count.min((limit / e.max(1)).max(1)));
            BatchBuffers::new(
                graph.species_count,
                count,
                n.saturating_mul(count),
                e.saturating_mul(count),
            )
        });
        buffers.append(graph);
    }
    if let Some(buffers) = current {
        batches.append(finish_graph_buffers(py, buffers, true, dtype)?)?;
    }
    Ok(batches.into_py(py))
}

// Batch assembly is specific to the graph schema, avoiding generic Python
// collation and per-structure torch objects. Slice/inc metadata retains PyG
// indexing, get_example(), and to_data_list() compatibility.
fn materialize_graphs(
    py: Python<'_>,
    graphs: Vec<PreparedGraph>,
    batch: bool,
) -> PyResult<PyObject> {
    let (dtype, _) = torch_default_dtype(py)?;
    if dtype.str()?.to_str()? == "torch.float32" {
        materialize_graphs_typed::<f32>(py, graphs, batch, &dtype)
    } else {
        materialize_graphs_typed::<f64>(py, graphs, batch, &dtype)
    }
}

trait BatchFloat: equiload_core::TensorElement + Send + Copy + 'static {
    const DTYPE: &'static str;
    fn from_f64(value: f64) -> Self;
}
impl BatchFloat for f32 {
    const DTYPE: &'static str = "torch.float32";
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}
impl BatchFloat for f64 {
    const DTYPE: &'static str = "torch.float64";
    fn from_f64(value: f64) -> Self {
        value
    }
}
fn batch_float_tensor<T: BatchFloat>(
    py: Python<'_>,
    values: Vec<T>,
    shape: &[usize],
    dtype: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let tensor = RtTensor::from_vec(values, shape).map_err(to_py_err)?;
    if dtype.str()?.to_str()? == T::DTYPE {
        rt_to_torch(py, &tensor)
    } else {
        rt_to_torch_dtype(py, &tensor, dtype)
    }
}

struct BatchBuffers<T> {
    species_count: usize,
    single_energy: f64,
    single_volume: f64,
    positions: Vec<T>,
    forces: Vec<T>,
    atomic_numbers: Vec<i64>,
    node_attrs: Vec<T>,
    cell: Vec<T>,
    // Equitrain obtains this from ASE's `Cell.volume`, which is a NumPy
    // float64. PyG preserves that type when collating, independently of
    // Torch's default dtype. Keep the same contract here: models may inspect
    // metadata fields in addition to the default-dtype model inputs.
    cell_volume: Vec<f64>,
    signed_volume: Vec<T>,
    rcell: Vec<T>,
    total_charge: Vec<T>,
    total_spin: Vec<T>,
    external_field: Vec<T>,
    source_id: Vec<i64>,
    reaction_id: Vec<i64>,
    state_id: Vec<i64>,
    energy: Vec<T>,
    stress: Vec<T>,
    pbc: Vec<bool>,
    indices: Vec<i64>,
    natoms: Vec<i64>,
    senders: Vec<i64>,
    receivers: Vec<i64>,
    shifts: Vec<T>,
    unit_shifts: Vec<T>,
    assignment: Vec<i64>,
    node_slices: Vec<i64>,
    edge_slices: Vec<i64>,
}

impl<T: BatchFloat> BatchBuffers<T> {
    fn new(species_count: usize, count: usize, nodes: usize, edges: usize) -> Self {
        Self {
            species_count,
            single_energy: 0.0,
            single_volume: 0.0,
            positions: Vec::with_capacity(nodes * 3),
            forces: Vec::with_capacity(nodes * 3),
            atomic_numbers: Vec::with_capacity(nodes),
            node_attrs: Vec::with_capacity(nodes * species_count),
            cell: Vec::with_capacity(count * 9),
            cell_volume: Vec::with_capacity(count),
            signed_volume: Vec::with_capacity(count),
            rcell: Vec::with_capacity(count * 9),
            total_charge: Vec::with_capacity(count),
            total_spin: Vec::with_capacity(count),
            external_field: Vec::with_capacity(count * 3),
            source_id: Vec::with_capacity(count),
            reaction_id: Vec::with_capacity(count),
            state_id: Vec::with_capacity(count),
            energy: Vec::with_capacity(count),
            stress: Vec::with_capacity(count * 9),
            pbc: Vec::with_capacity(count * 3),
            indices: Vec::with_capacity(count),
            natoms: Vec::with_capacity(count),
            senders: Vec::with_capacity(edges * 2),
            receivers: Vec::with_capacity(edges),
            shifts: Vec::with_capacity(edges * 3),
            unit_shifts: Vec::with_capacity(edges * 3),
            assignment: Vec::with_capacity(nodes),
            node_slices: vec![0],
            edge_slices: vec![0],
        }
    }
    fn append(&mut self, graph: PreparedGraph) {
        let n = graph.num_nodes();
        let e = graph.num_edges();
        let offset = *self.node_slices.last().unwrap();
        self.node_attrs
            .resize((offset as usize + n) * self.species_count, T::from_f64(0.0));
        if self.indices.is_empty() {
            self.single_energy = graph.entry.energy;
            self.single_volume = determinant(&graph.entry.cell).abs();
        }
        self.positions
            .extend(graph.positions.iter().map(|v| T::from_f64(*v)));
        self.forces
            .extend(graph.forces.iter().map(|v| T::from_f64(*v)));
        self.atomic_numbers.extend_from_slice(&graph.atomic_numbers);
        for (atom, species) in graph.species.iter().enumerate() {
            self.node_attrs[(offset as usize + atom) * self.species_count + species] =
                T::from_f64(1.0);
        }
        self.cell
            .extend(graph.neighborhood.cell.iter().map(|v| T::from_f64(*v)));
        self.cell_volume.push(determinant(&graph.entry.cell).abs());
        let signed_volume = determinant(&graph.entry.cell);
        self.signed_volume.push(T::from_f64(signed_volume));
        let reciprocal = if signed_volume.abs() <= 1e-10 {
            [0.0; 9]
        } else {
            let inv = inverse(&graph.entry.cell).unwrap();
            std::array::from_fn(|i| 2.0 * std::f64::consts::PI * inv[3 * (i % 3) + i / 3])
        };
        self.rcell.extend(reciprocal.map(T::from_f64));
        self.total_charge
            .push(T::from_f64(graph.entry.total_charge));
        self.total_spin.push(T::from_f64(graph.entry.total_spin));
        self.external_field
            .extend(graph.entry.external_field.map(T::from_f64));
        self.source_id.push(graph.entry.source_id as i64);
        self.reaction_id.push(graph.entry.reaction_id);
        self.state_id.push(graph.entry.state_id as i64);
        self.energy.push(T::from_f64(graph.entry.energy));
        self.stress
            .extend(voigt_to_matrix(graph.entry.stress).map(T::from_f64));
        self.pbc.extend_from_slice(&graph.entry.pbc);
        self.indices.push(graph.index as i64);
        self.natoms.push(n as i64);
        self.senders.extend(
            graph.neighborhood.edge_index[..e]
                .iter()
                .map(|v| v + offset),
        );
        self.receivers.extend(
            graph.neighborhood.edge_index[e..]
                .iter()
                .map(|v| v + offset),
        );
        self.shifts
            .extend(graph.neighborhood.shifts.iter().map(|v| T::from_f64(*v)));
        self.unit_shifts.extend(
            graph
                .neighborhood
                .unit_shifts
                .iter()
                .map(|v| T::from_f64(*v)),
        );
        self.assignment
            .extend(std::iter::repeat_n(self.indices.len() as i64 - 1, n));
        self.node_slices.push(offset + n as i64);
        self.edge_slices
            .push(self.edge_slices.last().unwrap() + e as i64);
    }
}

fn materialize_graphs_typed<T: BatchFloat>(
    py: Python<'_>,
    graphs: Vec<PreparedGraph>,
    batch: bool,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let count = graphs.len();
    let nodes = graphs.iter().map(PreparedGraph::num_nodes).sum();
    let edges = graphs.iter().map(PreparedGraph::num_edges).sum();
    let mut buffers = BatchBuffers::<T>::new(graphs[0].species_count, count, nodes, edges);
    for graph in graphs {
        buffers.append(graph);
    }
    finish_graph_buffers(py, buffers, batch, dtype)
}

fn finish_graph_buffers<T: BatchFloat>(
    py: Python<'_>,
    buffers: BatchBuffers<T>,
    batch: bool,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let count = buffers.indices.len();
    let nodes = buffers.atomic_numbers.len();
    let edges = buffers.receivers.len();
    let BatchBuffers {
        species_count,
        single_energy,
        single_volume,
        positions,
        forces,
        atomic_numbers,
        node_attrs,
        cell,
        cell_volume,
        signed_volume,
        rcell,
        total_charge,
        total_spin,
        external_field,
        source_id,
        reaction_id,
        state_id,
        energy,
        stress,
        pbc,
        indices,
        natoms,
        mut senders,
        receivers,
        shifts,
        unit_shifts,
        assignment,
        node_slices,
        edge_slices,
    } = buffers;
    senders.extend(receivers);
    let kwargs = PyDict::new_bound(py);
    let pos = batch_float_tensor(py, positions, &[nodes, 3], dtype)?;
    kwargs.set_item("pos", pos.clone_ref(py))?;
    kwargs.set_item("positions", pos)?;
    kwargs.set_item("force", batch_float_tensor(py, forces, &[nodes, 3], dtype)?)?;
    kwargs.set_item(
        "node_attrs",
        batch_float_tensor(py, node_attrs, &[nodes, species_count], dtype)?,
    )?;
    kwargs.set_item("atomic_numbers", i64_tensor(py, atomic_numbers, &[nodes])?)?;
    kwargs.set_item(
        "tags",
        batch_float_tensor(py, vec![T::from_f64(0.0); nodes], &[nodes], dtype)?,
    )?;
    kwargs.set_item(
        "cell",
        batch_float_tensor(py, cell, &[count * 3, 3], dtype)?,
    )?;
    kwargs.set_item("edge_index", i64_tensor(py, senders, &[2, edges])?)?;
    kwargs.set_item(
        "shifts",
        batch_float_tensor(py, shifts, &[edges, 3], dtype)?,
    )?;
    kwargs.set_item(
        "unit_shifts",
        batch_float_tensor(py, unit_shifts, &[edges, 3], dtype)?,
    )?;
    kwargs.set_item(
        "stress",
        batch_float_tensor(py, stress, &[count, 3, 3], dtype)?,
    )?;
    kwargs.set_item("pbc", bool_tensor(py, pbc, &[count * 3])?)?;
    let scalar_shape: &[usize] = if batch { &[count] } else { &[] };
    for (key, values) in [
        ("volume", signed_volume),
        ("total_charge", total_charge),
        ("total_spin", total_spin),
        ("fermi_level", vec![T::from_f64(0.0); count]),
    ] {
        kwargs.set_item(key, batch_float_tensor(py, values, scalar_shape, dtype)?)?;
    }
    for (key, values) in [
        ("source_id", source_id),
        ("reaction_id", reaction_id),
        ("state_id", state_id),
    ] {
        kwargs.set_item(key, i64_tensor(py, values, scalar_shape)?)?;
    }
    kwargs.set_item(
        "external_field",
        batch_float_tensor(py, external_field, &[count, 3], dtype)?,
    )?;
    kwargs.set_item(
        "rcell",
        batch_float_tensor(py, rcell, &[count * 3, 3], dtype)?,
    )?;
    let module = py.import_bound("torch_geometric.data")?;
    if !batch {
        // Equitrain forwards ASE's NumPy float64 value for an individual
        // graph. Retain that type as well as the batched torch.float64
        // contract below, so native datasets and loaders are interchangeable
        // with their portable counterparts.
        let numpy = py.import_bound("numpy")?;
        kwargs.set_item(
            "cell_volume",
            numpy.getattr("float64")?.call1((single_volume,))?,
        )?;
        kwargs.set_item("y", scalar_float_tensor(py, single_energy, dtype)?)?;
        kwargs.set_item("natoms", nodes)?;
        kwargs.set_item("idx", indices[0])?;
        return Ok(module.getattr("Data")?.call((), Some(&kwargs))?.into_py(py));
    }
    // This intentionally does not use `dtype`: the portable Equitrain loader
    // produces cell_volume from NumPy/ASE as float64 regardless of
    // torch.get_default_dtype().
    let cell_volume_dtype = py.import_bound("torch")?.getattr("float64")?;
    kwargs.set_item(
        "cell_volume",
        batch_float_tensor(py, cell_volume, &[count], &cell_volume_dtype)?,
    )?;
    kwargs.set_item("y", batch_float_tensor(py, energy, &[count], dtype)?)?;
    kwargs.set_item("natoms", i64_tensor(py, natoms, &[count])?)?;
    kwargs.set_item("idx", i64_tensor(py, indices, &[count])?)?;
    kwargs.set_item("batch", i64_tensor(py, assignment, &[nodes])?)?;
    let ptr = i64_tensor(py, node_slices.clone(), &[count + 1])?;
    kwargs.set_item("ptr", ptr.clone_ref(py))?;
    let slices = PyDict::new_bound(py);
    let incs = PyDict::new_bound(py);
    let scalar_slices = i64_tensor(py, (0..=count as i64).collect::<Vec<_>>(), &[count + 1])?;
    let triple_slices = i64_tensor(
        py,
        (0..=count as i64).map(|i| 3 * i).collect::<Vec<_>>(),
        &[count + 1],
    )?;
    let edge_ptr = i64_tensor(py, edge_slices, &[count + 1])?;
    let zero_incs = i64_tensor(py, vec![0; count], &[count])?;
    for key in [
        "pos",
        "positions",
        "force",
        "node_attrs",
        "atomic_numbers",
        "tags",
    ] {
        slices.set_item(key, ptr.clone_ref(py))?;
        incs.set_item(key, zero_incs.clone_ref(py))?;
    }
    for key in ["shifts", "unit_shifts", "edge_index"] {
        slices.set_item(key, edge_ptr.clone_ref(py))?;
        incs.set_item(key, zero_incs.clone_ref(py))?;
    }
    for key in ["cell", "pbc", "rcell"] {
        slices.set_item(key, triple_slices.clone_ref(py))?;
        incs.set_item(key, zero_incs.clone_ref(py))?;
    }
    for key in [
        "cell_volume",
        "y",
        "natoms",
        "idx",
        "stress",
        "volume",
        "total_charge",
        "total_spin",
        "fermi_level",
        "external_field",
        "source_id",
        "reaction_id",
        "state_id",
    ] {
        slices.set_item(key, scalar_slices.clone_ref(py))?;
        incs.set_item(key, zero_incs.clone_ref(py))?;
    }
    incs.set_item(
        "edge_index",
        i64_tensor(py, node_slices[..count].to_vec(), &[count])?,
    )?;
    let batch_cls = DATA_BATCH_CLASS
        .get_or_try_init(py, || -> PyResult<PyObject> {
            Ok(module.getattr("Batch")?.call0()?.get_type().into_py(py))
        })?
        .bind(py);
    // Reuse PyG's resolved DataBatch class without repeating dynamic class
    // selection and inspect.signature() for every batch.
    let out = batch_cls.call_method1("__new__", (batch_cls,))?;
    out.call_method("__init__", (), Some(&kwargs))?;
    out.setattr("_num_graphs", count)?;
    out.setattr("_slice_dict", slices)?;
    out.setattr("_inc_dict", incs)?;
    Ok(out.into_py(py))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized_edges(neighborhood: &Neighborhood) -> Vec<(i64, i64, i64, i64, i64)> {
        let edge_count = neighborhood.edge_index.len() / 2;
        let mut edges = (0..edge_count)
            .map(|edge| {
                (
                    neighborhood.edge_index[edge],
                    neighborhood.edge_index[edge_count + edge],
                    neighborhood.unit_shifts[3 * edge] as i64,
                    neighborhood.unit_shifts[3 * edge + 1] as i64,
                    neighborhood.unit_shifts[3 * edge + 2] as i64,
                )
            })
            .collect::<Vec<_>>();
        edges.sort_unstable();
        edges
    }

    fn assert_same_neighborhood(left: &Neighborhood, right: &Neighborhood) {
        assert_eq!(left.cell, right.cell);
        assert_eq!(normalized_edges(left), normalized_edges(right));

        for neighborhood in [left, right] {
            let (unit_shifts, _) = neighborhood.unit_shifts.as_chunks::<3>();
            let (shifts, _) = neighborhood.shifts.as_chunks::<3>();
            for (unit, shift) in unit_shifts.iter().zip(shifts) {
                let expected = lattice_shift(
                    [unit[0] as i64, unit[1] as i64, unit[2] as i64],
                    &neighborhood.cell,
                );
                for axis in 0..3 {
                    assert!((expected[axis] - shift[axis]).abs() <= 1.0e-12);
                }
            }
        }
    }

    fn assert_cell_list_matches_brute_force(prepared: &PreparedNeighborhood) {
        let cell_list = get_neighborhood_cell_list(prepared).unwrap();
        let translation_queries = get_neighborhood_translation_queries(prepared).unwrap();
        let brute_force = get_neighborhood_brute_force(prepared).unwrap();
        assert_same_neighborhood(&cell_list, &brute_force);
        assert_same_neighborhood(&translation_queries, &brute_force);
    }

    fn next_random(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
    }

    fn fractional_to_cartesian(fractional: [f64; 3], cell: &[f64; 9]) -> [f64; 3] {
        [
            fractional[0] * cell[0] + fractional[1] * cell[3] + fractional[2] * cell[6],
            fractional[0] * cell[1] + fractional[1] * cell[4] + fractional[2] * cell[7],
            fractional[0] * cell[2] + fractional[1] * cell[5] + fractional[2] * cell[8],
        ]
    }

    #[test]
    fn cell_list_matches_brute_force_without_periodic_boundaries() {
        let mut state = 0x5eed_u64;
        let mut positions = Vec::new();
        for _ in 0..128 {
            positions.extend([
                12.0 * next_random(&mut state) - 6.0,
                12.0 * next_random(&mut state) - 6.0,
                12.0 * next_random(&mut state) - 6.0,
            ]);
        }

        let prepared = prepare_neighborhood(&positions, 1.25, [false; 3], [0.0; 9]).unwrap();
        assert_cell_list_matches_brute_force(&prepared);
    }

    #[test]
    fn cell_list_matches_brute_force_for_triclinic_periodic_cell() {
        let cell = [4.0, 0.0, 0.0, 0.7, 3.8, 0.0, 0.2, 0.4, 4.2];
        let mut state = 0xc0ffee_u64;
        let mut positions = Vec::new();
        for _ in 0..48 {
            // Include atoms several cells outside the primary periodic box.
            let fractional = [
                4.0 * next_random(&mut state) - 1.5,
                4.0 * next_random(&mut state) - 1.5,
                4.0 * next_random(&mut state) - 1.5,
            ];
            positions.extend(fractional_to_cartesian(fractional, &cell));
        }

        let prepared = prepare_neighborhood(&positions, 1.3, [true; 3], cell).unwrap();
        assert_eq!(prepared.translations.len(), 27);
        assert_cell_list_matches_brute_force(&prepared);
    }

    #[test]
    fn periodic_shifts_are_relative_to_original_out_of_cell_positions() {
        let positions = [0.1, 0.0, 0.0, 3.9, 0.0, 0.0];
        let cell = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let neighborhood = get_neighborhood(&positions, 0.3, [true, false, false], cell).unwrap();

        assert_eq!(
            normalized_edges(&neighborhood),
            [(0, 1, -2, 0, 0), (1, 0, 2, 0, 0)]
        );
    }

    #[test]
    fn periodic_self_images_within_cutoff_are_retained() {
        let positions = [0.0, 0.0, 0.0];
        let cell = [1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0];
        let neighborhood = get_neighborhood(&positions, 2.1, [true, false, false], cell).unwrap();

        assert_eq!(
            normalized_edges(&neighborhood),
            [
                (0, 0, -2, 0, 0),
                (0, 0, -1, 0, 0),
                (0, 0, 1, 0, 0),
                (0, 0, 2, 0, 0),
            ]
        );
    }

    #[test]
    fn adaptive_search_is_deterministic_and_matches_reference_edges() {
        let mut positions = Vec::new();
        for x in 0..12 {
            for y in 0..12 {
                positions.extend([x as f64 * 0.8, y as f64 * 0.8, 0.0]);
            }
        }
        let prepared = prepare_neighborhood(&positions, 0.9, [false; 3], [0.0; 9]).unwrap();
        let adaptive = get_neighborhood(&positions, 0.9, [false; 3], [0.0; 9]).unwrap();
        let adaptive_again = get_neighborhood(&positions, 0.9, [false; 3], [0.0; 9]).unwrap();
        let reference = get_neighborhood_brute_force(&prepared).unwrap();
        assert_same_neighborhood(&adaptive, &reference);
        assert_eq!(adaptive.edge_index, adaptive_again.edge_index);
        assert_eq!(adaptive.unit_shifts, adaptive_again.unit_shifts);
        assert_eq!(adaptive.shifts, adaptive_again.shifts);
    }
}
