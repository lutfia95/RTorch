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

#[derive(Clone, Copy)]
struct EdgeRecord {
    sender: usize,
    receiver: usize,
    unit_shift: [i64; 3],
    shift: [f64; 3],
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
    let atom_positions = positions
        .chunks_exact(3)
        .map(|position| [position[0], position[1], position[2]])
        .collect::<Vec<_>>();
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
    let mut unit_shift = [0_i64; 3];
    for axis in 0..3 {
        unit_shift[axis] = translation.unit[axis]
            .checked_add(prepared.wrap_offsets[sender][axis])
            .and_then(|value| value.checked_sub(prepared.wrap_offsets[receiver][axis]))
            .ok_or_else(|| PyValueError::new_err("periodic image offset overflowed."))?;
    }
    if sender == receiver && unit_shift == [0, 0, 0] {
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
    if dist2 > prepared.cutoff2 {
        return Ok(None);
    }

    let shift = lattice_shift(unit_shift, &prepared.cell);
    if shift.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "periodic Cartesian shift contains non-finite values.",
        ));
    }

    Ok(Some(EdgeRecord {
        sender,
        receiver,
        unit_shift,
        shift,
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
    Ok(records_to_neighborhood(records, prepared.cell))
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
        .checked_mul(prepared.translations.len())
        .unwrap_or(usize::MAX);
    if expanded_atoms > MAX_EXPANDED_IMAGE_ATOMS {
        return get_neighborhood_translation_queries(prepared);
    }

    let natoms = prepared.positions.len();
    let inverse_bin_size = prepared.cutoff.recip();
    let mut bins: HashMap<BinKey, Vec<ImageAtom>> = HashMap::new();

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

    Ok(records_to_neighborhood(records, prepared.cell))
}

fn get_neighborhood_translation_queries(prepared: &PreparedNeighborhood) -> PyResult<Neighborhood> {
    let natoms = prepared.positions.len();
    let inverse_bin_size = prepared.cutoff.recip();
    let mut bins: HashMap<BinKey, Vec<usize>> = HashMap::new();
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

    Ok(records_to_neighborhood(records, prepared.cell))
}

fn records_to_neighborhood(records: Vec<EdgeRecord>, cell: [f64; 9]) -> Neighborhood {
    // Every search iterates senders in ascending order, matching Matscipy's
    // ordering guarantee. It deliberately does not sort receivers: sorting is
    // not part of that contract and adds O(E log E) work to every structure.
    let mut edge_index = Vec::with_capacity(2 * records.len());
    edge_index.extend(records.iter().map(|record| record.sender as i64));
    edge_index.extend(records.iter().map(|record| record.receiver as i64));

    let mut shifts = Vec::with_capacity(3 * records.len());
    let mut unit_shifts = Vec::with_capacity(3 * records.len());
    for record in records {
        shifts.extend_from_slice(&record.shift);
        unit_shifts.extend(record.unit_shift.map(|value| value as f64));
    }

    Neighborhood {
        edge_index,
        shifts,
        unit_shifts,
        cell,
    }
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

        let r_max = self.r_max;
        let pbc = entry.pbc;
        let input_cell = entry.cell;
        let neighborhood =
            py.allow_threads(move || get_neighborhood(&positions_values, r_max, pbc, input_cell))?;
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
            for (unit, shift) in neighborhood
                .unit_shifts
                .chunks_exact(3)
                .zip(neighborhood.shifts.chunks_exact(3))
            {
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
