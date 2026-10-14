"""Native graph batches must remain interchangeable with PyG collation."""
from pathlib import Path
import tempfile
import unittest
from itertools import product
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch
import equiload


class NativeGraphBatchTest(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(__file__).resolve().parents[3] / 'test_data/train.h5'
        with h5py.File(self.fixture) as f:
            self.numbers = sorted(set(f['atomic_numbers'][:].tolist()))

    def assert_batch_equal(self, actual, expected):
        self.assertIsInstance(actual, Batch)
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        self.assertEqual(actual.num_graphs, expected.num_graphs)
        for key in expected.keys():
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
        for attr in ('_slice_dict', '_inc_dict'):
            self.assertEqual(set(getattr(actual, attr)), set(getattr(expected, attr)))
            for key in getattr(expected, attr):
                torch.testing.assert_close(getattr(actual, attr)[key], getattr(expected, attr)[key])
        for a, b in zip(actual.to_data_list(), expected.to_data_list()):
            self.assertEqual(set(a.keys()), set(b.keys()))
            for key in b.keys():
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)

    def test_native_batch_matches_pyg_in_both_dtypes(self):
        old_dtype = torch.get_default_dtype()
        try:
            for dtype in (torch.float32, torch.float64):
                torch.set_default_dtype(dtype)
                dataset = equiload.HDF5GraphDataset.from_hdf5_file(str(self.fixture), 5., self.numbers)
                for size in (1, 2, 5, len(dataset) + 1):
                    loader = equiload.HDF5GraphLoader.from_hdf5_file(str(self.fixture), 5., self.numbers, batch_size=size)
                    offset = 0
                    for outer in loader:
                        expected = Batch.from_data_list([dataset[i] for i in range(offset, min(offset + size, len(dataset)))])
                        self.assertEqual(len(outer), 1)
                        self.assert_batch_equal(outer[0], expected)
                        self.assertEqual(outer[0].pos.data_ptr(), outer[0].positions.data_ptr())
                        offset += size
        finally:
            torch.set_default_dtype(old_dtype)

    def test_dynamic_splitting_dropping_and_retained_batches(self):
        dataset = equiload.HDF5GraphDataset.from_hdf5_file(str(self.fixture), 5., self.numbers)
        graphs = [dataset[i] for i in range(len(dataset))]
        for limits in ({'max_nodes': 24}, {'max_edges': 350}, {'max_nodes': 24, 'max_edges': 350},
                       {'max_nodes': 1, 'drop': True}, {'max_edges': 0, 'drop': True}):
            loader = equiload.HDF5GraphLoader.from_hdf5_file(str(self.fixture), 5., self.numbers,
                                                        batch_size=len(dataset), **limits)
            actual = next(iter(loader))
            expected = []
            current = []
            n = e = 0
            for graph in graphs:
                if limits.get('drop') and (graph.num_nodes > limits.get('max_nodes', float('inf')) or
                                           graph.num_edges > limits.get('max_edges', float('inf'))):
                    continue
                if current and (n + graph.num_nodes > limits.get('max_nodes', float('inf')) or
                                e + graph.num_edges > limits.get('max_edges', float('inf'))):
                    expected.append(Batch.from_data_list(current))
                    current = []; n = e = 0
                current.append(graph); n += graph.num_nodes; e += graph.num_edges
            if current:
                expected.append(Batch.from_data_list(current))
            list(loader)  # Old tensors and metadata must survive subsequent epochs.
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_batch_equal(a, b)

    def test_read_ahead_cache_views_survive_replacement(self):
        count = 10000
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'read_ahead.h5')
            values = np.arange(count * 3, dtype=np.float64).reshape(count, 3)
            entries = np.array([(i, 1) for i in range(count)], dtype=[('offset', 'i8'), ('length', 'i4')])
            with h5py.File(path, 'w') as f:
                f.create_dataset('positions', data=values, chunks=(1024, 3))
                f.create_dataset('structures', data=entries)
            dataset = equiload.TensorDataset.from_hdf5_file(path, 'positions')
            indices = [0, 1, 4096, 4097, 8193, 8194, 9999, 2, 5000, 5001]
            retained = [dataset[i].to_torch() for i in indices]
            for i, tensor in zip(indices, retained):
                torch.testing.assert_close(tensor, torch.from_numpy(values[i:i + 1]), rtol=0, atol=0)
            retained[-1].fill_(-1)
            torch.testing.assert_close(dataset[indices[-1]].to_torch(),
                                       torch.from_numpy(values[indices[-1]:indices[-1] + 1]))

    def test_native_path_does_not_call_python_collation(self):
        loader = equiload.HDF5GraphLoader.from_hdf5_file(str(self.fixture), 5., self.numbers, batch_size=2)
        with patch.object(Batch, 'from_data_list', side_effect=AssertionError('Python collation called')):
            self.assertEqual(next(iter(loader))[0].num_graphs, 2)

    def test_close_releases_handles_and_retained_batches_stay_valid(self):
        loader = equiload.HDF5GraphLoader.from_hdf5_file(str(self.fixture), 5., self.numbers, batch_size=2)
        batch = next(iter(loader))[0]
        saved = batch.pos.clone()
        loader.dataset.close()
        loader.dataset.close()
        self.assertGreater(len(loader.dataset), 0)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            loader.dataset[0]
        torch.testing.assert_close(batch.pos, saved)
        self.assertEqual(len(batch.to_data_list()), 2)
        batch.positions.requires_grad_(True)
        (batch.positions.square().sum()).backward()
        torch.testing.assert_close(batch.positions.grad, 2 * saved)

    def test_periodic_neighbors_against_independent_pair_search(self):
        with h5py.File(self.fixture) as f:
            template = f['structures'][:1]
        cases = [
            (np.array([[3.9, .1, .2], [.1, 1.2, -.3]]),
             np.array([[2., 0., 0.], [.3, 2.2, 0.], [.2, -.1, 2.5]]), pbc, 1.4)
            for pbc in [(True, True, True), (True, False, True), (False, False, False)]
        ]
        cases += [(np.zeros((1, 3)), np.eye(3) * 2, (True,) * 3, 2.1),
                  (np.array([[0., 0., 0.], [1., 0., 0.], [1. + 1e-10, 0., 0.]]),
                   np.eye(3) * 4, (False,) * 3, 1.)]
        cell = np.array([[2., 0., 0.], [.3, 2.2, 0.], [.2, -.1, 2.5]])
        cases.append((np.random.default_rng(42).random((128, 3)) @ cell,
                      cell, (True,) * 3, 1.4))  # Exercises the spatial-bin path.
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'periodic.h5')
            for positions, cell, pbc, cutoff in cases:
                entry = template.copy()
                entry['offset'] = 0; entry['length'] = len(positions)
                entry['cell'] = cell; entry['pbc'] = pbc
                with h5py.File(path, 'w') as f:
                    f.create_dataset('structures', data=entry)
                    f.create_dataset('positions', data=positions)
                    f.create_dataset('forces', data=np.zeros_like(positions))
                    f.create_dataset('atomic_numbers', data=np.ones(len(positions), dtype=np.int32))
                dataset = equiload.HDF5GraphDataset.from_hdf5_file(path, cutoff, [1])
                graph = dataset[0]
                actual = sorted((int(a), int(b), *map(int, unit)) for (a, b), unit in
                                zip(graph.edge_index.T.tolist(), graph.unit_shifts.tolist()))
                expected = []
                for unit in product(*(range(-3, 4) if periodic else [0] for periodic in pbc)):
                    displacement = positions[None, :, :] - positions[:, None, :] + np.array(unit) @ cell
                    accepted = np.sum(displacement * displacement, axis=-1) < cutoff * cutoff
                    if unit == (0, 0, 0):
                        np.fill_diagonal(accepted, False)
                    a, b = np.nonzero(accepted)
                    expected.extend((int(sender), int(receiver), *unit) for sender, receiver in zip(a, b))
                self.assertEqual(actual, sorted(expected))
                del dataset

    def test_strict_cutoff_matches_python_neighbor_list(self):
        try:
            from matscipy.neighbours import neighbour_list
        except ImportError:
            self.skipTest('Optional Python neighbor-list reference is unavailable')
        with h5py.File(self.fixture) as f:
            template = f['structures'][:1]
        cases = [
            (np.array([[0., 0., 0.], [5. - 1e-12, 0., 0.],
                       [5., 0., 0.], [5. + 1e-12, 0., 0.]]), np.eye(3) * 20, (False,) * 3),
            (np.zeros((1, 3)), np.diag([5., 8., 8.]), (True,) * 3),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'cutoff.h5')
            for positions, cell, pbc in cases:
                entry = template.copy()
                entry['offset'] = 0; entry['length'] = len(positions)
                entry['cell'] = cell; entry['pbc'] = pbc
                with h5py.File(path, 'w') as f:
                    f.create_dataset('structures', data=entry)
                    f.create_dataset('positions', data=positions)
                    f.create_dataset('forces', data=np.zeros_like(positions))
                    f.create_dataset('atomic_numbers', data=np.ones(len(positions), dtype=np.int32))
                dataset = equiload.HDF5GraphDataset.from_hdf5_file(path, 5., [1])
                graph = dataset[0]
                actual = sorted((int(a), int(b), *map(int, unit)) for (a, b), unit in
                                zip(graph.edge_index.T.tolist(), graph.unit_shifts.tolist()))
                a, b, units = neighbour_list('ijS', positions=positions, cell=cell, pbc=pbc, cutoff=5.)
                expected = sorted((int(i), int(j), *map(int, unit)) for i, j, unit in zip(a, b, units))
                self.assertEqual(actual, expected)
                del dataset

    def test_empty_files_in_dataset_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'empty.h5')
            with h5py.File(self.fixture) as source, h5py.File(path, 'w') as empty:
                for key in ('structures', 'positions', 'forces', 'atomic_numbers'):
                    empty.create_dataset(key, shape=(0, *source[key].shape[1:]), dtype=source[key].dtype)
                length = len(source['structures'])
            loader = equiload.HDF5GraphLoader.from_hdf5_files(
                [path, str(self.fixture), path, str(self.fixture), path],
                5., self.numbers, batch_size=5,
            )
            self.assertEqual([int(i) for outer in loader for batch in outer for i in batch.idx],
                             list(range(length)) * 2)

    def test_optional_metadata_defaults_and_values(self):
        fields = [('total_charge', 'f8'), ('total_spin', 'f8'), ('external_field', 'f8', (3,)),
                  ('source_id', 'i4'), ('reaction_id', 'i8'), ('state_id', 'i4')]
        values = dict(total_charge=3.5, total_spin=2., external_field=[1., 2., 3.],
                      source_id=9, reaction_id=14_000_000_000, state_id=4)
        defaults = dict(total_charge=0., total_spin=1., external_field=[0., 0., 0.],
                        source_id=0, reaction_id=-1, state_id=-1)
        with h5py.File(self.fixture) as source:
            original = source['structures'][:1]
            atoms = {key: source[key][:int(original['length'][0])]
                     for key in ('positions', 'forces', 'atomic_numbers')}
        # Exercise absent fields, a partially upgraded schema, and the full schema.
        base_fields = [d for d in original.dtype.descr if d[0] not in defaults]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'metadata.h5')
            for optional in ([], fields[1:2], fields):
                entries = np.zeros(2, dtype=base_fields + optional)
                for key in entries.dtype.names:
                    if key in original.dtype.names:
                        entries[key] = original[key]
                entries['offset'] = 0
                expected = defaults.copy()
                for field in optional:
                    key = field[0]
                    entries[key] = values[key]
                    expected[key] = values[key]
                with h5py.File(path, 'w') as f:
                    f.create_dataset('structures', data=entries)
                    for key, data in atoms.items():
                        f.create_dataset(key, data=data)
                dataset = equiload.HDF5GraphDataset.from_hdf5_file(path, 5., self.numbers)
                loader = equiload.HDF5GraphLoader.from_hdf5_file(path, 5., self.numbers, batch_size=2)
                graph = dataset[0]
                for key, value in expected.items():
                    np.testing.assert_array_equal(np.asarray(graph[key]).reshape(-1), np.asarray(value).reshape(-1))
                self.assert_batch_equal(next(iter(loader))[0], Batch.from_data_list([dataset[0], dataset[1]]))
                loader.dataset.close(); dataset.close()


if __name__ == '__main__':
    unittest.main()
