from pathlib import Path
import tempfile
import unittest


class EquiloadDataLoaderImportTest(unittest.TestCase):
    def test_imports_dataloader_and_reads_hdf5_batch(self):
        import torch

        import equiload

        self.assertTrue(hasattr(equiload, "DataLoader"))

        repo_root = Path(__file__).resolve().parents[3]
        fixture = repo_root / "test_data" / "train.h5"

        loader = equiload.DataLoader.from_hdf5_file(
            str(fixture),
            "positions",
            batch_size=2,
        )
        batch = next(iter(loader))
        tensor = batch.to_torch()

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.ndim, 3)
        self.assertEqual(int(tensor.shape[0]), 2)
        self.assertEqual(int(tensor.shape[-1]), 3)
        self.assertEqual(batch.labels, [0, 0])

    def test_imports_graph_loader_and_reads_hdf5_batch(self):
        import h5py
        import torch
        from torch_geometric.data import Batch

        import equiload

        self.assertTrue(hasattr(equiload, "HDF5GraphLoader"))

        repo_root = Path(__file__).resolve().parents[3]
        fixture = repo_root / "test_data" / "train.h5"

        with h5py.File(fixture, "r") as handle:
            atomic_numbers = sorted(
                {int(value) for value in handle["atomic_numbers"][:]}
            )

        loader = equiload.HDF5GraphLoader.from_hdf5_files(
            [str(fixture)],
            r_max=5.0,
            atomic_numbers=atomic_numbers,
            batch_size=2,
        )
        self.assertTrue(loader._uses_equiload)
        data_list = next(iter(loader))
        batch = data_list[0]

        self.assertIsInstance(data_list, list)
        self.assertEqual(len(data_list), 1)
        self.assertIsInstance(batch, Batch)
        self.assertEqual(batch.num_graphs, 2)
        self.assertIsInstance(batch.pos, torch.Tensor)
        self.assertEqual(int(batch.pos.shape[-1]), 3)
        self.assertEqual(batch.force.shape, batch.pos.shape)
        self.assertEqual(int(batch["node_attrs"].shape[-1]), len(atomic_numbers))
        self.assertEqual(batch.stress.shape, (2, 3, 3))
        self.assertEqual(batch.y.shape, (2,))
        self.assertEqual(batch.idx.tolist(), [0, 1])
        loader.dataset.close()

    def test_lazy_spans_and_metadata_across_cache_boundaries(self):
        import h5py
        import numpy as np
        import torch
        import equiload

        fixture = Path(__file__).resolve().parents[3] / "test_data" / "train.h5"
        with h5py.File(fixture) as source:
            template = source["structures"][0:1]
        count = 8205
        entries = np.repeat(template, count)
        entries["offset"] = np.arange(count)
        entries["length"] = 1
        entries["pbc"] = False
        entries["energy"] = np.arange(count)
        positions = np.arange(count * 3, dtype=np.float64).reshape(count, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "chunks.h5")
            with h5py.File(path, "w") as handle:
                # A large on-disk chunk must not force a full metadata cache.
                handle.create_dataset("structures", data=entries, chunks=(count,))
                handle.create_dataset("positions", data=positions)
                handle.create_dataset("forces", data=-positions)
                handle.create_dataset("atomic_numbers", data=np.ones(count, dtype=np.int32))
            tensors = equiload.TensorDataset.from_hdf5_file(path, "positions")
            graphs = equiload.HDF5GraphDataset.from_hdf5_file(path, 5.0, [1])
            self.assertEqual(len(tensors), count)
            self.assertEqual(len(graphs), count)
            for index in [0, 4095, 4096, 8191, 8192, count - 1, 1, 4100, 0]:
                with self.subTest(index=index):
                    actual = tensors[index].to_torch()
                    torch.testing.assert_close(actual, torch.from_numpy(positions[index:index+1]))
                    graph = graphs[index]
                    torch.testing.assert_close(graph.pos, actual.to(torch.get_default_dtype()))
                    torch.testing.assert_close(graph.force, -graph.pos)
                    self.assertEqual(graph.y.item(), index)
                    self.assertEqual(graph.num_nodes, 1)
                    self.assertEqual(graph.num_edges, 0)
            with self.assertRaises(RuntimeError):
                tensors[count]
            del tensors, graphs

    def test_empty_structures_and_unsliced_tensor(self):
        import h5py
        import numpy as np
        import torch
        import equiload

        with tempfile.TemporaryDirectory() as directory:
            for structured in (False, True):
                path = str(Path(directory) / f"empty_{structured}.h5")
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("positions", data=np.empty((0, 3)))
                    if structured:
                        handle.create_dataset("structures", shape=(0,), dtype=[("offset", "i8"), ("length", "i4")])
                dataset = equiload.TensorDataset.from_hdf5_file(path, "positions")
                self.assertEqual(len(dataset), 0 if structured else 1)
                if structured:
                    with self.assertRaises(RuntimeError):
                        dataset[0]
                else:
                    self.assertEqual(tuple(dataset[0].to_torch().shape), (0, 3))
                del dataset

    def test_invalid_span_is_reported_on_every_access(self):
        import h5py
        import numpy as np
        import torch
        import equiload

        with tempfile.TemporaryDirectory() as directory:
            for offset, message in [(-1, "negative offset"), (10, "exceeds dataset bounds")]:
                path = str(Path(directory) / f"invalid_{offset}.h5")
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("positions", data=np.zeros((1, 3)))
                    handle.create_dataset("structures", data=np.array(
                        [(offset, 1)], dtype=[("offset", "i8"), ("length", "i4")]
                    ))
                dataset = equiload.TensorDataset.from_hdf5_file(path, "positions")
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, message):
                        dataset[0]
                del dataset

    def test_graph_iteration_repeats_shuffle_and_drop_last(self):
        import h5py
        import torch
        import equiload

        fixture = Path(__file__).resolve().parents[3] / "test_data" / "train.h5"
        with h5py.File(fixture) as handle:
            numbers = sorted(set(handle["atomic_numbers"][:].tolist()))
            length = len(handle["structures"])
        for shuffle in (False, True):
            for drop_last in (False, True):
                loader = equiload.HDF5GraphLoader.from_hdf5_files(
                    [str(fixture), str(fixture)], 5.0, numbers,
                    batch_size=5, shuffle=shuffle, seed=42, drop_last=drop_last,
                )
                epochs = [
                    [i for batches in loader for batch in batches for i in batch.idx.tolist()]
                    for _ in range(2)
                ]
                self.assertEqual(epochs[0], epochs[1])
                expected = list(range(length)) * 2  # idx is local to each file
                count = len(expected) // 5 * 5 if drop_last else len(expected)
                self.assertEqual(len(epochs[0]), count)
                if not shuffle:
                    self.assertEqual(epochs[0], expected[:count])
                elif not drop_last:
                    self.assertCountEqual(epochs[0], expected)
                    self.assertNotEqual(epochs[0], expected)


if __name__ == "__main__":
    unittest.main()
