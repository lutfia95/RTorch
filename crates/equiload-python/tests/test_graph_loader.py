"""Public EquiLoad API tests."""
from pathlib import Path
import unittest


class EquiLoadGraphLoaderTest(unittest.TestCase):
    def test_public_api_is_graph_loader_focused(self):
        import h5py
        import torch
        from torch_geometric.data import Batch

        import equiload

        self.assertTrue(hasattr(equiload, "HDF5GraphLoader"))
        for legacy_name in ("Tensor", "Sample", "Batch", "TensorDataset", "DataLoader"):
            self.assertFalse(hasattr(equiload, legacy_name), legacy_name)

        fixture = Path(__file__).resolve().parents[3] / "test_data" / "train.h5"
        with h5py.File(fixture, "r") as handle:
            atomic_numbers = sorted({int(value) for value in handle["atomic_numbers"][:]})
        loader = equiload.HDF5GraphLoader.from_hdf5_files(
            [str(fixture)], r_max=5.0, atomic_numbers=atomic_numbers, batch_size=2,
        )
        data_list = next(iter(loader))
        batch = data_list[0]
        self.assertTrue(loader._uses_equiload)
        self.assertIsInstance(data_list, list)
        self.assertIsInstance(batch, Batch)
        self.assertEqual(batch.num_graphs, 2)
        self.assertIsInstance(batch.pos, torch.Tensor)
        loader.dataset.close()

    def test_iteration_is_repeatable_and_honours_drop_last(self):
        import h5py
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
                epochs = [[i for batches in loader for batch in batches for i in batch.idx.tolist()]
                          for _ in range(2)]
                self.assertEqual(epochs[0], epochs[1])
                expected = list(range(length)) * 2
                count = len(expected) // 5 * 5 if drop_last else len(expected)
                self.assertEqual(len(epochs[0]), count)
                if not shuffle:
                    self.assertEqual(epochs[0], expected[:count])
                elif not drop_last:
                    self.assertCountEqual(epochs[0], expected)
