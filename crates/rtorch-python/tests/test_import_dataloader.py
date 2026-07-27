from pathlib import Path
import unittest


class RtorchDataLoaderImportTest(unittest.TestCase):
    def test_imports_dataloader_and_reads_hdf5_batch(self):
        import torch

        import rtorch

        self.assertTrue(hasattr(rtorch, "DataLoader"))

        repo_root = Path(__file__).resolve().parents[3]
        fixture = repo_root / "test_data" / "train.h5"

        loader = rtorch.DataLoader.from_hdf5_file(
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

        import rtorch

        self.assertTrue(hasattr(rtorch, "HDF5GraphLoader"))

        repo_root = Path(__file__).resolve().parents[3]
        fixture = repo_root / "test_data" / "train.h5"

        with h5py.File(fixture, "r") as handle:
            atomic_numbers = sorted(
                {int(value) for value in handle["atomic_numbers"][:]}
            )

        loader = rtorch.HDF5GraphLoader.from_hdf5_files(
            [str(fixture)],
            r_max=5.0,
            atomic_numbers=atomic_numbers,
            batch_size=2,
        )
        self.assertTrue(loader._uses_rtorch)
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


if __name__ == "__main__":
    unittest.main()
