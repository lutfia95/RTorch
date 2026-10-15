#!/usr/bin/env python3
"""Check native graph values against the local Python loader across both files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'equitrain')]
import h5py
import numpy as np
import torch
import rtorch
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.format_hdf5 import HDF5GraphDataset


def edge_order(graph):
    keys = np.column_stack((graph.edge_index.numpy().T, graph.unit_shifts.numpy()))
    return keys, np.lexsort(tuple(keys[:, i] for i in reversed(range(5))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / '../../mptraj')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--json', type=Path, default=ROOT / 'python_comparison_results/parity.json')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    results = []
    for filename in ('train.h5', 'valid.h5'):
        path = args.dataset_dir / filename
        numbers = set()
        with h5py.File(path) as handle:
            atoms = handle['atomic_numbers']
            for start in range(0, len(atoms), 1_000_000):
                numbers.update(atoms[start:start + 1_000_000].tolist())
        numbers = sorted(numbers)
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            python = HDF5GraphDataset(path, 5., AtomicNumberTable(numbers))
            native = rtorch.HDF5GraphDataset.from_hdf5_file(str(path), 5., numbers)
            indices = np.unique(np.linspace(0, len(python) - 1, args.samples, dtype=int))
            tolerances = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-10)
            try:
                for index in indices:
                    a, b = python[int(index)], native[int(index)]
                    assert set(a.keys()) == set(b.keys()), (filename, index, set(a.keys()) ^ set(b.keys()))
                    ak, ai = edge_order(a)
                    bk, bi = edge_order(b)
                    np.testing.assert_array_equal(ak[ai], bk[bi], err_msg=f'{filename}: graph {index} edges')
                    np.testing.assert_allclose(a.shifts.numpy()[ai], b.shifts.numpy()[bi], **tolerances)
                    for key in a.keys():
                        if key not in ('edge_index', 'shifts', 'unit_shifts'):
                            np.testing.assert_allclose(a[key], b[key], **tolerances,
                                                       err_msg=f'{filename}: graph {index}, {key}')
            finally:
                python.close()
                native.close()
            results.append(dict(dataset=filename, dtype=str(dtype), graphs=len(indices)))
    report = dict(extension=rtorch.__file__, results=results,
                  total_graphs=sum(r['graphs'] for r in results),
                  checks='Matching field names and values, normalized edges, unit shifts and Cartesian shifts')
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
