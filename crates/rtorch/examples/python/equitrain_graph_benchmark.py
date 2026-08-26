"""Compare Equitrain and RTorch HDF5 graph construction throughput."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import h5py
import numpy as np

# Load the active PyTorch/LibTorch runtime before importing the RTorch extension.
import torch  # noqa: F401

import rtorch
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.format_hdf5 import HDF5GraphDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Equitrain HDF5 dataset")
    parser.add_argument("--r-max", type=float, required=True)
    parser.add_argument("--max-structures", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare every selected edge and periodic shift before timing.",
    )
    return parser.parse_args()


def atomic_numbers(path: Path) -> list[int]:
    with h5py.File(path, "r") as handle:
        dataset = handle["atomic_numbers"]
        numbers: set[int] = set()
        chunk_size = 1_000_000
        for start in range(0, len(dataset), chunk_size):
            numbers.update(
                int(value) for value in dataset[start : start + chunk_size]
            )
        return sorted(numbers)


def normalized_edges(graph) -> np.ndarray:
    edge_index = graph.edge_index.detach().cpu().numpy().T.astype(np.int64)
    unit_shifts = np.rint(
        graph.unit_shifts.detach().cpu().numpy()
    ).astype(np.int64)
    edges = np.concatenate((edge_index, unit_shifts), axis=1)
    if edges.shape[0] == 0:
        return edges
    order = np.lexsort(tuple(edges[:, column] for column in reversed(range(5))))
    return edges[order]


def verify_graphs(python_dataset, rust_dataset, indices: range) -> None:
    for index in indices:
        python_graph = python_dataset[index]
        rust_graph = rust_dataset[index]
        np.testing.assert_array_equal(
            normalized_edges(rust_graph),
            normalized_edges(python_graph),
            err_msg=f"neighbor mismatch at structure {index}",
        )
        np.testing.assert_allclose(rust_graph.pos, python_graph.pos)
        np.testing.assert_allclose(rust_graph.force, python_graph.force)
        np.testing.assert_allclose(rust_graph.y, python_graph.y)


def timed_pass(dataset, indices: range) -> tuple[float, int]:
    gc.collect()
    edge_count = 0
    start = time.perf_counter()
    for index in indices:
        graph = dataset[index]
        edge_count += int(graph.edge_index.shape[1])
    return time.perf_counter() - start, edge_count


def main() -> None:
    args = parse_args()
    numbers = atomic_numbers(args.path)
    python_dataset = HDF5GraphDataset(
        args.path,
        r_max=args.r_max,
        atomic_numbers=AtomicNumberTable(numbers),
    )
    rust_dataset = rtorch.HDF5GraphDataset.from_hdf5_file(
        args.path,
        r_max=args.r_max,
        atomic_numbers=numbers,
    )

    count = min(len(python_dataset), len(rust_dataset), args.max_structures)
    if count <= 0:
        raise ValueError("The selected dataset is empty.")
    indices = range(count)
    warmup_indices = range(min(count, args.warmup))

    for index in warmup_indices:
        python_dataset[index]
        rust_dataset[index]

    if args.verify:
        verify_graphs(python_dataset, rust_dataset, indices)
        print(f"Verified {count} graph pairs.")

    timings = {"Equitrain/Matscipy": [], "RTorch": []}
    edge_counts: dict[str, int] = {}
    for repeat in range(args.repeats):
        order = list(timings) if repeat % 2 == 0 else list(reversed(timings))
        datasets = {
            "Equitrain/Matscipy": python_dataset,
            "RTorch": rust_dataset,
        }
        for name in order:
            elapsed, edge_count = timed_pass(datasets[name], indices)
            timings[name].append(elapsed)
            edge_counts[name] = edge_count

    if len(set(edge_counts.values())) != 1:
        raise AssertionError(f"edge-count mismatch: {edge_counts}")

    medians = {name: statistics.median(values) for name, values in timings.items()}
    print(f"Structures: {count}; edges/pass: {next(iter(edge_counts.values()))}")
    for name, elapsed in medians.items():
        print(f"{name:20s} {elapsed:9.4f} s  {count / elapsed:10.2f} structures/s")
    print(
        "RTorch speedup: "
        f"{medians['Equitrain/Matscipy'] / medians['RTorch']:.3f}x"
    )

    python_dataset.close()
    rust_dataset.close()


if __name__ == "__main__":
    main()
