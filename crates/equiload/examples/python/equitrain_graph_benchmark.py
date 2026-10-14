"""Compare Equitrain and EquiLoad HDF5 graph construction throughput."""

from __future__ import annotations

import argparse
import gc
import os
import resource
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# Load the active PyTorch/LibTorch runtime before importing the EquiLoad extension.
import torch  # noqa: F401

import equiload
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
        "--memory-sample-ms",
        type=float,
        default=10.0,
        help="Interval between process-RSS samples during a timed pass (default: 10).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare every selected edge and periodic shift before timing.",
    )
    return parser.parse_args()


def process_rss_bytes() -> int:
    """Return current process RSS on Linux, with a high-water fallback."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.readline().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        # ru_maxrss is bytes on macOS and KiB on other supported Unix systems.
        scale = 1 if sys.platform == "darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units[:-1]:
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} {units[-1]}"


class PeakRssSampler:
    """Sample full-process resident memory from a lightweight helper thread."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.start_rss_bytes = 0
        self.end_rss_bytes = 0
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def _sample(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, process_rss_bytes())

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> PeakRssSampler:
        self.start_rss_bytes = process_rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._sample()
        self._stop.set()
        self._thread.join()
        self.end_rss_bytes = process_rss_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.end_rss_bytes)


@dataclass(frozen=True)
class PassMetrics:
    elapsed_seconds: float
    edge_count: int
    start_rss_bytes: int
    end_rss_bytes: int
    peak_rss_bytes: int

    @property
    def peak_rss_growth_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)


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


def timed_pass(
    dataset, indices: range, memory_sample_seconds: float
) -> PassMetrics:
    gc.collect()
    edge_count = 0
    with PeakRssSampler(memory_sample_seconds) as memory:
        start = time.perf_counter()
        for index in indices:
            graph = dataset[index]
            edge_count += int(graph.edge_index.shape[1])
        elapsed = time.perf_counter() - start
    return PassMetrics(
        elapsed_seconds=elapsed,
        edge_count=edge_count,
        start_rss_bytes=memory.start_rss_bytes,
        end_rss_bytes=memory.end_rss_bytes,
        peak_rss_bytes=memory.peak_rss_bytes,
    )


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.memory_sample_ms <= 0:
        raise ValueError("--memory-sample-ms must be positive.")

    numbers = atomic_numbers(args.path)
    python_dataset = HDF5GraphDataset(
        args.path,
        r_max=args.r_max,
        atomic_numbers=AtomicNumberTable(numbers),
    )
    rust_dataset = equiload.HDF5GraphDataset.from_hdf5_file(
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

    results: dict[str, list[PassMetrics]] = {
        "Equitrain/Matscipy": [],
        "EquiLoad": [],
    }
    edge_counts: dict[str, int] = {}
    for repeat in range(args.repeats):
        order = list(results) if repeat % 2 == 0 else list(reversed(results))
        datasets = {
            "Equitrain/Matscipy": python_dataset,
            "EquiLoad": rust_dataset,
        }
        for name in order:
            metrics = timed_pass(
                datasets[name], indices, args.memory_sample_ms / 1000.0
            )
            results[name].append(metrics)
            edge_counts[name] = metrics.edge_count
            print(
                f"Run {repeat + 1:>2}/{args.repeats} {name:20s} "
                f"{metrics.elapsed_seconds:9.4f} s  "
                f"{count / metrics.elapsed_seconds:10.2f} structures/s  "
                f"peak RSS {format_bytes(metrics.peak_rss_bytes):>10s}  "
                f"growth +{format_bytes(metrics.peak_rss_growth_bytes):>10s}"
            )

    if len(set(edge_counts.values())) != 1:
        raise AssertionError(f"edge-count mismatch: {edge_counts}")

    elapsed_medians = {
        name: statistics.median(result.elapsed_seconds for result in values)
        for name, values in results.items()
    }
    peak_rss_medians = {
        name: statistics.median(result.peak_rss_bytes for result in values)
        for name, values in results.items()
    }
    rss_growth_medians = {
        name: statistics.median(result.peak_rss_growth_bytes for result in values)
        for name, values in results.items()
    }

    print(f"\nStructures: {count}; edges/pass: {next(iter(edge_counts.values()))}")
    print("Median results:")
    for name, elapsed in elapsed_medians.items():
        print(
            f"{name:20s} {elapsed:9.4f} s  "
            f"{count / elapsed:10.2f} structures/s  "
            f"peak RSS {format_bytes(peak_rss_medians[name]):>10s}  "
            f"growth +{format_bytes(rss_growth_medians[name]):>10s}"
        )
    print(
        "EquiLoad speedup: "
        f"{elapsed_medians['Equitrain/Matscipy'] / elapsed_medians['EquiLoad']:.3f}x"
    )
    print(
        "Memory is full-process RSS; growth is each pass's sampled peak minus "
        "its starting RSS."
    )

    python_dataset.close()
    rust_dataset.close()


if __name__ == "__main__":
    main()
