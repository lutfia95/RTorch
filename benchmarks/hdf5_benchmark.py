from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import torch


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "test_data" / "train.h5",
        help="Equitrain-format HDF5 file (default: test_data/train.h5)",
    )
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--atomic-numbers",
        help="Comma-separated atomic numbers; by default they are read from the file.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Outer loader batches per pass; 0 consumes the complete file.",
    )
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--memory-sample-ms",
        type=float,
        default=5.0,
        help="Interval for Linux RSS/PSS/USS sampling (default: 5 ms).",
    )
    parser.add_argument(
        "--transfer-device",
        help="Move every graph batch to this CUDA device and include synchronized transfer time.",
    )
    parser.add_argument(
        "--skip-python-heap",
        action="store_true",
        help="Skip the extra pass that profiles Python allocations with tracemalloc.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="Also write all measurements to this JSON file.",
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {args.path}")
    if args.r_max <= 0:
        raise ValueError("--r-max must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_batches < 0:
        raise ValueError("--max-batches cannot be negative")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches cannot be negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.memory_sample_ms <= 0:
        raise ValueError("--memory-sample-ms must be positive")


def resolve_transfer_device(value: str | None) -> torch.device | None:
    if value is None:
        return None
    try:
        device = torch.device(value)
    except RuntimeError as error:
        raise ValueError(f"Invalid --transfer-device {value!r}: {error}") from error
    if device.type != "cuda":
        raise ValueError("--transfer-device must name a CUDA device, for example cuda or cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("--transfer-device was supplied but CUDA is not available to PyTorch")
    return device


def resolve_atomic_numbers(args: argparse.Namespace) -> list[int]:
    if args.atomic_numbers:
        numbers = sorted({int(value) for value in args.atomic_numbers.split(",")})
    else:
        unique_numbers: set[int] = set()
        with h5py.File(args.path, "r") as handle:
            dataset = handle["atomic_numbers"]
            chunk_size = 1_000_000
            for start in range(0, len(dataset), chunk_size):
                unique_numbers.update(int(value) for value in dataset[start : start + chunk_size])
        numbers = sorted(unique_numbers)
    if not numbers:
        raise ValueError("No atomic numbers were supplied or found in the HDF5 file")
    return numbers


def _read_kib_file(path: str) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(path, encoding="ascii") as handle:
            for line in handle:
                key, separator, remainder = line.partition(":")
                if not separator:
                    continue
                fields = remainder.split()
                if fields and fields[0].isdigit():
                    values[key] = int(fields[0]) * 1024
    except (FileNotFoundError, OSError, PermissionError):
        pass
    return values


@dataclass(frozen=True)
class MemorySnapshot:
    rss: int
    pss: int
    uss: int
    shared: int
    swap: int
    process_high_water: int

    @classmethod
    def current(cls) -> "MemorySnapshot":
        values = _read_kib_file("/proc/self/smaps_rollup")
        rss = values.get("Rss", 0)
        if not rss:
            try:
                fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
                rss = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
            except (FileNotFoundError, IndexError, OSError, ValueError):
                rss = 0
        uss = sum(
            values.get(key, 0) for key in ("Private_Clean", "Private_Dirty", "Private_Hugetlb")
        )
        shared = sum(
            values.get(key, 0) for key in ("Shared_Clean", "Shared_Dirty", "Shared_Hugetlb")
        )
        high_water_scale = 1 if sys.platform == "darwin" else 1024
        high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * high_water_scale
        return cls(
            rss=rss,
            pss=values.get("Pss", rss),
            uss=uss or rss,
            shared=shared,
            swap=values.get("Swap", 0),
            process_high_water=high_water,
        )


def _maximum(left: MemorySnapshot, right: MemorySnapshot) -> MemorySnapshot:
    return MemorySnapshot(
        **{
            field: max(getattr(left, field), getattr(right, field))
            for field in MemorySnapshot.__dataclass_fields__
        }
    )


class MemorySampler:
    """Sample process memory while the main thread consumes a loader."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.start = MemorySnapshot.current()
        self.end = self.start
        self.peak = self.start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        self.peak = _maximum(self.peak, MemorySnapshot.current())

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "MemorySampler":
        self.start = MemorySnapshot.current()
        self.peak = self.start
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._sample()
        self._stop.set()
        self._thread.join()
        self.end = MemorySnapshot.current()
        self.peak = _maximum(self.peak, self.end)


@dataclass(frozen=True)
class WorkCounts:
    outer_batches: int = 0
    graph_batches: int = 0
    structures: int = 0
    nodes: int = 0
    edges: int = 0


@dataclass(frozen=True)
class PassMetrics:
    wall_seconds: float
    cpu_seconds: float
    counts: WorkCounts
    memory_start: MemorySnapshot
    memory_end: MemorySnapshot
    memory_peak: MemorySnapshot
    minor_page_faults: int
    major_page_faults: int
    input_blocks: int
    voluntary_context_switches: int
    involuntary_context_switches: int
    python_heap_end: int | None = None
    python_heap_peak: int | None = None


def _consume(
    loader: Iterable[Any],
    max_batches: int,
    *,
    transfer_device: torch.device | None = None,
) -> WorkCounts:
    outer_batches = graph_batches = structures = nodes = edges = 0
    for outer_batch in loader:
        outer_batches += 1
        for batch in outer_batch:
            graph_batches += 1
            structures += int(batch.num_graphs)
            nodes += int(batch.num_nodes)
            edges += int(batch.num_edges)
            if transfer_device is not None:
                batch.to(transfer_device, non_blocking=True)
        if max_batches and outer_batches >= max_batches:
            break
    if transfer_device is not None:
        torch.cuda.synchronize(transfer_device)
    return WorkCounts(outer_batches, graph_batches, structures, nodes, edges)


def measured_pass(
    loader: Iterable[Any],
    *,
    max_batches: int,
    sample_seconds: float,
    trace_python_heap: bool = False,
    transfer_device: torch.device | None = None,
) -> PassMetrics:
    gc.collect()
    if transfer_device is not None:
        torch.cuda.synchronize(transfer_device)
    if trace_python_heap:
        tracemalloc.start()
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_start = time.process_time()
    with MemorySampler(sample_seconds) as memory:
        wall_start = time.perf_counter()
        counts = _consume(loader, max_batches, transfer_device=transfer_device)
        wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    after_usage = resource.getrusage(resource.RUSAGE_SELF)
    python_end = python_peak = None
    if trace_python_heap:
        python_end, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return PassMetrics(
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        counts=counts,
        memory_start=memory.start,
        memory_end=memory.end,
        memory_peak=memory.peak,
        minor_page_faults=after_usage.ru_minflt - before_usage.ru_minflt,
        major_page_faults=after_usage.ru_majflt - before_usage.ru_majflt,
        input_blocks=after_usage.ru_inblock - before_usage.ru_inblock,
        voluntary_context_switches=after_usage.ru_nvcsw - before_usage.ru_nvcsw,
        involuntary_context_switches=after_usage.ru_nivcsw - before_usage.ru_nivcsw,
        python_heap_end=python_end,
        python_heap_peak=python_peak,
    )


def _format_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _memory_line(label: str, snapshot: MemorySnapshot) -> str:
    return (
        f"  {label:>6}: RSS {_format_bytes(snapshot.rss):>10}, "
        f"PSS {_format_bytes(snapshot.pss):>10}, "
        f"USS {_format_bytes(snapshot.uss):>10}, "
        f"shared {_format_bytes(snapshot.shared):>10}, "
        f"swap {_format_bytes(snapshot.swap):>10}, "
        f"process HWM {_format_bytes(snapshot.process_high_water):>10}"
    )


def print_pass(label: str, metrics: PassMetrics) -> None:
    counts = metrics.counts
    rate = counts.structures / metrics.wall_seconds if metrics.wall_seconds else 0.0
    cpu_percent = 100 * metrics.cpu_seconds / metrics.wall_seconds if metrics.wall_seconds else 0.0
    print(
        f"{label}: {metrics.wall_seconds:.6f} s wall, "
        f"{metrics.cpu_seconds:.6f} s CPU ({cpu_percent:.1f}%), "
        f"{rate:.2f} structures/s"
    )
    print(
        f"  work: {counts.outer_batches} loader batches, "
        f"{counts.graph_batches} graph batches, {counts.structures} structures, "
        f"{counts.nodes} nodes, {counts.edges} edges"
    )
    print(_memory_line("start", metrics.memory_start))
    print(_memory_line("peak", metrics.memory_peak))
    print(_memory_line("end", metrics.memory_end))
    print(
        "  peak growth: "
        f"RSS +{_format_bytes(max(0, metrics.memory_peak.rss - metrics.memory_start.rss))}, "
        f"PSS +{_format_bytes(max(0, metrics.memory_peak.pss - metrics.memory_start.pss))}, "
        f"USS +{_format_bytes(max(0, metrics.memory_peak.uss - metrics.memory_start.uss))}"
    )
    print(
        f"  OS: minor/major faults {metrics.minor_page_faults}/"
        f"{metrics.major_page_faults}, input blocks {metrics.input_blocks}, "
        f"voluntary/involuntary context switches "
        f"{metrics.voluntary_context_switches}/"
        f"{metrics.involuntary_context_switches}"
    )
    if metrics.python_heap_peak is not None:
        print(
            f"  Python heap: end {_format_bytes(metrics.python_heap_end or 0)}, "
            f"peak {_format_bytes(metrics.python_heap_peak)}"
        )


def _close_dataset(dataset: Any) -> None:
    nested = getattr(dataset, "datasets", None)
    if nested is not None:
        for child in nested:
            _close_dataset(child)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def run_benchmark(
    implementation: str,
    args: argparse.Namespace,
    build_loader: Callable[[list[int]], Any],
) -> None:
    validate_args(args)
    numbers = resolve_atomic_numbers(args)
    transfer_device = resolve_transfer_device(args.transfer_device)
    baseline = MemorySnapshot.current()
    init_start = time.perf_counter()
    loader = build_loader(numbers)
    init_seconds = time.perf_counter() - init_start
    after_init = MemorySnapshot.current()

    print(f"Implementation: {implementation}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Dataset: {args.path.resolve()}")
    print(f"Atomic numbers: {numbers}")
    print(
        f"Settings: r_max={args.r_max}, batch_size={args.batch_size}, "
        f"max_batches={args.max_batches or 'all'}, repeats={args.repeats}, "
        f"transfer_device={transfer_device or 'none'}"
    )
    print(f"Loader initialization: {init_seconds:.6f} s")
    print(_memory_line("before", baseline))
    print(_memory_line("after", after_init))

    if args.warmup_batches:
        warmup = _consume(loader, args.warmup_batches, transfer_device=transfer_device)
        print(f"Warm-up: {warmup.outer_batches} loader batches, {warmup.structures} structures")

    runs: list[PassMetrics] = []
    for index in range(args.repeats):
        metrics = measured_pass(
            loader,
            max_batches=args.max_batches,
            sample_seconds=args.memory_sample_ms / 1000,
            transfer_device=transfer_device,
        )
        runs.append(metrics)
        print_pass(f"Run {index + 1}/{args.repeats}", metrics)

    first_counts = runs[0].counts
    if any(run.counts != first_counts for run in runs[1:]):
        raise RuntimeError("The loader produced different work counts between repeats")

    wall_values = [run.wall_seconds for run in runs]
    median_wall = statistics.median(wall_values)
    print(
        f"Summary: median {median_wall:.6f} s "
        f"(min {min(wall_values):.6f}, max {max(wall_values):.6f}), "
        f"{first_counts.structures / median_wall:.2f} structures/s"
    )

    heap_profile = None
    if not args.skip_python_heap:
        print("Python heap profile (separate pass; excluded from runtime summary):")
        heap_profile = measured_pass(
            loader,
            max_batches=args.max_batches,
            sample_seconds=args.memory_sample_ms / 1000,
            trace_python_heap=True,
            transfer_device=transfer_device,
        )
        print_pass("Heap profile", heap_profile)

    if args.json_path:
        report = {
            "implementation": implementation,
            "python": sys.version,
            "executable": sys.executable,
            "dataset": str(args.path.resolve()),
            "atomic_numbers": numbers,
            "settings": {
                "r_max": args.r_max,
                "batch_size": args.batch_size,
                "max_batches": args.max_batches,
                "warmup_batches": args.warmup_batches,
                "repeats": args.repeats,
                "memory_sample_ms": args.memory_sample_ms,
                **({"transfer_device": str(transfer_device)} if transfer_device else {}),
            },
            "loader_initialization_seconds": init_seconds,
            "memory_before_loader": asdict(baseline),
            "memory_after_loader": asdict(after_init),
            "runs": [asdict(run) for run in runs],
            "summary": {
                "median_wall_seconds": median_wall,
                "min_wall_seconds": min(wall_values),
                "max_wall_seconds": max(wall_values),
                "median_structures_per_second": first_counts.structures / median_wall,
            },
            "python_heap_profile": asdict(heap_profile) if heap_profile else None,
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {args.json_path.resolve()}")

    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        _close_dataset(dataset)
