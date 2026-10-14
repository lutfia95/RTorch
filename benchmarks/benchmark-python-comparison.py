#!/usr/bin/env python3
"""Run the checked-out Python and native loaders serially in fresh processes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
CASES = (
    ("50k", "train.h5", 2, 25_000, 50_000),
    ("100k", "train.h5", 2, 50_000, 100_000),
    ("small_2k", "train.h5", 2, 1_000, 2_000),
    ("full_valid", "valid.h5", 2, 0, 82_500),
    ("100k_batch32", "train.h5", 32, 3_125, 100_000),
    ("large_batch", "train.h5", 8_192, 2, 16_384),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equiload_extension_path() -> Path:
    spec = importlib.util.find_spec("equiload.equiload")
    if spec is None or spec.origin is None:
        raise RuntimeError("The installed equiload extension could not be located")
    return Path(spec.origin).resolve()


def equitrain_source_root() -> Path:
    import equitrain

    package_file = getattr(equitrain, "__file__", None)
    if package_file is None:
        raise RuntimeError("The installed equitrain package has no importable source path")
    return Path(package_file).resolve().parent


def source_key(path: Path, equitrain_root: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(Path("equitrain") / path.relative_to(equitrain_root))


def summarize(
    output: Path,
    repeats: int,
    datasets: Path,
    transfer_device: str | None,
) -> dict:
    pairs = []
    for name, filename, batch_size, max_batches, structures in CASES:
        reports = {}
        for implementation in ("python", "equiload"):
            path = output / f"{implementation}_{name}.json"
            report = json.loads(path.read_text())
            assert report["dataset"] == str((datasets / filename).resolve()), path
            expected_settings = dict(
                r_max=5., batch_size=batch_size, max_batches=max_batches,
                warmup_batches=1, repeats=repeats, memory_sample_ms=250.,
            )
            if transfer_device:
                expected_settings["transfer_device"] = transfer_device
            assert report["settings"] == expected_settings, path
            assert len(report["runs"]) == repeats, path
            counts = report["runs"][0]["counts"]
            assert counts["structures"] == structures, path
            assert all(run["counts"] == counts for run in report["runs"]), path
            reports[implementation] = report
        python, native = reports["python"], reports["equiload"]
        assert python["runs"][0]["counts"] == native["runs"][0]["counts"], name
        assert python["atomic_numbers"] == native["atomic_numbers"], name
        assert python["python"] == native["python"], name
        assert python["executable"] == native["executable"], name
        values = {}
        for implementation, report in reports.items():
            runs = report["runs"]
            values[implementation] = dict(
                median_wall_seconds=statistics.median(r["wall_seconds"] for r in runs),
                wall_seconds=[r["wall_seconds"] for r in runs],
                peak_rss_bytes=max(r["memory_peak"]["rss"] for r in runs),
                process_high_water_bytes=max(r["memory_peak"]["process_high_water"] for r in runs),
                peak_pss_bytes=max(r["memory_peak"]["pss"] for r in runs),
                peak_uss_bytes=max(r["memory_peak"]["uss"] for r in runs),
                initialization_seconds=report["loader_initialization_seconds"],
                rss_before_loader_bytes=report["memory_before_loader"]["rss"],
                retained_rss_bytes=runs[-1]["memory_end"]["rss"] - runs[0]["memory_start"]["rss"],
            )
        pairs.append(dict(
            case=name, dataset=filename, batch_size=batch_size,
            counts=python["runs"][0]["counts"], **values,
            speedup=values["python"]["median_wall_seconds"] / values["equiload"]["median_wall_seconds"],
            peak_rss_reduction_percent=100 * (1 - values["equiload"]["peak_rss_bytes"] / values["python"]["peak_rss_bytes"]),
        ))
    equitrain_root = equitrain_source_root()
    sources = (
        BENCHMARKS / "python_loader_test.py", BENCHMARKS / "equiload_test.py", BENCHMARKS / "hdf5_benchmark.py",
        equitrain_root / "data/backend_torch/loaders_impl.py",
        equitrain_root / "data/format_hdf5/dataset.py", equiload_extension_path(),
    )
    return dict(
        pairs=pairs,
        created_at=datetime.now(timezone.utc).isoformat(),
        executable=python["executable"], python=python["python"],
        environment={key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "EQUILOAD_CPU_BRIDGE", "HDF5_ROOT", "CUDA_VISIBLE_DEVICES",
        )},
        methodology="Local python_loader_test.py versus equiload_test.py; serial fresh processes, "
        "one warmup batch, two repeats by default, 250 ms memory sampling, no heap profiling. "
        "Runtime measures consumption after initialization. Memory includes interpreter/imports; "
        "process high-water captures brief peaks missed by RSS sampling. Counts match exactly.",
        source_sha256={source_key(path, equitrain_root): digest(path) for path in sources},
    )


def plot(summary: dict, output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/equiload-comparison-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pairs = summary["pairs"]
    x = np.arange(len(pairs))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
    colors = dict(python="#4C78A8", equiload="#F58518")
    for implementation, shift in (("python", -.19), ("equiload", .19)):
        for axis, key, scale in ((axes[0], "median_wall_seconds", 1),
                                 (axes[1], "peak_rss_bytes", 2**20),
                                 (axes[2], "process_high_water_bytes", 2**20)):
            heights = [p[implementation][key] / scale for p in pairs]
            bars = axis.bar(x + shift, heights, .38, label=implementation.title(), color=colors[implementation])
            axis.bar_label(bars, labels=[f"{v:.1f}" for v in heights], fontsize=7, padding=2)
    labels = ["Train 50k\nB=2", "Train 100k\nB=2", "Train 2k\nB=2",
              "Valid 82.5k\nB=2", "Train 100k\nB=32", "Train 16,384\nB=8192"]
    for axis, title, ylabel in zip(axes, ("Runtime (median)", "Sampled peak RSS", "Process memory high-water"),
                                   ("Seconds", "MiB", "MiB")):
        axis.set(title=title, ylabel=ylabel, xticks=x, xticklabels=labels)
        axis.tick_params(axis="x", labelsize=8)
        axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
        axis.legend()
    fig.suptitle("Fresh local Python versus optimized EquiLoad — matching structures, nodes and edges")
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "../../mptraj")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARKS / "results")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--transfer-device", help="Move every graph batch to this CUDA device during timing.")
    parser.add_argument("--implementations", nargs="+", choices=("python", "equiload"), default=("python", "equiload"),
                        help="Select loaders to run; the final summary still requires both reports.")
    parser.add_argument("--resume", action="store_true", help="Reuse reports already produced in this directory.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, filename, batch_size, max_batches, _ in CASES:
        for implementation in args.implementations:
            report = output / f"{implementation}_{name}.json"
            if args.resume and report.exists():
                previous = json.loads(report.read_text())
                if previous.get("settings", {}).get("transfer_device") == args.transfer_device:
                    print(f"Reusing {report}", flush=True)
                    continue
            script = BENCHMARKS / ("python_loader_test.py" if implementation == "python" else "equiload_test.py")
            command = [sys.executable, "-u", str(script),
                       str((args.dataset_dir / filename).resolve()), "--r-max", "5.0", "--batch-size", str(batch_size),
                       "--max-batches", str(max_batches), "--repeats", str(args.repeats), "--skip-python-heap",
                       "--memory-sample-ms", "250", "--json", str(report)]
            if args.transfer_device:
                command.extend(["--transfer-device", args.transfer_device])
            print(f"Starting {implementation}: {name}", flush=True)
            with (output / f"{implementation}_{name}.log").open("w") as log:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith(("Run ", "Summary:")):
                            print(f"{implementation} {name}: {line.rstrip()}", flush=True)
                    status = child.wait()
                    if status:
                        raise subprocess.CalledProcessError(status, command)
            results = json.loads(report.read_text())
            print(f"Completed {implementation}: {name}, median {results['summary']['median_wall_seconds']:.3f} s", flush=True)
    summary = summarize(output, args.repeats, args.dataset_dir, args.transfer_device)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, output)
    for pair in summary["pairs"]:
        print(f"{pair['case']}: {pair['speedup']:.2f}x speedup; RSS change {pair['peak_rss_reduction_percent']:.1f}% lower")


if __name__ == "__main__":
    main()
