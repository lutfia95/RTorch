#!/usr/bin/env python3
"""Compare Equitrain/Python and RTorch loader benchmark JSON reports."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMPLEMENTATIONS = ("Python", "RTorch")
COLORS = {"Python": "#4C78A8", "RTorch": "#F58518"}


@dataclass(frozen=True)
class Comparison:
    label: str
    expected_structures: int
    reports: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-50k", type=Path, default=Path("python_50k_results.json")
    )
    parser.add_argument(
        "--rtorch-50k", type=Path, default=Path("rtorch_50k_results.json")
    )
    parser.add_argument(
        "--python-100k", type=Path, default=Path("python_100k_results.json")
    )
    parser.add_argument(
        "--rtorch-100k", type=Path, default=Path("rtorch_100k_results.json")
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("loader_comparison.png"),
        help="Output image (default: loader_comparison.png)",
    )
    parser.add_argument(
        "--memory-metric",
        choices=("rss", "pss", "uss"),
        default="rss",
        help="Memory measurement used in plots (default: rss)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Plot complete pairs and warn about missing reports.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Open an interactive plot window."
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    required = ("implementation", "settings", "runs", "memory_after_loader")
    missing = [field for field in required if field not in report]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if not report["runs"]:
        raise ValueError(f"{path} contains no measured runs")
    return report


def run_structures(run: dict[str, Any]) -> int:
    return int(run["counts"]["structures"])


def validate_report(path: Path, report: dict[str, Any], expected: int) -> None:
    counts = report["runs"][0]["counts"]
    if int(counts["structures"]) != expected:
        raise ValueError(
            f"{path} processed {counts['structures']} structures; expected {expected}"
        )
    for index, run in enumerate(report["runs"][1:], start=2):
        if run["counts"] != counts:
            raise ValueError(f"{path} run {index} has different work counts")


def validate_pair(comparison: Comparison, paths: dict[str, Path]) -> None:
    python_report = comparison.reports["Python"]
    rtorch_report = comparison.reports["RTorch"]
    python_counts = python_report["runs"][0]["counts"]
    rtorch_counts = rtorch_report["runs"][0]["counts"]
    if python_counts != rtorch_counts:
        raise ValueError(
            f"{comparison.label} reports did different work:\n"
            f"  {paths['Python']}: {python_counts}\n"
            f"  {paths['RTorch']}: {rtorch_counts}"
        )

    settings_to_match = ("r_max", "batch_size", "max_batches")
    mismatches = [
        name
        for name in settings_to_match
        if python_report["settings"].get(name) != rtorch_report["settings"].get(name)
    ]
    if mismatches:
        raise ValueError(
            f"{comparison.label} settings differ: {', '.join(mismatches)}"
        )


def collect_comparisons(args: argparse.Namespace) -> list[Comparison]:
    specifications = (
        (
            "50k",
            50_000,
            {"Python": args.python_50k, "RTorch": args.rtorch_50k},
        ),
        (
            "100k",
            100_000,
            {"Python": args.python_100k, "RTorch": args.rtorch_100k},
        ),
    )
    comparisons: list[Comparison] = []
    for label, expected, paths in specifications:
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            message = f"{label}: missing {', '.join(missing)}"
            if args.allow_missing:
                print(f"Warning: {message}; skipping this pair.")
                continue
            raise FileNotFoundError(message)

        reports = {
            implementation: load_report(path)
            for implementation, path in paths.items()
        }
        for implementation, report in reports.items():
            validate_report(paths[implementation], report, expected)
        comparison = Comparison(label, expected, reports)
        validate_pair(comparison, paths)
        comparisons.append(comparison)

    if not comparisons:
        raise ValueError("No complete comparison pairs were found")
    return comparisons


def wall_values(report: dict[str, Any]) -> list[float]:
    return [float(run["wall_seconds"]) for run in report["runs"]]


def throughput_values(report: dict[str, Any]) -> list[float]:
    return [
        run_structures(run) / float(run["wall_seconds"]) for run in report["runs"]
    ]


def cpu_percent_values(report: dict[str, Any]) -> list[float]:
    return [
        100.0 * float(run["cpu_seconds"]) / float(run["wall_seconds"])
        for run in report["runs"]
    ]


def memory_after_loader(report: dict[str, Any], metric: str) -> float:
    return float(report["memory_after_loader"][metric])


def maximum_memory(report: dict[str, Any], metric: str) -> float:
    return max(float(run["memory_peak"][metric]) for run in report["runs"])


def retained_memory_by_run(report: dict[str, Any], metric: str) -> list[float]:
    return [
        float(run["memory_end"][metric]) - float(run["memory_start"][metric])
        for run in report["runs"]
    ]


def total_retained_memory(report: dict[str, Any], metric: str) -> float:
    runs = report["runs"]
    return float(runs[-1]["memory_end"][metric]) - float(
        runs[0]["memory_start"][metric]
    )


def implementation_centers(index: int, width: float = 0.36) -> dict[str, float]:
    return {
        "Python": index - width / 2,
        "RTorch": index + width / 2,
    }


def add_value_label(axis: Any, bar: Any, text: str) -> None:
    axis.annotate(
        text,
        (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
    )


def plot_median_bars(
    axis: Any,
    comparisons: list[Comparison],
    value_fn: Any,
    ylabel: str,
    title: str,
) -> None:
    width = 0.36
    for implementation in IMPLEMENTATIONS:
        centers = [
            implementation_centers(index, width)[implementation]
            for index in range(len(comparisons))
        ]
        values_by_workload = [
            value_fn(comparison.reports[implementation])
            for comparison in comparisons
        ]
        medians = [statistics.median(values) for values in values_by_workload]
        bars = axis.bar(
            centers,
            medians,
            width,
            label=implementation,
            color=COLORS[implementation],
            alpha=0.84,
        )
        for center, values, bar, median in zip(
            centers, values_by_workload, bars, medians, strict=True
        ):
            count = len(values)
            offsets = [0.0] if count == 1 else [0.10 * (i / (count - 1) - 0.5) for i in range(count)]
            axis.scatter(
                [center + offset for offset in offsets],
                values,
                s=24,
                color="white",
                edgecolor=COLORS[implementation],
                linewidth=1.2,
                zorder=3,
            )
            add_value_label(axis, bar, f"{median:,.1f}")

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(len(comparisons)), [item.label for item in comparisons])
    axis.grid(axis="y", alpha=0.25)


def plot_memory_footprint(
    axis: Any, comparisons: list[Comparison], metric: str
) -> None:
    width = 0.36
    gib = 1024.0**3
    for implementation in IMPLEMENTATIONS:
        centers = [
            implementation_centers(index, width)[implementation]
            for index in range(len(comparisons))
        ]
        peaks = [
            maximum_memory(item.reports[implementation], metric) / gib
            for item in comparisons
        ]
        initialized = [
            memory_after_loader(item.reports[implementation], metric) / gib
            for item in comparisons
        ]
        bars = axis.bar(
            centers,
            peaks,
            width,
            label=implementation,
            color=COLORS[implementation],
            alpha=0.84,
        )
        axis.scatter(
            centers,
            initialized,
            marker="D",
            s=34,
            color="black",
            zorder=3,
            label="After loader init" if implementation == "Python" else None,
        )
        for bar, peak in zip(bars, peaks, strict=True):
            add_value_label(axis, bar, f"{peak:.2f}")

    axis.set_title(f"Process memory footprint ({metric.upper()})")
    axis.set_ylabel("GiB")
    axis.set_xticks(range(len(comparisons)), [item.label for item in comparisons])
    axis.grid(axis="y", alpha=0.25)


def plot_retained_memory(
    axis: Any, comparisons: list[Comparison], metric: str
) -> None:
    width = 0.36
    mib = 1024.0**2
    hatches = ("", "///", "xxx", "...")
    for implementation in IMPLEMENTATIONS:
        centers = [
            implementation_centers(index, width)[implementation]
            for index in range(len(comparisons))
        ]
        bottoms = [0.0] * len(comparisons)
        run_growth = [
            retained_memory_by_run(item.reports[implementation], metric)
            for item in comparisons
        ]
        repeat_count = max(len(values) for values in run_growth)
        for repeat in range(repeat_count):
            values = [
                max(0.0, growth[repeat]) / mib if repeat < len(growth) else 0.0
                for growth in run_growth
            ]
            axis.bar(
                centers,
                values,
                width,
                bottom=bottoms,
                color=COLORS[implementation],
                alpha=0.84,
                hatch=hatches[repeat % len(hatches)],
                edgecolor="white" if repeat else COLORS[implementation],
                linewidth=0.8,
                label=(
                    f"Run {repeat + 1} retained"
                    if implementation == "Python"
                    else None
                ),
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
        for center, total in zip(centers, bottoms, strict=True):
            axis.annotate(
                f"{total:.1f}",
                (center, total),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_title(f"Retained memory added by repeats ({metric.upper()})")
    axis.set_ylabel("MiB (stacked by run)")
    axis.set_xticks(range(len(comparisons)), [item.label for item in comparisons])
    axis.grid(axis="y", alpha=0.25)


def print_summary(comparisons: list[Comparison], metric: str) -> None:
    headers = (
        "Size",
        "Implementation",
        "Wall median (s)",
        "Structures/s",
        "CPU median (%)",
        f"Peak {metric.upper()} (GiB)",
        f"Retained {metric.upper()} (MiB)",
        "Init (s)",
    )
    rows: list[tuple[str, ...]] = []
    gib = 1024.0**3
    mib = 1024.0**2
    for comparison in comparisons:
        for implementation in IMPLEMENTATIONS:
            report = comparison.reports[implementation]
            rows.append(
                (
                    comparison.label,
                    implementation,
                    f"{statistics.median(wall_values(report)):.3f}",
                    f"{statistics.median(throughput_values(report)):.2f}",
                    f"{statistics.median(cpu_percent_values(report)):.1f}",
                    f"{maximum_memory(report, metric) / gib:.3f}",
                    f"{total_retained_memory(report, metric) / mib:.1f}",
                    f"{float(report['loader_initialization_seconds']):.3f}",
                )
            )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)))

    print()
    for comparison in comparisons:
        python_wall = statistics.median(wall_values(comparison.reports["Python"]))
        rtorch_wall = statistics.median(wall_values(comparison.reports["RTorch"]))
        print(f"{comparison.label} RTorch speedup: {python_wall / rtorch_wall:.2f}x")


def make_plot(
    comparisons: list[Comparison], output: Path, metric: str, show: bool
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rtorch-matplotlib-cache")
    )
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    plot_median_bars(
        axes[0, 0],
        comparisons,
        throughput_values,
        "Structures per second",
        "Loader throughput (bars = median, dots = runs)",
    )
    plot_median_bars(
        axes[0, 1],
        comparisons,
        wall_values,
        "Seconds",
        "Wall time per pass",
    )
    plot_memory_footprint(axes[1, 0], comparisons, metric)
    plot_retained_memory(axes[1, 1], comparisons, metric)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    footprint_handles, footprint_labels = axes[1, 0].get_legend_handles_labels()
    retained_handles, retained_labels = axes[1, 1].get_legend_handles_labels()
    combined = dict(
        zip(
            labels + footprint_labels + retained_labels,
            handles + footprint_handles + retained_handles,
            strict=True,
        )
    )
    figure.legend(
        combined.values(),
        combined.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncols=min(5, len(combined)),
        frameon=False,
    )
    figure.suptitle(
        "Equitrain/Python vs RTorch HDF5 graph-loader benchmark",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )
    figure.subplots_adjust(
        left=0.07, right=0.985, bottom=0.07, top=0.88, hspace=0.30, wspace=0.12
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Plot: {output.resolve()}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    comparisons = collect_comparisons(args)
    print_summary(comparisons, args.memory_metric)
    make_plot(comparisons, args.output, args.memory_metric, args.show)


if __name__ == "__main__":
    main()
