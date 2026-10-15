"""Plot the runtime matrix and isolated large-batch peak-memory comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path, default=Path('optimization_results'))
args = parser.parse_args()
summary = json.loads((args.results_dir / 'summary.json').read_text())
rows = sorted(summary['matrix'], key=lambda r: (r['dataset'] != 'train.h5', r['shuffle'], r['batch_size']))
labels = [f"{r['structures']:,} {'validation' if r['dataset'] == 'valid.h5' else 'training'}\n"
          f"batch {r['batch_size']}{', shuffled' if r['shuffle'] else ', sequential'}" for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [1.8, 1]})
colors = ['#64748b', '#0891b2']
y = np.arange(len(rows))
for offset, field, color, label in [(-.18, 'before_seconds', colors[0], 'Previous RTorch'),
                                   (.18, 'after_seconds', colors[1], 'Optimized RTorch')]:
    bars = axes[0].barh(y + offset, [r[field] for r in rows], height=.32, color=color, label=label)
    axes[0].bar_label(bars, fmt='%.2f s', padding=4, fontsize=9)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlabel('Wall time (seconds) — median of two passes')
axes[0].set_title('Runtime across access patterns')
axes[0].set_xlim(0, max(r['before_seconds'] for r in rows) * 1.23)
axes[0].legend(loc='lower right', frameon=False)
large = summary['large_batch']
values = [large['before_high_water_bytes'] / 2**30, large['after_high_water_bytes'] / 2**30]
bars = axes[1].bar(['Previous', 'Optimized'], values, color=colors, width=.6)
axes[1].bar_label(bars, fmt='%.2f GiB', padding=5)
axes[1].set_ylabel('Process high-water mark (GiB)')
axes[1].set_ylim(0, max(values) * 1.25)
axes[1].set_title('Large-batch peak memory\n16,384 structures; batch size 8,192')
axes[1].text(.5, max(values) * 1.15,
             f"{large['peak_memory_reduction_fraction']:.0%} lower peak memory",
             ha='center', fontsize=11, color=colors[1])
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x' if ax is axes[0] else 'y', alpha=.15)
    ax.set_axisbelow(True)
fig.suptitle('RTorch loader optimizations — MP trajectory datasets, r_max = 5.0', fontsize=13)
fig.tight_layout()
fig.savefig(args.results_dir / 'comparison.png', dpi=180)
print(args.results_dir / 'comparison.png')
