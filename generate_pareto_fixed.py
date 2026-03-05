#!/usr/bin/env python3
"""
Publication-quality Pareto frontier figure for ECCV 2026 S³-Merge paper.
Fixes: removed confusing arrow, cleaner labels, professional styling.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Use measured 50k 3-seed data (from sr_tome_50k_3seed.csv + sr_tome_pareto_sweep.csv)
    methods = {
        'Baseline':     {'thr': 2674, 'acc': 79.72, 'color': '#2c3e50', 'marker': 'o', 'ms': 10},
        'ToMe r=8':     {'thr': 2800, 'acc': 79.72 - 0.38, 'color': '#8e44ad', 'marker': 'p', 'ms': 10},
        'ToMe r=10':    {'thr': 3000, 'acc': 79.08, 'color': '#3498db', 'marker': 'D', 'ms': 9},
        'ToMe r=12':    {'thr': 3050, 'acc': 78.91, 'color': '#2980b9', 'marker': 'd', 'ms': 9},
        'S³-Merge l8':   {'thr': 3131, 'acc': 78.97, 'color': '#e74c3c', 'marker': 's', 'ms': 10},
        'S³-Merge l6':   {'thr': 3529, 'acc': 78.63, 'color': '#27ae60', 'marker': '^', 'ms': 11},
    }

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 9.5,
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    # Plot each method
    for name, d in methods.items():
        zorder = 6 if 'S³-Merge' in name else 5
        ax.scatter(d['thr'], d['acc'], s=d['ms']**2, c=d['color'], marker=d['marker'],
                   label=name, zorder=zorder, edgecolors='white', linewidths=0.8)

    # S³-Merge Pareto frontier line
    sr_pts = sorted([(methods[k]['thr'], methods[k]['acc']) for k in ['Baseline', 'S³-Merge l8', 'S³-Merge l6']])
    ax.plot([p[0] for p in sr_pts], [p[1] for p in sr_pts], '--', color='#27ae60', alpha=0.5, linewidth=1.8, zorder=3)

    # ToMe frontier line
    tome_pts = sorted([(methods[k]['thr'], methods[k]['acc']) for k in ['Baseline', 'ToMe r=8', 'ToMe r=10', 'ToMe r=12']])
    ax.plot([p[0] for p in tome_pts], [p[1] for p in tome_pts], ':', color='#3498db', alpha=0.4, linewidth=1.5, zorder=2)

    # Clean annotations (no overlapping arrow)
    offsets = {
        'Baseline': (-15, 12),
        'ToMe r=8': (12, 10),
        'ToMe r=10': (-50, -16),
        'ToMe r=12': (-50, -14),
        'S³-Merge l8': (12, 8),
        'S³-Merge l6': (12, -12),
    }
    aligns = {
        'Baseline': 'right',
        'ToMe r=8': 'left',
        'ToMe r=10': 'right',
        'ToMe r=12': 'right',
        'S³-Merge l8': 'left',
        'S³-Merge l6': 'left',
    }

    for name, d in methods.items():
        ox, oy = offsets[name]
        ax.annotate(name, (d['thr'], d['acc']), xytext=(ox, oy), textcoords='offset points',
                    fontsize=8.5, ha=aligns[name], va='center',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5) if abs(ox) > 20 else None)

    ax.set_xlabel('Throughput @ $B{=}64$ (img/s)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_xlim(2550, 3650)
    ax.set_ylim(78.3, 80.0)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc', ncol=2)

    # Subtle "better" arrow in corner (non-overlapping)
    ax.annotate('', xy=(3580, 79.95), xytext=(3510, 79.7),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.2))
    ax.text(3545, 79.97, 'Better', fontsize=7, color='#999', ha='center', va='bottom')

    plt.tight_layout()

    out = Path('documents/figures')
    out.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'pareto_frontier_fixed.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved pareto_frontier_fixed.pdf/.png")
    plt.close()


if __name__ == '__main__':
    main()
