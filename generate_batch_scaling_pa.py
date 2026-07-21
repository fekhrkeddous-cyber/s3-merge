#!/usr/bin/env python3
"""
Historical batch-scaling figure reproducer (PA-campaign data embedded below).
Dual-panel: (a) Latency vs Batch Size, (b) Throughput vs Batch Size.
The published Figure 6 uses the unified FP32 campaign (B in [1, 64]) instead;
see EXPERIMENTS.md.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Measured data — PA variants (from Table 3 in paper)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    # Baseline
    baseline_lat = [2.40, 2.35, 2.71, 3.98, 6.38, 11.94, 23.88]
    baseline_thr = [416, 851, 1476, 2006, 2504, 2680, 2680]

    # S³-Merge-Acc (l8+PA)
    acc_lat = [2.92, 2.89, 2.81, 3.69, 5.86, 10.67, 20.81]
    acc_thr = [342, 692, 1421, 2168, 2730, 2998, 3076]

    # S³-Merge-Speed (l6+PA)
    spd_lat = [3.03, 2.97, 2.93, 3.40, 5.26, 9.47, 18.38]
    spd_thr = [330, 673, 1367, 2345, 3043, 3377, 3478]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 8.5,
        'figure.dpi': 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Colors
    c_base = '#2c3e50'
    c_acc = '#e74c3c'
    c_spd = '#27ae60'
    lw = 2.0
    ms = 6

    x_labels = ['1', '2', '4', '8', '16', '32', '64']
    x_pos = np.arange(len(batch_sizes))

    # ===== (a) Latency =====
    ax1.plot(x_pos, baseline_lat, '-o', color=c_base, lw=lw, ms=ms, label='Baseline', zorder=4)
    ax1.plot(x_pos, acc_lat, '--s', color=c_acc, lw=lw, ms=ms, label='S³-Merge-Acc (l8+PA)', zorder=4)
    ax1.plot(x_pos, spd_lat, '-^', color=c_spd, lw=lw, ms=ms, label='S³-Merge-Speed (l6+PA)', zorder=4)

    # Shaded regions — crossover at B*=8 (index 3)
    crossover_x = 2.5  # between B=4 (idx 2) and B=8 (idx 3)
    ax1.axvspan(-0.5, crossover_x, alpha=0.06, color='red', zorder=0)
    ax1.axvspan(crossover_x, 6.5, alpha=0.06, color='green', zorder=0)

    # B* annotation
    ax1.axvline(x=3, color='#e74c3c', linestyle=':', alpha=0.4, linewidth=1.5, zorder=1)
    ax1.text(3.0, max(baseline_lat)*0.58, '$B^*{=}8$', fontsize=8.5, color='#c0392b',
             ha='center', fontweight='bold', alpha=0.8)

    # Region labels
    ax1.text(1.0, max(baseline_lat)*0.65, 'Safe Mode:\nBaseline', fontsize=7.5,
             color='#c0392b', ha='center', style='italic', alpha=0.6)
    ax1.text(5.0, max(baseline_lat)*0.65, 'Safe Mode:\nS³-Merge', fontsize=7.5,
             color='#27ae60', ha='center', style='italic', alpha=0.6)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('(a) Latency vs Batch Size', fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='#ccc')
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.set_xlim(-0.3, 6.3)

    # ===== (b) Throughput =====
    ax2.plot(x_pos, baseline_thr, '-o', color=c_base, lw=lw, ms=ms, label='Baseline', zorder=4)
    ax2.plot(x_pos, acc_thr, '--s', color=c_acc, lw=lw, ms=ms, label='S³-Merge-Acc (l8+PA)', zorder=4)
    ax2.plot(x_pos, spd_thr, '-^', color=c_spd, lw=lw, ms=ms, label='S³-Merge-Speed (l6+PA)', zorder=4)

    # Shaded regions
    ax2.axvspan(-0.5, crossover_x, alpha=0.06, color='red', zorder=0)
    ax2.axvspan(crossover_x, 6.5, alpha=0.06, color='green', zorder=0)

    # Crossover annotation
    ax2.axvline(x=3, color='#e74c3c', linestyle=':', alpha=0.4, linewidth=1.5, zorder=1)
    ax2.annotate('$B^*{=}8$\n(crossover)', xy=(3, 2100), fontsize=8,
                 ha='center', va='center', color='#555',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#aaa', alpha=0.9))

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (img/s)')
    ax2.set_title('(b) Throughput vs Batch Size', fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='#ccc')
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_xlim(-0.3, 6.3)

    plt.tight_layout(w_pad=2.0)

    # Save to tmp (overleaf files may be root-owned)
    out = Path('/tmp/figures')
    out.mkdir(parents=True, exist_ok=True)

    for ext in ['pdf', 'png']:
        fig.savefig(out / f'batch_scaling_b32.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    print(f"Saved to {out}/batch_scaling_b32.pdf/.png")
    print("Copy to overleaf_upload/figures/ with: sudo cp /tmp/figures/batch_scaling_b32.* overleaf_upload/figures/")
    plt.close()


if __name__ == '__main__':
    main()
