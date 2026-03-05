#!/usr/bin/env python3
"""
Amdahl's Law Stacked Bar Chart for S³-Merge Paper
===================================================
Horizontal stacked bars showing ViTDet-B latency decomposition.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Measured data (B=8, 1024², ViTDet-B / Mask R-CNN, H100 NVL)
baseline_vit = 141.3
merge_vit = 141.3 / 1.32  # 107.0 ms
fpn = 16.7
heads = 40.2

baseline_total = baseline_vit + fpn + heads  # 198.2
merge_total = merge_vit + fpn + heads        # 163.9

# Colors
c_vit = '#2196F3'
c_fpn = '#FF9800'
c_heads = '#E53935'

fig, ax = plt.subplots(figsize=(5.5, 1.8))

bars = ['Baseline', 'S³-Merge']
y = [0, 1]
bh = 0.5

# Draw bars
for i, (vit_val, total) in enumerate([(baseline_vit, baseline_total), (merge_vit, merge_total)]):
    ax.barh(y[i], vit_val, bh, left=0, color=c_vit, edgecolor='white', lw=0.5)
    ax.barh(y[i], fpn, bh, left=vit_val, color=c_fpn, edgecolor='white', lw=0.5)
    ax.barh(y[i], heads, bh, left=vit_val + fpn, color=c_heads, edgecolor='white', lw=0.5)
    
    # Labels inside bars
    ax.text(vit_val/2, y[i], f'{vit_val:.0f} ms\n({vit_val/total*100:.0f}%)',
           ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    if fpn > 12:
        ax.text(vit_val + fpn/2, y[i], f'{fpn:.0f}',
               ha='center', va='center', fontsize=6, color='white')
    ax.text(vit_val + fpn + heads/2, y[i], f'{heads:.0f} ms\n({heads/total*100:.0f}%)',
           ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # Total
    ax.text(total + 2, y[i], f'{total:.0f} ms',
           ha='left', va='center', fontsize=7.5, color='#555555', fontweight='bold')

# Backbone speedup bracket
savings = baseline_vit - merge_vit
mid_x = (merge_vit + baseline_vit) / 2

# Draw bracket with vertical lines
bracket_y = 1.65
ax.plot([merge_vit, merge_vit], [1.28, bracket_y], color=c_vit, lw=1, alpha=0.7)
ax.plot([baseline_vit, baseline_vit], [0.28, bracket_y], color=c_vit, lw=1, alpha=0.7)
ax.plot([merge_vit, baseline_vit], [bracket_y, bracket_y], color=c_vit, lw=1.5)
ax.text(mid_x, bracket_y + 0.15, f'1.32× backbone speedup (−{savings:.0f} ms)',
       ha='center', va='bottom', fontsize=7, color='#1565C0', fontweight='bold')

# E2E speedup
e2e = baseline_total / merge_total
ax.text(merge_total + 2, 1.25, f'→ E2E: {e2e:.2f}× (Amdahl limit)',
       ha='left', va='center', fontsize=6.5, color='#888888', fontstyle='italic')

# Fixed-cost label
ax.annotate('Fixed cost\n(decoder)', xy=(baseline_vit + fpn + heads/2, -0.35),
           fontsize=6, ha='center', va='top', color='#999999',
           arrowprops=dict(arrowstyle='->', color='#cccccc', lw=0.8),
           xytext=(baseline_vit + fpn + heads/2, -0.55))

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=c_vit, label='ViT Backbone'),
    Patch(facecolor=c_fpn, label='FPN'),
    Patch(facecolor=c_heads, label='RPN + ROI Heads'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=6.5,
         framealpha=0.9, edgecolor='#dddddd', ncol=3,
         bbox_to_anchor=(1.0, -0.45))

# Formatting
ax.set_yticks(y)
ax.set_yticklabels(bars, fontsize=8.5, fontweight='bold')
ax.set_xlabel('Latency (ms)', fontsize=8)
ax.set_xlim(0, 220)
ax.set_ylim(-0.7, 2.3)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=7)

plt.tight_layout()
plt.savefig('figures/amdahl_decomposition.pdf',
           bbox_inches='tight', dpi=300, pad_inches=0.03)
plt.savefig('figures/amdahl_decomposition.png',
           bbox_inches='tight', dpi=300, pad_inches=0.03)
print("Saved amdahl_decomposition.pdf and .png")
