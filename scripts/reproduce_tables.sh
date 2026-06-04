#!/usr/bin/env bash
# =============================================================================
# S³-Merge: Reproduce All Paper Results
# =============================================================================
#
# Prerequisites:
#   - Docker container running (see README.md)
#   - ImageNet-1K val set at /data/imagenet (or adjust IMAGENET_DIR below)
#   - For downstream: ADE20K/COCO datasets + mmseg/detectron2 checkpoints
#
# Usage:
#   bash scripts/reproduce_tables.sh [table_number]
#   bash scripts/reproduce_tables.sh           # Run all
#   bash scripts/reproduce_tables.sh 1         # Table 1 only
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

IMAGENET_DIR="${IMAGENET_DIR:-/data/imagenet}"

run_table() {
    local table="$1"
    echo ""
    echo "================================================================"
    echo "  Reproducing Table $table"
    echo "================================================================"
    echo ""

    case "$table" in
        2)
            echo "Table 2: Main ImageNet-1K comparison (DeiT-S, 50k val, 5 seeds)"
            echo "Expected runtime: ~45 min on H100"
            python eval_imagenet_baselines.py --mode full
            ;;
        3)
            echo "Table 3: Throughput scaling across batch sizes (B=1..64)"
            python benchmark_batch_scaling.py
            ;;
        4)
            echo "Table 4: Overhead decomposition"
            python profile_overhead.py
            ;;
        7)
            echo "Table 7: Factorised ablation"
            python run_ablation_experiments.py
            ;;
        8)
            echo "Table 8: ADE20K UPerNet-ViT-B backbone throughput (requires mmseg)"
            python eval_ade20k_upernet.py
            ;;
        9)
            echo "Table 9: COCO ViTDet-B end-to-end evaluation (requires detectron2)"
            python eval_coco_vitdet.py
            ;;
        fig4)
            echo "Figure 4: Accuracy-throughput Pareto frontier"
            python generate_pareto_fixed.py
            ;;
        fig5)
            echo "Figure 5: Latency/throughput scaling vs batch size"
            python generate_batch_scaling_pa.py
            ;;
        fig6)
            echo "Figure 6: CUDA kernel time breakdown (168 launches vs 10)"
            python scripts/profile_kernel_breakdown.py --batch_sizes 1 32 --figure_out artifacts/kernel_profiling.pdf
            ;;
        fig7)
            echo "Figure 7: Amdahl decomposition"
            python plot_amdahl_decomposition.py
            ;;
        sdpa)
            echo "Supplementary Table S13: Controlled SDPA backend experiment"
            python benchmark_sdpa_control.py
            ;;
        cross_hw)
            echo "Supplementary Tables S5 / S14: Cross-hardware validation"
            python benchmark_cross_hardware.py
            ;;
        *)
            echo "Unknown table: $table"
            echo "Valid options: 2, 3, 4, 7, 8, 9, fig4, fig5, fig6, fig7, sdpa, cross_hw"
            exit 1
            ;;
    esac
}

if [ $# -eq 0 ]; then
    echo "Running ALL reproducibility experiments..."
    for t in 2 3 4 7 fig4 fig5 fig6 fig7; do
        run_table "$t"
    done
    echo ""
    echo "=========================================="
    echo "  Skipped: Tables 8 / 9 (require mmseg / detectron2 setup)"
    echo "  Skipped: cross_hw (requires L40S / Jetson Orin)"
    echo "  Run individually if needed."
    echo "=========================================="
else
    run_table "$1"
fi

echo ""
echo "Done! Results saved to stdout / figures/ directory."
