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
        1)
            echo "Table 1: Main ImageNet-1K comparison (DeiT-S, 50k val, 5 seeds)"
            echo "Expected runtime: ~45 min on H100"
            python eval_imagenet_baselines.py --mode full
            ;;
        2)
            echo "Table 2: Batch-size scaling (B=1..64)"
            python benchmark_batch_scaling.py
            ;;
        3)
            echo "Table 3: ADE20K UPerNet-ViT-B (requires mmseg)"
            python eval_ade20k_upernet.py
            ;;
        4)
            echo "Table 4: Overhead decomposition"
            python profile_overhead.py
            ;;
        5)
            echo "Table 5: Ablation studies"
            python run_ablation_experiments.py
            ;;
        fig3)
            echo "Figure 3: Pareto frontier"
            python generate_pareto_fixed.py
            ;;
        fig4)
            echo "Figure 4: Batch scaling plot"
            python generate_batch_scaling_pa.py
            ;;
        fig5)
            echo "Figure 5: Amdahl decomposition"
            python plot_amdahl_decomposition.py
            ;;
        sdpa)
            echo "Supplementary: Controlled SDPA backend experiment"
            python benchmark_sdpa_control.py
            ;;
        cross_hw)
            echo "Supplementary: Cross-hardware validation"
            python benchmark_cross_hardware.py
            ;;
        *)
            echo "Unknown table: $table"
            echo "Valid options: 1, 2, 3, 4, 5, fig3, fig4, fig5, sdpa, cross_hw"
            exit 1
            ;;
    esac
}

if [ $# -eq 0 ]; then
    echo "Running ALL reproducibility experiments..."
    for t in 1 2 4 5 fig3 fig4; do
        run_table "$t"
    done
    echo ""
    echo "=========================================="
    echo "  Skipped: Table 3 (requires mmseg setup)"
    echo "  Skipped: cross_hw (requires L40S/Orin)"
    echo "  Run individually if needed."
    echo "=========================================="
else
    run_table "$1"
fi

echo ""
echo "Done! Results saved to stdout / figures/ directory."
