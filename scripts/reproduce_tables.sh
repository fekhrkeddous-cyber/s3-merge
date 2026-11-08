#!/usr/bin/env bash
# Run the paper campaigns whose producers are included in this release (see EXPERIMENTS.md for protocols and outputs).
# Usage: bash scripts/reproduce_tables.sh <item>
#   table1   unified comparison with official ToMe (Table 1, Fig. 5, S20-S22)   needs IMAGENET_SHARDS, TOME_PATH
#   table4   controlled 2x2 factorial study (Table 4, S17-S19)                   needs IMAGENET_SHARDS
#   table7   ViT-B/16 row of Table 7 (PA-off reconstruction)                     needs IMAGENET_CACHE
#   safemode Safe Mode calibration (Section 5)
#   tests    unit tests (CPU) and the bounded GPU execution check
# Measurements are hardware-dependent and are not expected to be bit-identical to the paper.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:-out}"
need() { if [ -z "${!1:-}" ]; then echo "set $1 (see EXPERIMENTS.md)"; exit 2; fi; }
case "${1:-}" in
    table1)
        need IMAGENET_SHARDS; need TOME_PATH
        python r3/unified_tome/accuracy_campaign.py --output-dir "$OUT/table1_accuracy" --data-dir "$IMAGENET_SHARDS" --tome-path "$TOME_PATH"
        python r3/unified_tome/accuracy_analysis.py --runs-dir "$OUT/table1_accuracy/runs" --output-dir "$OUT/table1_accuracy/analysis"
        python r3/unified_tome/timing_campaign.py --output-dir "$OUT/table1_timing" --tome-path "$TOME_PATH" \
            --accuracy-analysis "$OUT/table1_accuracy/analysis/accuracy_analysis.json"
        python r3/unified_tome/gmac.py --output-dir "$OUT/table1_gmac" ;;
    table4)
        need IMAGENET_SHARDS
        python r3/factorial/campaign.py accuracy --output-dir "$OUT/table4/accuracy" -- --data-dir "$IMAGENET_SHARDS" --device cuda
        python r3/factorial/analyze_accuracy.py --runs-dir "$OUT/table4/accuracy" --output-dir "$OUT/table4/analysis"
        python r3/factorial/campaign.py timing --modes latency --output-dir "$OUT/table4/timing" -- --device cuda
        python r3/factorial/analyze_timing.py --raw-dir "$OUT/table4/timing" --output-dir "$OUT/table4/analysis" ;;
    table7)
        need IMAGENET_CACHE
        python r3/table7_vitb16/run_table7_vitb16.py --output-dir "$OUT/table7_vitb16" --imagenet-cache "$IMAGENET_CACHE" ;;
    safemode)
        python -m models.safemode --validate ;;
    tests)
        python tests/test_release.py
        python tests/test_public.py
        python r3/factorial/test_factorial.py
        python tests/smoke_gpu.py ;;
    *)
        sed -n '2,10p' "$0"; exit 1 ;;
esac
