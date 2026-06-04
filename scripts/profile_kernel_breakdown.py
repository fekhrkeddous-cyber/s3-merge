#!/usr/bin/env python3
"""
CUDA kernel-breakdown profiler for S^3-Merge vs ToMe vs baseline.

Reproduces the kernel-launch and category-time breakdown reported in
Section 6.7 / Figure 6 of the paper: at B=1 the baseline DeiT-S issues
roughly the same kernel-launch count as S^3-Merge (which adds ~10 merge
launches), whereas ToMe r=12 issues ~168 scatter/gather launches across
its 12 per-layer matching steps.

Outputs:
  - CSV: artifacts/kernel_profiling.csv (one row per Method x BatchSize x Category)
  - Optional stacked bar chart (--figure_out FILE)

Usage:
  python scripts/profile_kernel_breakdown.py
  python scripts/profile_kernel_breakdown.py --batch_sizes 1 32 --output_dir artifacts

ToMe is enabled via the optional `tome` pip package (see baselines/README.md).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import timm
from torch.profiler import profile, ProfilerActivity

# Repository root is one level above scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.sr_tome import apply_patch as srtome_patch


def classify_kernel(name: str) -> str:
    """Classify a CUDA kernel name into a semantic category."""
    name_lower = name.lower()
    if any(k in name_lower for k in [
        "sdpa", "flash", "attention", "softmax", "fmha",
        "scaled_dot", "attn", "cutlass",
    ]):
        return "Attention"
    if any(k in name_lower for k in [
        "gemm", "cublas", "linear", "addmm", "mm_",
        "sgemm", "ampere", "volta",
    ]):
        return "MLP / Linear"
    if any(k in name_lower for k in [
        "layernorm", "layer_norm", "batch_norm", "normalize",
        "vectorized_elementwise",
    ]):
        return "Normalization"
    if any(k in name_lower for k in [
        "scatter", "gather", "reduce", "index", "topk",
        "sort", "unique",
    ]):
        return "Scatter / Merge"
    if any(k in name_lower for k in [
        "cosine", "bmm", "baddbmm",
    ]):
        return "Matching"
    if any(k in name_lower for k in [
        "gelu", "relu", "silu", "elementwise", "pointwise",
        "add_kernel", "mul_kernel", "copy",
    ]):
        return "Elementwise"
    return "Other"


def profile_model(model, x, name, n_warmup=20, n_profile=10):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(n_profile):
                _ = model(x)
                torch.cuda.synchronize()
    category_time = defaultdict(float)
    category_count = defaultdict(int)
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            cat = classify_kernel(evt.key)
            category_time[cat] += evt.cuda_time_total / 1000.0
            category_count[cat] += evt.count
    for cat in category_time:
        category_time[cat] /= n_profile
        category_count[cat] = int(category_count[cat] / n_profile)
    total_time = sum(category_time.values())
    print(f"\n{'='*60}\n  {name}  (total: {total_time:.2f} ms)\n{'='*60}")
    print(f"  {'Category':<20} {'Time (ms)':>10} {'%':>7} {'Launches':>9}")
    print(f"  {'-'*20} {'-'*10} {'-'*7} {'-'*9}")
    results = {}
    for cat, t in sorted(category_time.items(), key=lambda x: -x[1]):
        pct = 100.0 * t / total_time if total_time > 0 else 0.0
        results[cat] = {"time_ms": t, "pct": pct, "launches": category_count[cat]}
        print(f"  {cat:<20} {t:>10.2f} {pct:>6.1f}% {category_count[cat]:>9}")
    return results, total_time


def run_profiling(args):
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "kernel_profiling.csv"
    all_results = {}

    try:
        import tome
        tome_available = True
    except ImportError:
        tome_available = False
        print("Note: `tome` pip package not installed; skipping ToMe comparison "
              "(see baselines/README.md for install instructions).")

    for batch_size in args.batch_sizes:
        x = torch.randn(batch_size, 3, args.img_size, args.img_size, device=device)

        baseline = timm.create_model(args.model, pretrained=True).to(device).eval()
        res, total = profile_model(baseline, x, f"Baseline B={batch_size}",
                                    n_warmup=args.warmup, n_profile=args.runs)
        all_results[f"Baseline_B{batch_size}"] = (res, total)
        del baseline
        torch.cuda.empty_cache()

        s3model = timm.create_model(args.model, pretrained=True).to(device).eval()
        srtome_patch(s3model, target_tokens=args.target_tokens,
                     late_start=args.late_start, window_size=args.window_size,
                     prop_attn=args.prop_attn)
        res, total = profile_model(s3model, x, f"S3Merge B={batch_size}",
                                    n_warmup=args.warmup, n_profile=args.runs)
        all_results[f"S3Merge_B{batch_size}"] = (res, total)
        del s3model
        torch.cuda.empty_cache()

        if tome_available:
            tome_model = timm.create_model(args.model, pretrained=True).to(device).eval()
            tome.patch.timm(tome_model)
            tome_model.r = args.tome_r
            res, total = profile_model(tome_model, x, f"ToMe B={batch_size}",
                                        n_warmup=args.warmup, n_profile=args.runs)
            all_results[f"ToMe_B{batch_size}"] = (res, total)
            del tome_model
            torch.cuda.empty_cache()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "BatchSize", "Category", "Time_ms", "Pct", "Launches"])
        for key, (res, _total) in all_results.items():
            method, bs = key.rsplit("_B", 1)
            for cat, v in res.items():
                writer.writerow([method, bs, cat, f"{v['time_ms']:.3f}",
                                 f"{v['pct']:.1f}", v["launches"]])
    print(f"\nSaved: {csv_path}")
    return all_results


def generate_figure(all_results, figure_out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure generation")
        return
    canonical_cats = [
        "Attention", "MLP / Linear", "Matching", "Scatter / Merge",
        "Normalization", "Elementwise", "Other",
    ]
    methods = list(all_results.keys())
    bottom = np.zeros(len(methods))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cat in canonical_cats:
        heights = [all_results[m][0].get(cat, {"pct": 0})["pct"] for m in methods]
        ax.bar(methods, heights, bottom=bottom, label=cat)
        bottom = bottom + np.asarray(heights)
    ax.set_ylabel("GPU time (% of total)")
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_out, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {figure_out}")


def main():
    parser = argparse.ArgumentParser(description="CUDA kernel-breakdown profiler "
                                                 "for S^3-Merge / ToMe / baseline")
    parser.add_argument("--model", default="deit_small_patch16_224")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--target_tokens", type=int, default=84)
    parser.add_argument("--late_start", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--prop_attn", action="store_true", default=True)
    parser.add_argument("--no_prop_attn", dest="prop_attn", action="store_false")
    parser.add_argument("--tome_r", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--figure_out", default="")
    args = parser.parse_args()

    all_results = run_profiling(args)
    if args.figure_out:
        generate_figure(all_results, args.figure_out)


if __name__ == "__main__":
    main()
