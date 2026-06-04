#!/usr/bin/env python3
"""
Multi-batch-size latency benchmark for S^3-Merge vs baseline.

Reproduces Table 3 / Figure 5 (throughput scaling) under the timing protocol
of the paper's Section 6.1: one-shot tensor placement, 30 warmup forward
passes each followed by torch.cuda.synchronize(), and 200 timed forward
passes wall-clocked with time.perf_counter and synchronised per iteration.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.sr_tome import create_sr_tome_v3


def measure_latency(model, input_tensor, warmup=30, runs=200):
    """Per-iteration synchronised latency measurement (Section 6.1 protocol)."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
    return np.array(latencies)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-size scaling benchmark for Table 3 / Figure 5"
    )
    parser.add_argument('--model', default='deit_small_patch16_224', type=str)
    parser.add_argument('--target_tokens', default=84, type=int,
                        help="Tokens retained by S^3-Merge (84 = paper default).")
    parser.add_argument('--late_start', default=8, type=int,
                        help="Merge layer L_s (8 = paper default).")
    parser.add_argument('--window_size', default=7, type=int)
    parser.add_argument('--prop_attn', action='store_true', default=True,
                        help="Enable proportional attention (paper default).")
    parser.add_argument('--no_prop_attn', dest='prop_attn', action='store_false')
    parser.add_argument('--batch_sizes', default='1,2,4,8,16,32,64', type=str)
    parser.add_argument('--warmup', default=30, type=int)
    parser.add_argument('--runs', default=200, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--output_dir', default='artifacts', type=str)
    args = parser.parse_args()

    device = torch.device("cuda")
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]

    print(f"Model:              {args.model}")
    print(f"Target tokens:      {args.target_tokens} of {(args.img_size // 16)**2}")
    print(f"Merge layer L_s:    {args.late_start}")
    print(f"Window size:        {args.window_size}")
    print(f"Proportional attn:  {args.prop_attn}")
    print(f"Batch sizes:        {batch_sizes}")
    print()
    print(f"{'Batch':<8} {'Baseline (ms)':<15} {'S3-Merge (ms)':<16} {'Speedup':<10}")
    print("-" * 55)

    results = []
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, args.img_size, args.img_size, device=device)

            base = timm.create_model(args.model, pretrained=True).to(device).eval()
            base_lat = measure_latency(base, x, warmup=args.warmup, runs=args.runs)
            del base
            torch.cuda.empty_cache()

            merged = create_sr_tome_v3(
                model_name=args.model,
                target_tokens=args.target_tokens,
                late_start=args.late_start,
                window_size=args.window_size,
                prop_attn=args.prop_attn,
            ).to(device).eval()
            merged_lat = measure_latency(merged, x, warmup=args.warmup, runs=args.runs)
            del merged
            torch.cuda.empty_cache()

            base_med = float(np.median(base_lat))
            merged_med = float(np.median(merged_lat))
            speedup = base_med / merged_med
            print(f"{bs:<8} {base_med:<15.2f} {merged_med:<16.2f} {speedup:<10.2f}x")
            results.append({
                'batch_size': bs,
                'baseline_median_ms': round(base_med, 4),
                's3_merge_median_ms': round(merged_med, 4),
                'speedup': round(speedup, 4),
                'throughput_baseline_ips': round(bs * 1000.0 / base_med, 1),
                'throughput_s3_merge_ips': round(bs * 1000.0 / merged_med, 1),
            })
        except RuntimeError as exc:
            if "out of memory" in str(exc):
                print(f"{bs:<8} OOM")
                torch.cuda.empty_cache()
                break
            raise

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / 'batch_size_scaling.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # First batch size at which S^3-Merge becomes faster than baseline (B*).
    for r in results:
        if r['speedup'] > 1.0:
            print(f"\nCrossover B* = {r['batch_size']}: {r['speedup']:.2f}x speedup")
            break


if __name__ == '__main__':
    main()
