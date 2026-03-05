#!/usr/bin/env python3
"""
Multi-batch-size latency benchmark to find the crossover point.
"""
import argparse
import time
import torch
import numpy as np
import timm
import csv
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.htm import HTMWrapper


def measure_latency(model, input_tensor, warmup=20, runs=50):
    """Measure latency with proper CUDA synchronization."""
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
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    return np.array(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deit_small_patch16_224', type=str)
    parser.add_argument('--merge_rate', default=12, type=int)
    parser.add_argument('--beta', default=2.0, type=float)
    parser.add_argument('--batch_sizes', default='1,2,4,8,16,32,64,128', type=str)
    parser.add_argument('--output_dir', default='artifacts', type=str)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]
    
    print(f"Model: {args.model}")
    print(f"Merge rate: {args.merge_rate}")
    print(f"Batch sizes: {batch_sizes}")
    print()
    
    # Load models once
    base_model = timm.create_model(args.model, pretrained=True).to(device).eval()
    
    num_layers = len(base_model.blocks)
    merge_schedule = [0] + [args.merge_rate] * (num_layers - 1)
    htm_model = HTMWrapper(base_model, merge_schedule=merge_schedule, beta=args.beta).to(device).eval()
    
    results = []
    
    print(f"{'Batch':<8} {'Baseline (ms)':<15} {'HTM (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for bs in batch_sizes:
        try:
            input_tensor = torch.randn(bs, 3, 224, 224, device=device)
            
            # Measure baseline
            base_model_fresh = timm.create_model(args.model, pretrained=True).to(device).eval()
            base_lat = measure_latency(base_model_fresh, input_tensor)
            del base_model_fresh
            torch.cuda.empty_cache()
            
            # Measure HTM
            base_model_for_htm = timm.create_model(args.model, pretrained=True).to(device).eval()
            htm_m = HTMWrapper(base_model_for_htm, merge_schedule=merge_schedule, beta=args.beta).to(device).eval()
            htm_lat = measure_latency(htm_m, input_tensor)
            del htm_m, base_model_for_htm
            torch.cuda.empty_cache()
            
            base_med = np.median(base_lat)
            htm_med = np.median(htm_lat)
            speedup = base_med / htm_med
            
            print(f"{bs:<8} {base_med:<15.2f} {htm_med:<15.2f} {speedup:<10.2f}x")
            
            results.append({
                'batch_size': bs,
                'baseline_median_ms': base_med,
                'htm_median_ms': htm_med,
                'speedup': speedup,
                'throughput_baseline': bs * 1000 / base_med,
                'throughput_htm': bs * 1000 / htm_med,
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{bs:<8} OOM")
                torch.cuda.empty_cache()
                break
            raise
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'batch_size_scaling.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")
    
    # Find crossover point
    for r in results:
        if r['speedup'] > 1.0:
            print(f"\n✓ Crossover at batch size {r['batch_size']}: {r['speedup']:.2f}x speedup")
            break


if __name__ == '__main__':
    main()
