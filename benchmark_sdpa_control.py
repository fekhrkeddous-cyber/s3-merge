#!/usr/bin/env python3
"""
W4: Controlled SDPA Backend Experiment (Simplified)
====================================================
Tests Baseline vs S³-Merge (no PA) under default and forced mem-efficient backends.
The no-PA variant uses identical SDPA code paths as baseline, isolating the
merging contribution from any kernel-path artifacts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import timm

from models.sr_tome import sr_tome_merge_exact


def patch_s3merge(model, merge_layer=8, target_tokens=84,
                  window_size=7, grid_size=14):
    """Patch model with S³-Merge (no PA, no attention mask changes)."""
    def s3_forward(x, **kwargs):
        x = model.patch_embed(x)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        
        for i, blk in enumerate(model.blocks):
            if i == merge_layer:
                x, _ = sr_tome_merge_exact(
                    x, target_tokens,
                    window_size=window_size,
                    grid_size=grid_size,
                    weighted_merge=True,
                    tau=0.1,
                    track_sizes=False,
                )
            x = blk(x)
        
        x = model.norm(x)
        return model.head(x[:, 0])
    
    model.forward = s3_forward
    return model


def measure_latency(model, batch_size, input_size=224, warmup=100, runs=500):
    device = next(model.parameters()).device
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return np.median(times)


def run_config(flash_on, label):
    """Run Baseline vs S³-Merge under given SDPA config."""
    device = 'cuda'
    torch.backends.cuda.enable_flash_sdp(flash_on)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(not flash_on)  # fallback if both off
    
    print(f"\n{'='*70}")
    print(f"Backend: {label} (flash_sdp={'ON' if flash_on else 'OFF'})")
    print(f"{'='*70}")
    
    results = {}
    batch_sizes = [1, 8, 32]
    
    for bname, make_model in [
        ("Baseline", lambda: timm.create_model('deit_small_patch16_224', pretrained=False).to(device).eval()),
        ("S3-Merge l8 (no PA)", lambda: patch_s3merge(timm.create_model('deit_small_patch16_224', pretrained=False).to(device).eval())),
    ]:
        row = {}
        for B in batch_sizes:
            model = make_model()
            lat = measure_latency(model, B)
            thr = B / (lat / 1000)
            row[B] = (lat, thr)
            del model; torch.cuda.empty_cache()
        results[bname] = row
    
    # Print
    print(f"\n{'Method':<25} | {'B=1 (ms)':>10} | {'B=8 (ms)':>10} | {'B=32 img/s':>12}")
    print("-" * 65)
    for method, data in results.items():
        b1 = data[1][0]
        b8 = data[8][0]
        b32_thr = data[32][1]
        print(f"{method:<25} | {b1:>8.3f}ms | {b8:>8.3f}ms | {b32_thr:>10.0f}")
    
    # Speedup
    base = results["Baseline"]
    merge = results["S3-Merge l8 (no PA)"]
    print(f"\nSpeedups (Baseline / S³-Merge):")
    for B in batch_sizes:
        speedup = base[B][0] / merge[B][0]
        print(f"  B={B:>2}: {speedup:.3f}× ({'faster' if speedup > 1 else 'SLOWER'})")
    
    return results


def main():
    print("W4: Controlled SDPA Backend Experiment")
    print(f"DeiT-S/16, 224², H100 NVL")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    
    # Config 1: Default (Flash ON — baseline uses FlashAttention-2, S³-Merge no-PA also uses Flash)
    default = run_config(flash_on=True, label="DEFAULT (FlashAttention-2 ON)")
    
    # Config 2: Controlled (Flash OFF — everyone uses mem-efficient attention)
    controlled = run_config(flash_on=False, label="CONTROLLED (FlashAttention-2 OFF)")
    
    # Summary
    print(f"\n{'='*70}")
    print("CONCLUSION: Do speedups persist under controlled backend?")
    print(f"{'='*70}")
    for B in [1, 8, 32]:
        def_speedup = default["Baseline"][B][0] / default["S3-Merge l8 (no PA)"][B][0]
        ctrl_speedup = controlled["Baseline"][B][0] / controlled["S3-Merge l8 (no PA)"][B][0]
        delta = abs(def_speedup - ctrl_speedup)
        print(f"  B={B:>2}: Default={def_speedup:.3f}×  Controlled={ctrl_speedup:.3f}×  "
              f"(Δ={delta:.3f}) {'✓ Consistent' if delta < 0.05 else '⚠ Changed'}")


if __name__ == '__main__':
    main()
