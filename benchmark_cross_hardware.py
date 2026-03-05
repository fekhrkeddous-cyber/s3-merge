#!/usr/bin/env python3
"""
=============================================================================
S³-Merge Cross-Hardware Benchmark — L40S Validation
=============================================================================
FULLY SELF-CONTAINED. No project imports needed.
Requirements: torch, timm, numpy  (pip install torch timm numpy)

This script measures:
  1. Baseline DeiT-S latency/throughput at B=1,4,8,16,32
  2. ToMe r=10 (no PA) latency/throughput at same batch sizes
  3. S³-Merge l8 (no PA) latency/throughput at same batch sizes  
  4. Fits analytic B* model: T(B) = α + β·B
  5. Predicts B* and compares to measured crossover

OUTPUT FORMAT: structured text that can be directly copy-pasted back.
=============================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np
import timm
import time
import sys

# ============================================================================
# INLINE S³-MERGE (no project imports needed)
# ============================================================================

def sr_tome_merge_inline(x, target_tokens=84, window_size=7, grid_size=14):
    """Spatially-restricted single-shot merge. Self-contained implementation.
    
    Args:
        x: [B, 1+N, C] input tokens (CLS + patches)
        target_tokens: number of tokens to keep (excluding CLS)
        window_size: spatial window for matching
        grid_size: grid dimension (sqrt of N)
    
    Returns:
        merged_x: [B, 1+target_tokens, C]
    """
    B, N_plus1, C = x.shape
    cls = x[:, :1]       # [B, 1, C]
    patches = x[:, 1:]   # [B, N, C]
    N = patches.shape[1]
    H = W = grid_size
    
    assert H * W == N, f"Grid {H}x{W} != {N} patches"
    
    # Reshape to spatial grid
    patches_2d = patches.view(B, H, W, C)
    
    # Process each window
    all_kept = []
    w = window_size
    n_windows_h = H // w
    n_windows_w = W // w
    
    # Per-window target count
    total_patches = H * W
    patches_per_window = w * w
    n_windows = n_windows_h * n_windows_w
    keep_per_window = target_tokens // n_windows
    merge_per_window = patches_per_window - keep_per_window
    
    for wi in range(n_windows_h):
        for wj in range(n_windows_w):
            # Extract window
            row_start = wi * w
            col_start = wj * w
            window = patches_2d[:, row_start:row_start+w, col_start:col_start+w, :]
            window_flat = window.reshape(B, patches_per_window, C)  # [B, w*w, C]
            
            # Random target selection (select keep_per_window targets)
            target_idx = torch.randperm(patches_per_window, device=x.device)[:keep_per_window]
            source_mask = torch.ones(patches_per_window, dtype=torch.bool, device=x.device)
            source_mask[target_idx] = False
            source_idx = torch.where(source_mask)[0]
            
            targets = window_flat[:, target_idx]   # [B, K, C]
            sources = window_flat[:, source_idx]   # [B, M, C]
            
            # Cosine similarity matching
            t_norm = F.normalize(targets, dim=-1)
            s_norm = F.normalize(sources, dim=-1)
            sim = torch.bmm(s_norm, t_norm.transpose(1, 2))  # [B, M, K]
            
            # Each source merges into its most similar target
            assignments = sim.argmax(dim=-1)  # [B, M]
            
            # Weighted merge via scatter_add
            merged = targets.clone()
            target_counts = torch.ones(B, keep_per_window, device=x.device)
            
            for b in range(B):
                for s in range(sources.shape[1]):
                    t = assignments[b, s].item()
                    merged[b, t] += sources[b, s]
                    target_counts[b, t] += 1
            
            # Average
            merged = merged / target_counts.unsqueeze(-1)
            all_kept.append(merged)
    
    # Reassemble
    merged_patches = torch.cat(all_kept, dim=1)  # [B, target_tokens, C]
    return torch.cat([cls, merged_patches], dim=1)


def sr_tome_merge_fast(x, target_tokens=84, window_size=7, grid_size=14):
    """Vectorized S³-Merge (faster for benchmarking)."""
    B, N1, C = x.shape
    cls = x[:, :1]
    patches = x[:, 1:]
    N = patches.shape[1]
    H = W = grid_size
    
    patches_2d = patches.view(B, H, W, C)
    w = window_size
    nwh = H // w
    nww = W // w
    n_windows = nwh * nww
    pw = w * w
    kpw = target_tokens // n_windows
    
    # Unfold into windows: [B, nwh, nww, w, w, C] -> [B, n_windows, pw, C]
    windows = patches_2d.unfold(1, w, w).unfold(2, w, w)  # [B, nwh, nww, C, w, w]
    windows = windows.permute(0, 1, 2, 4, 5, 3).contiguous()  # [B, nwh, nww, w, w, C]
    windows = windows.view(B, n_windows, pw, C)
    
    # Random targets (same for all batch items for speed)
    target_idx = torch.randperm(pw, device=x.device)[:kpw]
    source_mask = torch.ones(pw, dtype=torch.bool, device=x.device)
    source_mask[target_idx] = False
    source_idx = torch.where(source_mask)[0]
    
    targets = windows[:, :, target_idx]  # [B, nw, kpw, C]
    sources = windows[:, :, source_idx]  # [B, nw, mpw, C]
    
    # Reshape for batch matmul
    Bw = B * n_windows
    t_flat = targets.reshape(Bw, kpw, C)
    s_flat = sources.reshape(Bw, sources.shape[2], C)
    
    t_norm = F.normalize(t_flat, dim=-1)
    s_norm = F.normalize(s_flat, dim=-1)
    sim = torch.bmm(s_norm, t_norm.transpose(1, 2))
    assignments = sim.argmax(dim=-1)  # [Bw, mpw]
    
    # Scatter merge
    merged = t_flat.clone()
    counts = torch.ones(Bw, kpw, 1, device=x.device)
    
    src_expanded = s_flat  # [Bw, mpw, C]
    idx_expanded = assignments.unsqueeze(-1).expand(-1, -1, C)  # [Bw, mpw, C]
    merged.scatter_add_(1, idx_expanded, src_expanded)
    
    ones = torch.ones(Bw, sources.shape[2], 1, device=x.device)
    counts.scatter_add_(1, assignments.unsqueeze(-1), ones)
    
    merged = merged / counts
    merged = merged.view(B, n_windows * kpw, C)
    
    return torch.cat([cls, merged], dim=1)


# ============================================================================
# INLINE ToMe (simplified, no PA)
# ============================================================================

def tome_merge_step(x, r=10):
    """ToMe-style bipartite merge per layer. First-half/second-half split."""
    B, N, C = x.shape
    if N <= 2 or r <= 0:
        return x
    
    cls = x[:, :1]
    patches = x[:, 1:]
    P = patches.shape[1]
    r = min(r, P // 2)
    if r <= 0:
        return x
    
    half = P // 2
    a = patches[:, :half]
    b = patches[:, half:2*half]
    nA = nB = half
    
    an = F.normalize(a, dim=-1)
    bn = F.normalize(b, dim=-1)
    sim = torch.bmm(an, bn.transpose(1, 2))
    
    max_sim, b_assign = sim.max(dim=-1)
    _, top_a = max_sim.topk(r, dim=1)
    top_b = torch.gather(b_assign, 1, top_a)
    
    a_tok = torch.gather(a, 1, top_a.unsqueeze(-1).expand(-1, -1, C))
    b_tok = torch.gather(b, 1, top_b.unsqueeze(-1).expand(-1, -1, C))
    merged = (a_tok + b_tok) / 2
    
    a_mask = torch.ones(B, nA, dtype=torch.bool, device=x.device)
    a_mask.scatter_(1, top_a, False)
    kept_a = a[a_mask].view(B, nA - r, C)
    kept_b = b[:, r:]
    extra = patches[:, 2*half:]
    
    return torch.cat([cls, kept_a, kept_b, extra, merged], dim=1)


# ============================================================================
# PATCHING FUNCTIONS
# ============================================================================

def patch_s3merge(model, merge_layer=8, target_tokens=84):
    """Patch DeiT-S with single-shot S³-Merge at layer merge_layer."""
    def forward(x, **kwargs):
        x = model.patch_embed(x)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        for i, blk in enumerate(model.blocks):
            if i == merge_layer:
                x = sr_tome_merge_fast(x, target_tokens, window_size=7, grid_size=14)
            x = blk(x)
        x = model.norm(x)
        return model.head(x[:, 0])
    model.forward = forward
    return model


def patch_tome(model, r=10):
    """Patch DeiT-S with ToMe (merge r tokens per layer, no PA)."""
    def forward(x, **kwargs):
        x = model.patch_embed(x)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        for blk in model.blocks:
            x = tome_merge_step(x, r)
            x = blk(x)
        x = model.norm(x)
        return model.head(x[:, 0])
    model.forward = forward
    return model


# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def measure_latency(model, batch_size, warmup=100, runs=500, device='cuda'):
    """Measure median latency in ms."""
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            model(x)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    
    return np.median(times)


def fit_bstar(base_b1, base_b32, merge_b1, merge_b32):
    """Fit T(B) = α + β·B and predict B*."""
    beta_base = (base_b32 - base_b1) / 31
    alpha_base = base_b1 - beta_base
    
    beta_merge = (merge_b32 - merge_b1) / 31
    alpha_merge = merge_b1 - beta_merge
    
    if beta_base <= beta_merge:
        bstar = float('inf')
    else:
        bstar = (alpha_merge - alpha_base) / (beta_base - beta_merge)
    
    return alpha_base, beta_base, alpha_merge, beta_merge, bstar


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = 'cuda'
    
    print("=" * 70)
    print("S³-MERGE CROSS-HARDWARE BENCHMARK")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version:    {torch.version.cuda}")
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Check Flash/SDPA availability
    try:
        flash_available = torch.backends.cuda.flash_sdp_enabled()
        print(f"Flash SDP:       {flash_available}")
    except:
        print("Flash SDP:       N/A (older PyTorch)")
    
    print(f"Timestamp:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create models
    def mk_base():
        return timm.create_model('deit_small_patch16_224', pretrained=False)
    def mk_tome():
        return patch_tome(mk_base(), r=10)
    def mk_s3():
        return patch_s3merge(mk_base(), merge_layer=8, target_tokens=84)
    
    methods = [
        ("Baseline", mk_base),
        ("ToMe_r10", mk_tome),
        ("S3Merge_l8", mk_s3),
    ]
    
    batch_sizes = [1, 4, 8, 16, 32]
    
    # ========================================
    print("=" * 70)
    print("LATENCY AND THROUGHPUT MEASUREMENTS")
    print("=" * 70)
    
    results = {}
    for name, mk_fn in methods:
        row = {}
        for B in batch_sizes:
            model = mk_fn().to(device).eval()
            lat = measure_latency(model, B, device=device)
            thr = B / (lat / 1000)
            row[B] = (lat, thr)
            del model
            torch.cuda.empty_cache()
            print(f"  {name} B={B}: {lat:.3f}ms ({thr:.0f} img/s)")
        results[name] = row
        print()
    
    # ========================================
    # STRUCTURED OUTPUT (copy-paste this back)
    # ========================================
    print()
    print("=" * 70)
    print("STRUCTURED RESULTS — COPY THIS BLOCK")
    print("=" * 70)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print()
    
    # Table 1: Raw latency
    header = f"{'Method':<15}"
    for B in batch_sizes:
        header += f" | B={B:>2} (ms)"
    print(header)
    print("-" * len(header))
    
    for name in ["Baseline", "ToMe_r10", "S3Merge_l8"]:
        row = f"{name:<15}"
        for B in batch_sizes:
            lat = results[name][B][0]
            row += f" | {lat:>8.3f}"
        print(row)
    
    print()
    
    # Table 2: Throughput
    header2 = f"{'Method':<15}"
    for B in batch_sizes:
        header2 += f" | B={B:>2} img/s"
    print(header2)
    print("-" * len(header2))
    
    for name in ["Baseline", "ToMe_r10", "S3Merge_l8"]:
        row = f"{name:<15}"
        for B in batch_sizes:
            thr = results[name][B][1]
            row += f" | {thr:>9.0f}"
        print(row)
    
    print()
    
    # Table 3: Speedup relative to baseline
    print("Speedup vs Baseline:")
    for name in ["ToMe_r10", "S3Merge_l8"]:
        row = f"  {name:<15}"
        for B in batch_sizes:
            base_lat = results["Baseline"][B][0]
            this_lat = results[name][B][0]
            speedup = base_lat / this_lat
            row += f" | {speedup:>5.2f}x"
        print(row)
    
    print()
    
    # ========================================
    # ANALYTIC B* MODEL
    # ========================================
    print("=" * 70)
    print("ANALYTIC B* MODEL")
    print("=" * 70)
    
    base_b1 = results["Baseline"][1][0]
    base_b32_lat = results["Baseline"][32][0]
    s3_b1 = results["S3Merge_l8"][1][0]
    s3_b32_lat = results["S3Merge_l8"][32][0]
    
    alpha_b, beta_b, alpha_m, beta_m, bstar = fit_bstar(
        base_b1, base_b32_lat, s3_b1, s3_b32_lat
    )
    
    print(f"Baseline:  α = {alpha_b:.3f}ms, β = {beta_b:.4f} ms/img")
    print(f"S3Merge:   α = {alpha_m:.3f}ms, β = {beta_m:.4f} ms/img")
    print(f"Δα (overhead) = {alpha_m - alpha_b:.3f}ms")
    print(f"Δβ (savings)  = {beta_b - beta_m:.4f} ms/img")
    print(f"Predicted B*  = {bstar:.1f}")
    print()
    
    # Find measured crossover
    measured_bstar = None
    for B in batch_sizes:
        base_thr = results["Baseline"][B][1]
        s3_thr = results["S3Merge_l8"][B][1]
        if s3_thr > base_thr:
            measured_bstar = B
            break
    
    print(f"Measured crossover B* = {measured_bstar}")
    print(f"Model prediction error = {abs(bstar - (measured_bstar or 0)):.1f} batch sizes")
    
    # Validate at each batch size
    print()
    print("Validation:")
    print(f"{'B':>4} | {'Pred base':>10} | {'Meas base':>10} | {'Pred S3':>10} | {'Meas S3':>10} | {'Pred':>6} | {'Meas':>6}")
    print("-" * 75)
    for B in batch_sizes:
        pred_base = alpha_b + beta_b * B
        pred_s3 = alpha_m + beta_m * B
        meas_base = results["Baseline"][B][0]
        meas_s3 = results["S3Merge_l8"][B][0]
        pred_winner = "S3" if pred_s3 < pred_base else "Base"
        meas_winner = "S3" if meas_s3 < meas_base else "Base"
        check = "✓" if pred_winner == meas_winner else "✗"
        print(f"{B:>4} | {pred_base:>8.3f}ms | {meas_base:>8.3f}ms | {pred_s3:>8.3f}ms | {meas_s3:>8.3f}ms | {pred_winner:>6} | {meas_winner:>6} {check}")
    
    print()
    print("=" * 70)
    print("END OF BENCHMARK")
    print("=" * 70)


if __name__ == '__main__':
    main()
