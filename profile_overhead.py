#!/usr/bin/env python3
"""
Overhead breakdown profiler for S³-Merge vs ToMe vs Baseline.

Profiles the DeiT-S forward pass and breaks down time into:
  (i)   Target selection (random score generation + topK)
  (ii)  Similarity computation (cosine similarity matrix)
  (iii) Merge reduction (scatter_reduce / averaging)
  (iv)  Transformer compute (all transformer blocks)
  (v)   Unmerge / restore (for downstream mode)

Outputs a CSV and prints a LaTeX-ready table.

Usage:
  python profile_overhead.py [--batch_size 8] [--n_iter 200] [--warmup 30]
"""
import argparse
import time
import torch
import torch.nn.functional as F
import timm
import csv
import os
from pathlib import Path

# ─── helpers ───────────────────────────────────────────────────────────────────

def cuda_timer(fn, n_iter=200, warmup=30):
    """Time a callable with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms


# ─── S³-Merge merge step, broken into phases ──────────────────────────────────

def srtome_phases(patches, grid_size, window_size, target_tokens):
    """
    Run the S³-Merge merge step and return per-phase outputs.
    patches: [B, N, C]
    Returns: (merged_patches, restore_idx, phase_closures)
    where phase_closures is a dict of callables for timing.
    """
    B, N, C = patches.shape
    H = W = grid_size
    ws = window_size
    n_win_h = H // ws
    n_win_w = W // ws
    n_windows = n_win_h * n_win_w
    tokens_per_window = ws * ws
    K = max(1, min(target_tokens // n_windows, tokens_per_window))
    n_sources = tokens_per_window - K
    device = patches.device

    if n_sources <= 0:
        return patches, None, {}

    # ── Phase closures (each can be timed independently) ──

    def phase_window_reshape():
        grid = patches.view(B, H, W, C)
        grid_t = grid.permute(0, 3, 1, 2)
        windows = grid_t.unfold(2, ws, ws).unfold(3, ws, ws)
        windows = windows.permute(0, 2, 3, 4, 5, 1)
        return windows.reshape(B * n_windows, tokens_per_window, C)

    flat_windows = phase_window_reshape()
    BnW = flat_windows.shape[0]

    def phase_target_selection():
        rand_scores = torch.rand(BnW, tokens_per_window, device=device)
        target_idx = torch.topk(rand_scores, K, dim=1, sorted=False).indices
        source_idx = torch.topk(rand_scores, n_sources, dim=1, largest=False, sorted=False).indices
        return target_idx, source_idx

    target_idx, source_idx = phase_target_selection()

    def phase_gather():
        targets = flat_windows.gather(1, target_idx.unsqueeze(-1).expand(-1, -1, C))
        sources = flat_windows.gather(1, source_idx.unsqueeze(-1).expand(-1, -1, C))
        return targets, sources

    targets, sources = phase_gather()

    def phase_similarity():
        targets_n = F.normalize(targets, dim=-1)
        sources_n = F.normalize(sources, dim=-1)
        sim = torch.bmm(sources_n, targets_n.transpose(1, 2))
        best = sim.max(dim=-1).indices
        return best

    best_target_local = phase_similarity()

    def phase_merge_scatter():
        scatter_idx = best_target_local.unsqueeze(-1).expand(-1, -1, C)
        merged = targets.scatter_reduce(1, scatter_idx, sources, reduce='mean')
        return merged.view(B, n_windows, K, C).reshape(B, -1, C)

    merged_patches = phase_merge_scatter()

    def phase_restore_idx():
        """Build the restore index mapping."""
        restore_idx_window = torch.zeros(BnW, tokens_per_window, dtype=torch.long, device=device)
        win_idx_flat = torch.arange(BnW, device=device) % n_windows
        window_base = win_idx_flat * K
        target_local_rank = torch.arange(K, device=device).unsqueeze(0).expand(BnW, -1)
        target_global_idx = window_base.unsqueeze(1) + target_local_rank
        restore_idx_window.scatter_(1, target_idx, target_global_idx)
        source_global_idx = target_global_idx.gather(1, best_target_local)
        restore_idx_window.scatter_(1, source_idx, source_global_idx)

        # Precomputed grid mapping
        h_idx = torch.arange(H, device=device).view(H, 1).expand(H, W)
        w_idx = torch.arange(W, device=device).view(1, W).expand(H, W)
        local_win_idx = ((h_idx // ws) * n_win_w + (w_idx // ws)).view(-1)
        pos_in_window = ((h_idx % ws) * ws + (w_idx % ws)).view(-1)

        batch_off = torch.arange(B, device=device).view(B, 1) * n_windows
        flat_win = batch_off + local_win_idx.unsqueeze(0)
        pos_exp = pos_in_window.unsqueeze(0).expand(B, -1)
        gather_idx = flat_win * tokens_per_window + pos_exp
        restore = restore_idx_window.view(-1)[gather_idx.view(-1)].view(B, N)
        return restore

    restore_idx = phase_restore_idx()

    def phase_unmerge():
        """Reconstruct dense grid from merged patches using restore_idx."""
        idx = restore_idx.unsqueeze(-1).expand(-1, -1, C)
        dense = merged_patches.gather(1, idx)
        return dense.view(B, H, W, C)

    phases = {
        'window_reshape': phase_window_reshape,
        'target_selection': phase_target_selection,
        'gather': phase_gather,
        'similarity': phase_similarity,
        'merge_scatter': phase_merge_scatter,
        'restore_idx': phase_restore_idx,
        'unmerge': phase_unmerge,
    }

    return merged_patches, restore_idx, phases


# ─── ToMe-style merge for comparison ──────────────────────────────────────────

def tome_merge_step(patches, r):
    """Simplified ToMe bipartite merge for one layer (for timing comparison)."""
    B, N, C = patches.shape
    device = patches.device

    # Even/odd partition
    a = patches[:, ::2, :]   # [B, N//2, C]
    b = patches[:, 1::2, :]  # [B, N//2, C]

    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    sim = torch.bmm(a_n, b_n.transpose(1, 2))  # [B, N//2, N//2]

    # Top-r most similar pairs
    node_max, node_idx = sim.max(dim=-1)  # [B, N//2]
    _, edge_idx = node_max.topk(r, dim=-1)  # [B, r]

    # Merge top-r sources into their matched targets
    src_idx = edge_idx  # indices into a
    dst_gather = node_idx.gather(1, edge_idx)  # indices into b

    # Average merge
    src_tokens = a.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, C))
    dst_tokens = b.gather(1, dst_gather.unsqueeze(-1).expand(-1, -1, C))
    merged = (src_tokens + dst_tokens) / 2

    # Replace in b
    b = b.scatter(1, dst_gather.unsqueeze(-1).expand(-1, -1, C), merged)

    # Remove merged from a
    keep_mask = torch.ones(B, a.shape[1], device=device, dtype=torch.bool)
    keep_mask.scatter_(1, src_idx, False)

    # Rebuild token sequence (simplified: just concat kept-a and b)
    kept_a = a[keep_mask.unsqueeze(-1).expand_as(a)].view(B, -1, C)
    return torch.cat([kept_a, b], dim=1)


# ─── Main profiling ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model', type=str, default='deit_small_patch16_224')
    args = parser.parse_args()

    B = args.batch_size
    n_iter = args.n_iter
    warmup = args.warmup

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {args.model}, B={B}, img_size={args.img_size}")

    # Load model
    model = timm.create_model(args.model, pretrained=False).eval().to(device)
    patch_size = model.patch_embed.patch_size[0]
    grid_size = args.img_size // patch_size
    N = grid_size * grid_size
    C = model.embed_dim
    num_layers = len(model.blocks)

    # Find compatible window size
    ws = 7
    for candidate in [7, 8, 4, 2]:
        if grid_size % candidate == 0:
            ws = candidate
            break

    target_tokens = int(N * 0.7)  # r=0.7
    tome_r = 10  # standard ToMe r per layer

    print(f"Grid: {grid_size}x{grid_size} = {N} tokens, C={C}")
    print(f"Window size: {ws}, target_tokens: {target_tokens}")
    print(f"Layers: {num_layers}")
    print()

    # ── Create dummy inputs ──
    dummy_img = torch.randn(B, 3, args.img_size, args.img_size, device=device, dtype=torch.float16)

    # ── 1) Full baseline forward ──
    with torch.no_grad(), torch.cuda.amp.autocast():
        baseline_ms = cuda_timer(lambda: model(dummy_img), n_iter, warmup)
    print(f"Baseline full forward:     {baseline_ms:.2f} ms")

    # ── 2) S³-Merge: full forward with merge ──
    # We need to time the transformer blocks + merge
    def srtome_forward():
        with torch.no_grad(), torch.cuda.amp.autocast():
            x = model.patch_embed(dummy_img)
            x = x + model.pos_embed[:, 1:, :]
            cls_token = model.cls_token.expand(B, -1, -1) + model.pos_embed[:, :1, :]
            x = torch.cat([cls_token, x], dim=1)
            x = model.pos_drop(x)

            merge_layer = 8  # l8
            for i, block in enumerate(model.blocks):
                x = block(x)
                if i == merge_layer - 1:  # after layer L_s
                    cls_tok = x[:, :1]
                    patches = x[:, 1:]
                    merged, restore, _ = srtome_phases(patches, grid_size, ws, target_tokens)
                    x = torch.cat([cls_tok, merged], dim=1)

            x = model.norm(x)
            return x[:, 0]

    srtome_ms = cuda_timer(srtome_forward, n_iter, warmup)
    print(f"S³-Merge full forward:     {srtome_ms:.2f} ms")

    # ── 3) ToMe: full forward with per-layer merge (simplified) ──
    def tome_forward():
        with torch.no_grad(), torch.cuda.amp.autocast():
            x = model.patch_embed(dummy_img)
            x = x + model.pos_embed[:, 1:, :]
            cls_token = model.cls_token.expand(B, -1, -1) + model.pos_embed[:, :1, :]
            x = torch.cat([cls_token, x], dim=1)
            x = model.pos_drop(x)

            for i, block in enumerate(model.blocks):
                x = block(x)
                # ToMe merges at every layer
                if x.shape[1] > 1 + tome_r:
                    cls_tok = x[:, :1]
                    patches = x[:, 1:]
                    patches = tome_merge_step(patches, min(tome_r, patches.shape[1] // 2 - 1))
                    x = torch.cat([cls_tok, patches], dim=1)

            x = model.norm(x)
            return x[:, 0]

    tome_ms = cuda_timer(tome_forward, n_iter, warmup)
    print(f"ToMe full forward:         {tome_ms:.2f} ms")

    # ── 4) Phase-level breakdown (S³-Merge merge step only) ──
    print("\n--- S³-Merge merge step breakdown ---")
    dummy_patches = torch.randn(B, N, C, device=device, dtype=torch.float16)
    with torch.no_grad(), torch.cuda.amp.autocast():
        _, _, phases = srtome_phases(dummy_patches, grid_size, ws, target_tokens)

    phase_times = {}
    for name, fn in phases.items():
        with torch.no_grad(), torch.cuda.amp.autocast():
            t = cuda_timer(fn, n_iter, warmup)
        phase_times[name] = t
        print(f"  {name:20s}: {t:.3f} ms")

    merge_overhead = sum(phase_times.values())
    print(f"  {'TOTAL merge overhead':20s}: {merge_overhead:.3f} ms")

    # ── 5) Transformer block timing ──
    pre_merge_layers = 8
    post_merge_layers = num_layers - pre_merge_layers

    # Time a single block (pre-merge, full N tokens)
    dummy_full = torch.randn(B, 1 + N, C, device=device, dtype=torch.float16)
    with torch.no_grad(), torch.cuda.amp.autocast():
        block_full_ms = cuda_timer(lambda: model.blocks[0](dummy_full), n_iter, warmup)

    # Time a single block (post-merge, reduced tokens)
    T = target_tokens
    dummy_reduced = torch.randn(B, 1 + T, C, device=device, dtype=torch.float16)
    with torch.no_grad(), torch.cuda.amp.autocast():
        block_reduced_ms = cuda_timer(lambda: model.blocks[0](dummy_reduced), n_iter, warmup)

    pre_merge_total = pre_merge_layers * block_full_ms
    post_merge_baseline = post_merge_layers * block_full_ms
    post_merge_srtome = post_merge_layers * block_reduced_ms

    print(f"\n--- Transformer block timing ---")
    print(f"  Single block (full {1+N} tok):    {block_full_ms:.3f} ms")
    print(f"  Single block (reduced {1+T} tok): {block_reduced_ms:.3f} ms")
    print(f"  Pre-merge ({pre_merge_layers} layers):           {pre_merge_total:.2f} ms")
    print(f"  Post-merge baseline ({post_merge_layers} layers): {post_merge_baseline:.2f} ms")
    print(f"  Post-merge S³-Merge ({post_merge_layers} layers): {post_merge_srtome:.2f} ms")
    print(f"  Compute saved:                    {post_merge_baseline - post_merge_srtome:.2f} ms")
    print(f"  Net gain (saved - overhead):      {(post_merge_baseline - post_merge_srtome) - merge_overhead:.2f} ms")

    # ── 6) Summary table ──
    print("\n" + "=" * 70)
    print("OVERHEAD BREAKDOWN SUMMARY (DeiT-S, B={})".format(B))
    print("=" * 70)

    summary = {
        'Pre-merge blocks (layers 1–{})'.format(pre_merge_layers): pre_merge_total,
        'Merge step (S³-Merge)': merge_overhead,
        '  ├ Window reshape': phase_times.get('window_reshape', 0),
        '  ├ Target selection': phase_times.get('target_selection', 0),
        '  ├ Gather': phase_times.get('gather', 0),
        '  ├ Similarity': phase_times.get('similarity', 0),
        '  ├ Merge scatter': phase_times.get('merge_scatter', 0),
        '  ├ Restore index': phase_times.get('restore_idx', 0),
        '  └ Unmerge': phase_times.get('unmerge', 0),
        'Post-merge blocks (baseline)': post_merge_baseline,
        'Post-merge blocks (S³-Merge)': post_merge_srtome,
    }

    for label, val in summary.items():
        print(f"  {label:42s}  {val:7.3f} ms")

    print(f"\n  Baseline total (est.):  {pre_merge_total + post_merge_baseline:.2f} ms")
    print(f"  S³-Merge total (est.): {pre_merge_total + merge_overhead + post_merge_srtome:.2f} ms")
    print(f"  Measured baseline:      {baseline_ms:.2f} ms")
    print(f"  Measured S³-Merge:      {srtome_ms:.2f} ms")
    print(f"  Measured ToMe:          {tome_ms:.2f} ms")

    # ── 7) Save CSV ──
    out_dir = Path('results_downstream')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / 'overhead_breakdown.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component', 'time_ms'])
        writer.writerow(['pre_merge_blocks', f'{pre_merge_total:.3f}'])
        for name in ['window_reshape', 'target_selection', 'gather',
                      'similarity', 'merge_scatter', 'restore_idx', 'unmerge']:
            writer.writerow([f'srtome_{name}', f'{phase_times.get(name, 0):.3f}'])
        writer.writerow(['srtome_merge_total', f'{merge_overhead:.3f}'])
        writer.writerow(['post_merge_baseline', f'{post_merge_baseline:.3f}'])
        writer.writerow(['post_merge_srtome', f'{post_merge_srtome:.3f}'])
        writer.writerow(['full_baseline', f'{baseline_ms:.3f}'])
        writer.writerow(['full_srtome', f'{srtome_ms:.3f}'])
        writer.writerow(['full_tome', f'{tome_ms:.3f}'])
    print(f"\nSaved: {csv_path}")

    # ── 8) LaTeX table snippet ──
    print("\n--- LaTeX table snippet ---")
    print(r"""
\begin{table}[t]
\centering
\small
\caption{\textbf{Overhead breakdown} (DeiT-S, $B=%d$, H100 NVL).
Time is in ms; ``Compute saved'' is the reduction in post-merge transformer blocks.}
\label{tab:overhead}
\begin{tabular}{lrrr}
\toprule
\textbf{Component} & \textbf{Baseline} & \textbf{ToMe} & \textbf{S$^3$-Merge} \\
\midrule
Pre-merge blocks & %.2f & %.2f & %.2f \\
Merge overhead   & --   & (per-layer) & %.3f \\
Post-merge blocks & %.2f & -- & %.2f \\
\midrule
\textbf{Total}   & \textbf{%.2f} & \textbf{%.2f} & \textbf{%.2f} \\
\bottomrule
\end{tabular}
\end{table}
""" % (B,
       pre_merge_total, pre_merge_total, pre_merge_total,
       merge_overhead,
       post_merge_baseline, post_merge_srtome,
       baseline_ms, tome_ms, srtome_ms))


if __name__ == '__main__':
    main()
