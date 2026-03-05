#!/usr/bin/env python3
"""
Ablation Study: What makes ToMe work?

A1: Replace even/odd partition with deterministic top-k partition
    - If accuracy collapses → diversity of partition matters

A2: Replace similarity matching with uniform/spatial merge
    - If accuracy holds → diversity > matching quality

SR-ToMe: Stratified Random ToMe with windowed local merging
    - Keep diversity but add spatial coverage constraints
"""
import argparse
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import tome
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_imagenet_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = load_dataset("ILSVRC/imagenet-1k", split='validation', cache_dir=os.environ.get("IMAGENET_CACHE", "./data/imagenet_hf"))
    def collate(batch):
        imgs = [transform(x['image'].convert('RGB')) for x in batch]
        labels = [x['label'] for x in batch]
        return torch.stack(imgs), torch.tensor(labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)


# ============ ABLATION MERGE FUNCTIONS ============

def merge_deterministic_topk(x, r, importance='norm'):
    """
    A1: Replace ToMe's even/odd with deterministic top-k partition.
    Sources = bottom r by importance, Targets = rest.
    Uses ToMe-style similarity matching for merge.
    """
    B, N, C = x.shape
    if r <= 0 or r >= N - 1:
        return x
    
    # Compute importance
    if importance == 'norm':
        scores = x.norm(dim=-1)
    else:
        scores = torch.rand(B, N, device=x.device)
    
    # Protect CLS
    scores[:, 0] = float('inf')
    
    # Get sources (lowest importance) and targets
    _, order = scores.sort(dim=1)
    source_idx = order[:, :r]  # lowest r
    target_idx = order[:, r:]  # rest
    
    # Gather
    sources = x.gather(1, source_idx.unsqueeze(-1).expand(-1, -1, C))
    targets = x.gather(1, target_idx.unsqueeze(-1).expand(-1, -1, C))
    
    # Similarity matching (like ToMe)
    src_n = F.normalize(sources, dim=-1)
    tgt_n = F.normalize(targets, dim=-1)
    sim = torch.bmm(src_n, tgt_n.transpose(1, 2))
    
    _, best_tgt = sim.max(dim=-1)
    
    # Merge via scatter_reduce (mean)
    scatter_idx = best_tgt.unsqueeze(-1).expand(-1, -1, C)
    targets = targets.scatter_reduce(1, scatter_idx, sources, reduce='mean')
    
    return targets


def merge_uniform_random(x, r):
    """
    A2: Keep ToMe's even/odd partition but use uniform merge (no similarity).
    Sources merge into their positional neighbor, not best match.
    """
    B, N, C = x.shape
    if r <= 0:
        return x
    
    # ToMe-style even/odd split
    src = x[:, ::2, :]  # even indices
    dst = x[:, 1::2, :]  # odd indices
    
    n_src = src.shape[1]
    n_dst = dst.shape[1]
    
    if r > n_src:
        return x
    
    # Simple: merge first r sources into corresponding dst (positional)
    # No similarity matching - just take neighbors
    merged_dst = dst.clone()
    
    # Uniform merge: each source r goes to dst at same position
    for i in range(min(r, n_dst)):
        # Average merge
        merged_dst[:, i] = (merged_dst[:, i] + src[:, i]) / 2
    
    # Keep remaining sources
    unmerged_src = src[:, r:]
    
    return torch.cat([unmerged_src, merged_dst], dim=1)


def merge_even_odd_vectorized(x, r):
    """
    A2b: Vectorized even/odd with positional (not similarity) merge.
    """
    B, N, C = x.shape
    if r <= 0:
        return x
    
    src = x[:, ::2, :]  # even
    dst = x[:, 1::2, :]  # odd
    
    n_src = src.shape[1]
    r = min(r, n_src)
    
    # Source tokens to merge: just take first r
    to_merge = src[:, :r]  # [B, r, C]
    to_keep = src[:, r:]   # [B, n_src-r, C]
    
    # Merge into corresponding dst position (uniform, no similarity)
    merged_dst = dst.clone()
    if r <= dst.shape[1]:
        merged_dst[:, :r] = (dst[:, :r] + to_merge) / 2
    
    return torch.cat([to_keep, merged_dst], dim=1)


def merge_sr_tome_windowed(x, r, window_size=2, preserve_cls=True):
    """
    SR-ToMe: Stratified Random ToMe with windowed local merging.
    
    1. Reshape tokens to 14x14 grid (for 224px, patch_size=16)
    2. Partition into window_size x window_size windows
    3. Within each window: random bipartite + similarity merge
    4. Ensure each window keeps at least 1 token
    """
    B, N, C = x.shape
    
    # Separate CLS
    if preserve_cls:
        cls_token = x[:, :1]
        patches = x[:, 1:]
        N_patches = N - 1
    else:
        patches = x
        N_patches = N
    
    # Determine grid size
    H = W = int(N_patches ** 0.5)
    if H * W != N_patches:
        # Not a perfect square, fall back to ToMe
        return x
    
    # Reshape to grid: [B, H, W, C]
    grid = patches.view(B, H, W, C)
    
    # Compute tokens per window and how many to merge
    n_windows_h = H // window_size
    n_windows_w = W // window_size
    tokens_per_window = window_size * window_size
    n_windows = n_windows_h * n_windows_w
    
    # Distribute r merges across windows
    r_per_window = r // n_windows
    r_remainder = r % n_windows
    
    # Cap: must keep at least 1 token per window
    r_per_window = min(r_per_window, tokens_per_window - 1)
    
    outputs = []
    
    for wi in range(n_windows_h):
        for wj in range(n_windows_w):
            # Extract window
            i_start = wi * window_size
            i_end = (wi + 1) * window_size
            j_start = wj * window_size
            j_end = (wj + 1) * window_size
            
            window = grid[:, i_start:i_end, j_start:j_end, :]  # [B, ws, ws, C]
            window = window.reshape(B, -1, C)  # [B, tokens_per_window, C]
            
            # Local r for this window
            win_idx = wi * n_windows_w + wj
            local_r = r_per_window + (1 if win_idx < r_remainder else 0)
            local_r = min(local_r, tokens_per_window - 1)
            
            if local_r > 0 and window.shape[1] > 2:
                # Random bipartite within window (ToMe-style)
                window = local_bipartite_merge(window, local_r)
            
            outputs.append(window)
    
    # Concatenate
    merged_patches = torch.cat(outputs, dim=1)
    
    if preserve_cls:
        return torch.cat([cls_token, merged_patches], dim=1)
    return merged_patches


def local_bipartite_merge(x, r):
    """ToMe-style bipartite merge within a local window."""
    B, N, C = x.shape
    if r <= 0 or N < 2:
        return x
    
    r = min(r, N // 2)
    
    # Even/odd split (like ToMe)
    src = x[:, ::2, :]
    dst = x[:, 1::2, :]
    
    n_src = src.shape[1]
    n_dst = dst.shape[1]
    
    if n_src == 0 or n_dst == 0:
        return x
    
    r = min(r, n_src)
    
    # Similarity matching
    src_n = F.normalize(src, dim=-1)
    dst_n = F.normalize(dst, dim=-1)
    sim = torch.bmm(src_n, dst_n.transpose(1, 2))  # [B, n_src, n_dst]
    
    # For each source, find best dst
    best_sim, best_idx = sim.max(dim=-1)  # [B, n_src]
    
    # Sort by similarity to decide which to merge
    _, merge_order = best_sim.sort(dim=1, descending=True)
    
    merge_mask = merge_order[:, :r]  # [B, r]
    keep_mask = merge_order[:, r:]   # [B, n_src-r]
    
    # Gather sources to merge and keep
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
    to_merge = src[batch_idx.expand(-1, r), merge_mask]  # [B, r, C]
    to_keep = src[batch_idx.expand(-1, n_src - r), keep_mask]  # [B, n_src-r, C]
    
    # Get target indices for merged sources
    tgt_idx = best_idx.gather(1, merge_mask)  # [B, r]
    
    # Scatter merge into dst
    scatter_idx = tgt_idx.unsqueeze(-1).expand(-1, -1, C)
    merged_dst = dst.scatter_reduce(1, scatter_idx, to_merge, reduce='mean')
    
    return torch.cat([to_keep, merged_dst], dim=1)


# ============ WRAPPER CLASSES ============

class AblationWrapper(nn.Module):
    """Wrapper that applies a custom merge function."""
    
    def __init__(self, model, merge_fn, r_per_layer=10, late_start=0):
        super().__init__()
        self.model = model
        self.merge_fn = merge_fn
        self.r_per_layer = r_per_layer
        self.late_start = late_start
        
    def forward(self, x):
        x = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            
            if i >= self.late_start:
                # Apply custom merge
                x = self.merge_fn(x, self.r_per_layer)
        
        x = self.model.norm(x)
        return self.model.head(x[:, 0])


# ============ EVALUATION ============

def evaluate(model, loader, max_batches=50, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(loader, total=max_batches)):
            if i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_batches', default=50, type=int)
    parser.add_argument('--output_dir', default='artifacts')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    
    loader = get_imagenet_loader(batch_size=64)
    results = []
    
    # Baseline
    print("="*70)
    print("BASELINE")
    print("="*70)
    baseline = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    base_acc = evaluate(baseline, loader, args.max_batches)
    print(f"Accuracy: {base_acc:.2f}%")
    results.append({'method': 'baseline', 'accuracy': base_acc, 'drop': 0})
    del baseline
    torch.cuda.empty_cache()
    
    # ToMe (original)
    print("\n" + "="*70)
    print("ToMe (original)")
    print("="*70)
    tome_m = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    tome.patch.timm(tome_m)
    tome_m.r = 10
    tome_acc = evaluate(tome_m, loader, args.max_batches)
    print(f"Accuracy: {tome_acc:.2f}% (drop: {base_acc - tome_acc:.2f}%)")
    results.append({'method': 'tome', 'accuracy': tome_acc, 'drop': base_acc - tome_acc})
    del tome_m
    torch.cuda.empty_cache()
    
    # A1: Deterministic top-k partition
    print("\n" + "="*70)
    print("A1: Deterministic Top-K (norm-based)")
    print("="*70)
    base = timm.create_model('deit_small_patch16_224', pretrained=True).to(device)
    a1 = AblationWrapper(base, merge_deterministic_topk, r_per_layer=10, late_start=0).eval()
    a1_acc = evaluate(a1, loader, args.max_batches)
    print(f"Accuracy: {a1_acc:.2f}% (drop: {base_acc - a1_acc:.2f}%)")
    results.append({'method': 'A1_deterministic_topk', 'accuracy': a1_acc, 'drop': base_acc - a1_acc})
    del a1
    torch.cuda.empty_cache()
    
    # A2: Uniform merge (no similarity)
    print("\n" + "="*70)
    print("A2: Uniform Merge (even/odd + positional merge)")
    print("="*70)
    base = timm.create_model('deit_small_patch16_224', pretrained=True).to(device)
    a2 = AblationWrapper(base, merge_even_odd_vectorized, r_per_layer=10, late_start=0).eval()
    a2_acc = evaluate(a2, loader, args.max_batches)
    print(f"Accuracy: {a2_acc:.2f}% (drop: {base_acc - a2_acc:.2f}%)")
    results.append({'method': 'A2_uniform_merge', 'accuracy': a2_acc, 'drop': base_acc - a2_acc})
    del a2
    torch.cuda.empty_cache()
    
    # SR-ToMe (windowed)
    print("\n" + "="*70)
    print("SR-ToMe (windowed, 2x2)")
    print("="*70)
    base = timm.create_model('deit_small_patch16_224', pretrained=True).to(device)
    sr = AblationWrapper(base, lambda x, r: merge_sr_tome_windowed(x, r, window_size=2), r_per_layer=10, late_start=6).eval()
    sr_acc = evaluate(sr, loader, args.max_batches)
    print(f"Accuracy: {sr_acc:.2f}% (drop: {base_acc - sr_acc:.2f}%)")
    results.append({'method': 'sr_tome_2x2', 'accuracy': sr_acc, 'drop': base_acc - sr_acc})
    del sr
    torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'Accuracy':<12} {'Drop':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['method']:<30} {r['accuracy']:<12.2f} {r['drop']:<10.2f}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    a1_worse = results[2]['drop'] > results[1]['drop'] + 0.3
    a2_close = abs(results[3]['drop'] - results[1]['drop']) < 0.5
    
    print(f"A1 (deterministic) vs ToMe: {'WORSE → randomness matters' if a1_worse else 'SIMILAR → randomness not key'}")
    print(f"A2 (uniform merge) vs ToMe: {'CLOSE → similarity not critical' if a2_close else 'WORSE → similarity matters'}")
    
    # Save
    csv_path = output_dir / 'ablation_a1_a2_sr.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")


if __name__ == '__main__':
    main()
