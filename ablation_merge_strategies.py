#!/usr/bin/env python3
"""
Ablation study: Isolate what makes HTM slow and inaccurate.
Tests:
1. HTM with β=0 (attention only, no energy)
2. HTM random targets (test if selection is culprit)
3. ToMe-style merge (random bipartite partition)
4. Measure accuracy difference
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

import sys
sys.path.insert(0, str(Path(__file__).parent))


def get_imagenet_loader(batch_size=64, split='validation'):
    """Load ImageNet from HuggingFace."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=os.environ.get("IMAGENET_CACHE", "./data/imagenet_hf"))
    
    def collate_fn(batch):
        images, labels = [], []
        for item in batch:
            images.append(transform(item['image'].convert('RGB')))
            labels.append(item['label'])
        return torch.stack(images), torch.tensor(labels)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)


# ============ Merge Strategies ============

def merge_importance_based(x, r, prefix_tokens=1, strategy='norm'):
    """Merge using importance-based selection."""
    B, N, D = x.shape
    if r <= 0 or N - prefix_tokens <= r:
        return x
    
    prefix = x[:, :prefix_tokens]
    patches = x[:, prefix_tokens:]
    N_patches = patches.shape[1]
    
    if N_patches <= r:
        return x
    
    # Compute importance
    if strategy == 'norm':
        importance = patches.norm(dim=-1)
    elif strategy == 'random':
        importance = torch.rand(B, N_patches, device=x.device)
    else:
        importance = patches.norm(dim=-1)
    
    # Sort and split
    _, sorted_idx = importance.sort(dim=1, descending=True)
    n_targets = N_patches - r
    target_idx = sorted_idx[:, :n_targets]
    source_idx = sorted_idx[:, n_targets:]
    
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
    targets = patches[batch_idx.expand(-1, n_targets), target_idx]
    sources = patches[batch_idx.expand(-1, r), source_idx]
    
    # Similarity-based merge
    targets_norm = F.normalize(targets, dim=-1)
    sources_norm = F.normalize(sources, dim=-1)
    similarity = torch.bmm(sources_norm, targets_norm.transpose(1, 2))
    
    best_idx = similarity.argmax(dim=-1)
    best_sim = similarity.gather(2, best_idx.unsqueeze(-1)).squeeze(-1)
    
    scatter_idx = best_idx.unsqueeze(-1).expand(-1, -1, D)
    weighted_sources = sources * best_sim.unsqueeze(-1)
    merged = targets.scatter_add(1, scatter_idx, weighted_sources)
    
    weight_sum = torch.ones(B, n_targets, device=x.device)
    weight_sum = weight_sum.scatter_add(1, best_idx, best_sim)
    merged = merged / weight_sum.unsqueeze(-1).clamp(min=1e-6)
    
    return torch.cat([prefix, merged], dim=1)


def merge_tome_style(x, r, prefix_tokens=1):
    """Merge using ToMe-style random bipartite partition."""
    B, N, D = x.shape
    if r <= 0 or N - prefix_tokens <= r:
        return x
    
    prefix = x[:, :prefix_tokens]
    patches = x[:, prefix_tokens:]
    N_patches = patches.shape[1]
    
    if N_patches <= 2 * r:  # Need at least 2r tokens
        return x
    
    # Random bipartite partition (ToMe style)
    # Split into A and B randomly
    n_a = N_patches // 2
    n_b = N_patches - n_a
    
    # Random permutation
    perm = torch.randperm(N_patches, device=x.device)
    a_idx = perm[:n_a]
    b_idx = perm[n_a:]
    
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
    a_tokens = patches[batch_idx.expand(-1, n_a), a_idx.unsqueeze(0).expand(B, -1)]
    b_tokens = patches[batch_idx.expand(-1, n_b), b_idx.unsqueeze(0).expand(B, -1)]
    
    # Compute similarity: A matches to B
    a_norm = F.normalize(a_tokens, dim=-1)
    b_norm = F.normalize(b_tokens, dim=-1)
    similarity = torch.bmm(a_norm, b_norm.transpose(1, 2))  # [B, n_a, n_b]
    
    # Find best match for each A token
    best_idx = similarity.argmax(dim=-1)  # [B, n_a]
    best_sim = similarity.gather(2, best_idx.unsqueeze(-1)).squeeze(-1)
    
    # Select top-r A tokens to merge (by similarity)
    _, merge_order = best_sim.sort(dim=1, descending=True)
    merge_idx = merge_order[:, :r]  # [B, r]
    keep_idx = merge_order[:, r:]    # [B, n_a - r]
    
    # Merge selected A tokens into their best B match
    merged_b = b_tokens.clone()
    for i in range(r):
        src_idx = merge_idx[:, i]  # [B]
        tgt_idx = best_idx.gather(1, src_idx.unsqueeze(1)).squeeze(1)  # [B]
        
        src_tokens = a_tokens[batch_idx.squeeze(), src_idx]  # [B, D]
        src_sim = best_sim.gather(1, src_idx.unsqueeze(1)).squeeze(1)  # [B]
        
        # Average into target
        for b in range(B):
            merged_b[b, tgt_idx[b]] = (merged_b[b, tgt_idx[b]] + src_sim[b] * src_tokens[b]) / (1 + src_sim[b])
    
    # Keep remaining A tokens
    kept_a = a_tokens[batch_idx.expand(-1, n_a - r), keep_idx]
    
    # Reconstruct
    output = torch.cat([prefix, kept_a, merged_b], dim=1)
    return output


# ============ Wrapper Classes ============

class MergeWrapper(nn.Module):
    def __init__(self, model, merge_schedule, strategy='norm'):
        super().__init__()
        self.model = model
        self.merge_schedule = merge_schedule
        self.strategy = strategy
        
    def forward(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        
        for i, blk in enumerate(self.model.blocks):
            r = self.merge_schedule[i] if i < len(self.merge_schedule) else 0
            if r > 0:
                x = merge_importance_based(x, r, prefix_tokens=1, strategy=self.strategy)
            x = blk(x)
        
        x = self.model.norm(x)
        return self.model.head(x[:, 0])


# ============ Evaluation ============

def evaluate(model, dataloader, max_batches=50, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, total=max_batches)):
            if i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def time_model(model, x, runs=50, device='cuda'):
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
            torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / runs * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deit_small_patch16_224')
    parser.add_argument('--merge_rate', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_batches', default=50, type=int)
    parser.add_argument('--output_dir', default='artifacts')
    args = parser.parse_args()
    
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Model: {args.model}")
    print(f"Merge rate: {args.merge_rate}")
    print()
    
    # Data
    dataloader = get_imagenet_loader(batch_size=args.batch_size)
    input_b64 = torch.randn(args.batch_size, 3, 224, 224, device=device)
    
    # Models
    num_layers = 12
    merge_schedule = [0] + [args.merge_rate] * (num_layers - 1)
    
    models = {}
    
    # Baseline
    models['baseline'] = timm.create_model(args.model, pretrained=True).to(device).eval()
    
    # HTM-norm (standard)
    base = timm.create_model(args.model, pretrained=True).to(device)
    models['htm_norm'] = MergeWrapper(base, merge_schedule, strategy='norm').eval()
    
    # HTM-random (random importance)
    base = timm.create_model(args.model, pretrained=True).to(device)
    models['htm_random'] = MergeWrapper(base, merge_schedule, strategy='random').eval()
    
    # ToMe (official)
    tome_model = timm.create_model(args.model, pretrained=True).to(device).eval()
    tome.patch.timm(tome_model)
    tome_model.r = args.merge_rate
    models['tome'] = tome_model
    
    # Results
    results = []
    
    print("="*70)
    print("ABLATION STUDY: MERGE STRATEGIES")
    print("="*70)
    print(f"{'Method':<20} {'Latency (ms)':<15} {'Accuracy':<12} {'Drop':<10}")
    print("-"*70)
    
    baseline_acc = None
    
    for name, model in models.items():
        lat = time_model(model, input_b64, device=device)
        acc = evaluate(model, dataloader, args.max_batches, device)
        
        if name == 'baseline':
            baseline_acc = acc
            drop = 0.0
        else:
            drop = baseline_acc - acc
        
        print(f"{name:<20} {lat:<15.2f} {acc:<12.2f}% {drop:<10.2f}%")
        
        results.append({
            'method': name,
            'latency_ms': lat,
            'accuracy': acc,
            'drop': drop,
        })
        
        del model
        torch.cuda.empty_cache()
    
    print("="*70)
    
    # Save
    csv_path = output_dir / 'ablation_merge_strategies.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")


if __name__ == '__main__':
    main()
