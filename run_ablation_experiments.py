#!/usr/bin/env python3
"""
Experiments for reviewer response:
1. Target selection ablation: random vs norm vs grid vs attention  
2. 5-seed classification with t-test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = torch.device('cuda')

# ============================================================
# Data
# ============================================================
def get_loader(batch_size=64):
    from datasets import load_dataset
    from torchvision import transforms
    
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation",
                      cache_dir=os.environ.get("IMAGENET_CACHE", "./data/imagenet_hf"))
    
    class W(torch.utils.data.Dataset):
        def __init__(s, d, t): s.d, s.t = d, t
        def __len__(s): return len(s.d)
        def __getitem__(s, i):
            x = s.d[i]; return s.t(x['image'].convert('RGB')), x['label']
    
    return torch.utils.data.DataLoader(
        W(ds, tfm), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

def evaluate(model, loader, max_batches=None):
    correct = total = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if max_batches and i >= max_batches: break
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            correct += (out.argmax(-1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# ============================================================
# PART 1: Selection strategy ablation
# Uses a standalone merge that accepts strategy parameter,
# called from a custom SRToMeBlock replacement.
# ============================================================

def merge_with_strategy(x, target_tokens, window_size=7, grid_size=14,
                        strategy='random', track_sizes=False):
    """S³-Merge with configurable target selection strategy."""
    B, N, C = x.shape
    cls_token = x[:, :1]
    patches = x[:, 1:]
    H = W = grid_size
    
    if patches.shape[1] != H * W or window_size >= grid_size:
        return x, None
    
    ws = window_size
    nH, nW = H // ws, W // ws
    n_windows = nH * nW
    tpw = ws * ws  # tokens per window
    K = max(1, min(target_tokens // n_windows, tpw))
    n_src = tpw - K
    if n_src <= 0:
        return x, None
    
    grid = patches.view(B, H, W, C)
    windows = grid.view(B, nH, ws, nW, ws, C).permute(0, 1, 3, 2, 4, 5)
    flat = windows.reshape(B * n_windows, tpw, C)
    BnW = flat.shape[0]
    
    # ---- Target selection ----
    if strategy == 'random':
        rs = torch.rand(BnW, tpw, device=x.device)
        _, t_idx = torch.topk(rs, K, dim=1, sorted=False)
        _, s_idx = torch.topk(rs, n_src, dim=1, largest=False, sorted=False)
        
    elif strategy == 'norm':
        norms = flat.norm(dim=-1)
        _, t_idx = torch.topk(norms, K, dim=1, sorted=False)
        all_i = torch.arange(tpw, device=x.device).expand(BnW, -1)
        mask = torch.ones(BnW, tpw, device=x.device, dtype=torch.bool)
        mask.scatter_(1, t_idx, False)
        s_idx = all_i[mask].view(BnW, n_src)
        
    elif strategy == 'norm_low':
        norms = flat.norm(dim=-1)
        _, t_idx = torch.topk(norms, K, dim=1, largest=False, sorted=False)
        all_i = torch.arange(tpw, device=x.device).expand(BnW, -1)
        mask = torch.ones(BnW, tpw, device=x.device, dtype=torch.bool)
        mask.scatter_(1, t_idx, False)
        s_idx = all_i[mask].view(BnW, n_src)
        
    elif strategy == 'grid':
        step = tpw / K
        t_base = (torch.arange(K, device=x.device).float() * step).long().clamp(0, tpw-1)
        t_idx = t_base.unsqueeze(0).expand(BnW, -1)
        all_i = torch.arange(tpw, device=x.device).expand(BnW, -1)
        mask = torch.ones(BnW, tpw, device=x.device, dtype=torch.bool)
        mask.scatter_(1, t_idx, False)
        s_idx = all_i[mask].view(BnW, n_src)
        
    elif strategy == 'attention':
        fn = F.normalize(flat, dim=-1)
        sim = torch.bmm(fn, fn.transpose(1, 2))
        centrality = sim.mean(dim=2)
        _, t_idx = torch.topk(centrality, K, dim=1, sorted=False)
        all_i = torch.arange(tpw, device=x.device).expand(BnW, -1)
        mask = torch.ones(BnW, tpw, device=x.device, dtype=torch.bool)
        mask.scatter_(1, t_idx, False)
        s_idx = all_i[mask].view(BnW, n_src)
    else:
        raise ValueError(strategy)
    
    # ---- Gather + Similarity match + Merge ----
    tgt = flat.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, C))
    src = flat.gather(1, s_idx.unsqueeze(-1).expand(-1, -1, C))
    
    tn = F.normalize(tgt, dim=-1)
    sn = F.normalize(src, dim=-1)
    sim = torch.bmm(sn, tn.transpose(1, 2))
    _, best_t = sim.max(dim=-1)
    
    sc = best_t.unsqueeze(-1).expand(-1, -1, C)
    merged = tgt.scatter_reduce(1, sc, src, reduce='mean')
    
    # Token sizes for PA
    token_sizes = None
    if track_sizes:
        counts = torch.ones(BnW, K, device=x.device)
        counts.scatter_add_(1, best_t, torch.ones(BnW, n_src, device=x.device))
        counts = counts.view(B, -1)[:, :target_tokens]
        token_sizes = torch.cat([torch.ones(B, 1, device=x.device), counts], dim=1)
    
    mp = merged.view(B, -1, C)[:, :target_tokens]
    return torch.cat([cls_token, mp], dim=1), token_sizes


class StrategyMergeBlock(nn.Module):
    """SRToMeBlock variant that accepts a selection strategy."""
    def __init__(self, block, target_tokens, window_size, strategy, prop_attn):
        super().__init__()
        self.block = block
        self.target_tokens = target_tokens
        self.window_size = window_size
        self.strategy = strategy
        self.prop_attn = prop_attn
        self._token_sizes = None
        
    def forward(self, x):
        x = self.block(x)
        x, sizes = merge_with_strategy(
            x, self.target_tokens, window_size=self.window_size,
            strategy=self.strategy, track_sizes=self.prop_attn)
        self._token_sizes = sizes
        return x


def create_strategy_model(strategy, target_tokens=84, late_start=8, window_size=7):
    """Create DeiT-S with the specified selection strategy."""
    from models.sr_tome import PropAttnBlock
    
    model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    blocks = model.blocks
    
    # Wrap merge layer
    merge_block = StrategyMergeBlock(
        blocks[late_start], target_tokens, window_size, strategy, prop_attn=True
    )
    blocks[late_start] = merge_block
    
    # Wrap post-merge layers with proportional attention
    for i in range(late_start + 1, len(blocks)):
        blocks[i] = PropAttnBlock(blocks[i], merge_block)
    
    return model


def run_selection_ablation(loader, seeds=3, max_batches=None):
    strategies = ['random', 'norm', 'norm_low', 'grid', 'attention']
    results = {}
    
    for strat in strategies:
        accs = []
        print(f"\n{'='*50}")
        print(f"Strategy: {strat}")
        print(f"{'='*50}")
        
        for seed in range(seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            model = create_strategy_model(strategy=strat)
            acc = evaluate(model, loader, max_batches=max_batches)
            accs.append(acc)
            print(f"  Seed {seed}: {acc:.2f}%")
            del model; torch.cuda.empty_cache()
        
        m, s = np.mean(accs), np.std(accs)
        results[strat] = {'mean': m, 'std': s, 'accs': accs}
        print(f"  → {strat}: {m:.2f}% ± {s:.2f}%")
    
    return results


# ============================================================
# PART 2: 5-seed runs
# ============================================================
def run_5seeds(loader, max_batches=None):
    from models.sr_tome import create_sr_tome_v3
    
    configs = [
        ("Baseline", None),
        ("S3-l8+PA", dict(target_tokens=84, late_start=8, window_size=7, prop_attn=True)),
        ("S3-l6+PA", dict(target_tokens=84, late_start=6, window_size=7, prop_attn=True)),
        ("S3-l8-noPA", dict(target_tokens=84, late_start=8, window_size=7, prop_attn=False)),
    ]
    
    results = {}
    for name, kw in configs:
        accs = []
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        
        for seed in range(5):
            torch.manual_seed(seed); np.random.seed(seed)
            if kw is None:
                model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
            else:
                model = create_sr_tome_v3(**kw).to(device).eval()
            acc = evaluate(model, loader, max_batches=max_batches)
            accs.append(acc)
            print(f"  Seed {seed}: {acc:.2f}%")
            del model; torch.cuda.empty_cache()
        
        m, s = np.mean(accs), np.std(accs)
        results[name] = {'mean': m, 'std': s, 'accs': accs}
        print(f"  → {name}: {m:.2f}% ± {s:.2f}%")
    
    # t-test
    if "Baseline" in results and "S3-l8+PA" in results:
        from scipy import stats
        t, p = stats.ttest_ind(results["Baseline"]['accs'], results["S3-l8+PA"]['accs'])
        print(f"\n  t-test Baseline vs S3-l8+PA: t={t:.3f}, p={p:.4f}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['selection', 'seeds', 'both'], default='both')
    parser.add_argument('--max_batches', type=int, default=None)
    args = parser.parse_args()
    
    print("Loading ImageNet val...")
    loader = get_loader()
    
    if args.mode in ['selection', 'both']:
        print("\n" + "="*60)
        print("TARGET SELECTION ABLATION")
        print("="*60)
        sel = run_selection_ablation(loader, seeds=3, max_batches=args.max_batches)
        
        print("\n" + "="*60)
        print("SELECTION SUMMARY")  
        print("="*60)
        print(f"{'Strategy':<15} {'Acc (mean±std)':<20} {'Δ vs Random'}")
        print("-"*55)
        rm = sel['random']['mean']
        for s, r in sorted(sel.items(), key=lambda x: -x[1]['mean']):
            print(f"{s:<15} {r['mean']:.2f}±{r['std']:.2f}%        {r['mean']-rm:+.2f}%")
    
    if args.mode in ['seeds', 'both']:
        print("\n" + "="*60)
        print("5-SEED CLASSIFICATION")
        print("="*60)
        sr = run_5seeds(loader, max_batches=args.max_batches)
        
        print("\n" + "="*60)
        print("5-SEED SUMMARY")
        print("="*60)
        for n, r in sr.items():
            print(f"  {n:<20} {r['mean']:.2f}±{r['std']:.2f}%")
    
    print("\nDone!")
