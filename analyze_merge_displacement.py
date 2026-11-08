#!/usr/bin/env python3
"""
Merge displacement analysis: compare spatial distances between merged tokens
for global (ToMe-style) vs windowed (S³-Merge) matching.

Produces a histogram figure showing that windowed merging bounds displacement.
Also runs CAMR ablation (content_adaptive on/off).
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Part 1: Merge Displacement Analysis
# ============================================================

def compute_merge_distances_global(patches, r=10, grid_size=14):
    """
    ToMe-style global even/odd merge. Returns spatial distances 
    between each source and its assigned target in grid units.
    """
    B, N, C = patches.shape
    H = W = grid_size
    
    # Even/odd split (ToMe's approach)
    even_idx = torch.arange(0, N, 2, device=patches.device)
    odd_idx = torch.arange(1, N, 2, device=patches.device)
    
    set_a = patches[:, even_idx]  # sources
    set_b = patches[:, odd_idx]   # targets
    
    # Cosine similarity
    a_norm = F.normalize(set_a, dim=-1)
    b_norm = F.normalize(set_b, dim=-1)
    sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # [B, |A|, |B|]
    
    # Each source finds its best target
    best_target = sim.argmax(dim=2)  # [B, |A|]
    
    # Only merge top-r most similar pairs
    best_sim = sim.gather(2, best_target.unsqueeze(-1)).squeeze(-1)  # [B, |A|]
    _, top_r_idx = best_sim.topk(r, dim=1)  # [B, r] — indices into even_idx
    
    # Compute spatial distances for merged pairs
    distances = []
    for b in range(B):
        for src_local in top_r_idx[b]:
            src_global = even_idx[src_local.item()].item()
            tgt_local = best_target[b, src_local.item()].item()
            tgt_global = odd_idx[tgt_local].item()
            
            # Grid positions
            src_row, src_col = src_global // W, src_global % W
            tgt_row, tgt_col = tgt_global // W, tgt_global % W
            
            dist = np.sqrt((src_row - tgt_row)**2 + (src_col - tgt_col)**2)
            distances.append(dist)
    
    return distances


def compute_merge_distances_windowed(patches, target_tokens=84, window_size=7, grid_size=14):
    """
    S³-Merge windowed merge. Returns spatial distances between 
    each source and its assigned target in grid units.
    """
    B, N, C = patches.shape
    H = W = grid_size
    ws = window_size
    nH = H // ws
    nW = W // ws
    n_windows = nH * nW
    tokens_per_window = ws * ws
    K = target_tokens // n_windows
    n_src = tokens_per_window - K
    
    grid = patches.view(B, H, W, C)
    windows = grid.view(B, nH, ws, nW, ws, C)
    windows = windows.permute(0, 1, 3, 2, 4, 5)  # [B, nH, nW, ws, ws, C]
    
    distances = []
    
    for b in range(B):
        for wh in range(nH):
            for ww in range(nW):
                win = windows[b, wh, ww]  # [ws, ws, C]
                flat = win.reshape(tokens_per_window, C)
                
                # Random target selection
                rs = torch.rand(tokens_per_window, device=patches.device)
                _, t_idx = torch.topk(rs, K, sorted=False)
                _, s_idx = torch.topk(rs, n_src, largest=False, sorted=False)
                
                tgt = flat[t_idx]  # [K, C]
                src = flat[s_idx]  # [n_src, C]
                
                # Similarity matching
                t_norm = F.normalize(tgt, dim=-1)
                s_norm = F.normalize(src, dim=-1)
                sim = torch.mm(s_norm, t_norm.t())  # [n_src, K]
                best_t = sim.argmax(dim=1)  # [n_src]
                
                # Compute distances in grid space
                for si in range(n_src):
                    s_local = s_idx[si].item()
                    t_local = t_idx[best_t[si].item()].item()
                    
                    # Local to global grid position
                    s_row = wh * ws + s_local // ws
                    s_col = ww * ws + s_local % ws
                    t_row = wh * ws + t_local // ws
                    t_col = ww * ws + t_local % ws
                    
                    dist = np.sqrt((s_row - t_row)**2 + (s_col - t_col)**2)
                    distances.append(dist)
    
    return distances


def plot_displacement_histogram(global_dists, windowed_dists, output_path):
    """Create publication-quality merge displacement histogram."""
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
    
    max_dist = max(max(global_dists), max(windowed_dists))
    bins = np.arange(0, max_dist + 1, 0.5)
    
    # Theoretical bound for w=7
    bound_w7 = np.sqrt(2) * (7 - 1)  # ≈ 8.49
    
    ax.hist(global_dists, bins=bins, alpha=0.55, color='#e74c3c', 
            label=f'Global (ToMe-style)\n(mean={np.mean(global_dists):.1f}, max={max(global_dists):.1f})',
            edgecolor='#c0392b', linewidth=0.5, density=True)
    ax.hist(windowed_dists, bins=bins, alpha=0.55, color='#2ecc71',
            label=f'Windowed ($w$=7, S³-Merge)\n(mean={np.mean(windowed_dists):.1f}, max={max(windowed_dists):.1f})',
            edgecolor='#27ae60', linewidth=0.5, density=True)
    
    # Theoretical bound
    ax.axvline(x=bound_w7, color='#2c3e50', linestyle='--', linewidth=1.5, 
               label=f'Prop. 1 bound: $\\sqrt{{2}}(w{{-}}1)$ = {bound_w7:.1f}')
    
    ax.set_xlabel('Merge displacement (grid units)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Spatial Displacement of Merged Token Pairs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_xlim(0, max_dist + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved displacement histogram to {output_path}")
    plt.close()


# ============================================================
# Part 2: CAMR Ablation
# ============================================================

def run_camr_ablation(model_base, loader, device, max_batches=None):
    """Compare fixed vs content-adaptive merge ratio."""
    from models.sr_tome import create_sr_tome_v3
    
    results = {}
    
    for camr in [False, True]:
        label = "CAMR" if camr else "Fixed"
        print(f"\n{'='*50}")
        print(f"Running: {label}")
        print(f"{'='*50}")
        
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = create_sr_tome_v3(
                target_tokens=84,
                late_start=8,
                window_size=7,
                content_adaptive=camr,
            ).to(device).eval()
            
            correct = total = 0
            with torch.no_grad():
                for i, (imgs, labels) in enumerate(loader):
                    if max_batches and i >= max_batches:
                        break
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    correct += (out.argmax(-1) == labels).sum().item()
                    total += labels.size(0)
            
            acc = 100 * correct / total
            accs.append(acc)
            print(f"  Seed {seed}: {acc:.2f}%")
            del model
            torch.cuda.empty_cache()
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[label] = {'mean': mean_acc, 'std': std_acc}
        print(f"  → {label}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['displacement', 'camr', 'both'], default='both')
    parser.add_argument('--n_images', type=int, default=200, help='Images for displacement analysis')
    parser.add_argument('--max_batches', type=int, default=None, help='Batches for CAMR (None=full 50k)')
    parser.add_argument('--output_dir', default='figures')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    
    if args.mode in ['displacement', 'both']:
        print("="*60)
        print("MERGE DISPLACEMENT ANALYSIS")
        print("="*60)
        
        # Load a few images for displacement analysis
        from torchvision import transforms
        from datasets import load_dataset
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            cache_dir=os.environ.get("IMAGENET_CACHE", "./data/imagenet_hf"),
        )
        
        all_global_dists = []
        all_windowed_dists = []
        
        n_imgs = min(args.n_images, len(dataset))
        print(f"Analyzing {n_imgs} images...")
        
        with torch.no_grad():
            for idx in range(n_imgs):
                if idx % 50 == 0:
                    print(f"  Image {idx}/{n_imgs}")
                
                img = dataset[idx]['image'].convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Get features at layer 8 (our default merge layer)
                # Hook into layer 7 output (0-indexed, so block[7] = layer 8)
                features = {}
                def hook_fn(module, input, output):
                    features['out'] = output
                
                handle = model.blocks[7].register_forward_hook(hook_fn)
                _ = model(img_tensor)
                handle.remove()
                
                # Extract patch tokens (remove CLS)
                patch_features = features['out'][:, 1:, :]  # [1, 196, 384]
                
                # Compute merge distances
                g_dists = compute_merge_distances_global(patch_features, r=10, grid_size=14)
                w_dists = compute_merge_distances_windowed(patch_features, target_tokens=84, window_size=7, grid_size=14)
                
                all_global_dists.extend(g_dists)
                all_windowed_dists.extend(w_dists)
        
        print(f"\nGlobal: {len(all_global_dists)} merges, mean={np.mean(all_global_dists):.2f}, max={max(all_global_dists):.2f}")
        print(f"Windowed: {len(all_windowed_dists)} merges, mean={np.mean(all_windowed_dists):.2f}, max={max(all_windowed_dists):.2f}")
        
        bound_w7 = np.sqrt(2) * 6
        violations = sum(1 for d in all_windowed_dists if d > bound_w7 + 0.01)
        print(f"Bound violations (windowed): {violations}/{len(all_windowed_dists)}")
        
        fig_path = output_dir / 'merge_displacement.pdf'
        plot_displacement_histogram(all_global_dists, all_windowed_dists, str(fig_path))
        
        # Also save as PNG for the paper
        fig_path_png = output_dir / 'merge_displacement.png'
        plot_displacement_histogram(all_global_dists, all_windowed_dists, str(fig_path_png))
    
    del model
    torch.cuda.empty_cache()
    
    if args.mode in ['camr', 'both']:
        print("\n" + "="*60)
        print("CAMR ABLATION")
        print("="*60)
        
        # Load ImageNet
        from torchvision import transforms, datasets
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use HuggingFace dataset with torch DataLoader
        from datasets import load_dataset
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            cache_dir=os.environ.get("IMAGENET_CACHE", "./data/imagenet_hf"),
            trust_remote_code=True
        )
        
        class HFImageNetWrapper(torch.utils.data.Dataset):
            def __init__(self, hf_dataset, transform):
                self.dataset = hf_dataset
                self.transform = transform
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                item = self.dataset[idx]
                img = item['image'].convert('RGB')
                return self.transform(img), item['label']
        
        wrapped = HFImageNetWrapper(dataset, transform)
        loader = torch.utils.data.DataLoader(
            wrapped, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
        )
        
        camr_results = run_camr_ablation(None, loader, device, max_batches=args.max_batches)
        
        print("\n" + "="*60)
        print("CAMR RESULTS SUMMARY")
        print("="*60)
        for label, r in camr_results.items():
            print(f"  {label}: {r['mean']:.2f}% ± {r['std']:.2f}%")
        
        if 'Fixed' in camr_results and 'CAMR' in camr_results:
            delta = camr_results['CAMR']['mean'] - camr_results['Fixed']['mean']
            print(f"  CAMR gain: {delta:+.2f}%")


if __name__ == '__main__':
    main()
