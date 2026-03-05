#!/usr/bin/env python3
"""
Full 50k ImageNet Validation with 3-Seed Repeats

Methods:
- Baseline (no merge)
- ToMe r=10, r=12
- SR-ToMe l6_t84, l8_t84

For stochastic methods, run 3 seeds and report mean ± std.
"""
import argparse
import time
import csv
from pathlib import Path

import torch
import numpy as np
import timm
import tome
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.sr_tome import create_sr_tome_v3


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


def evaluate(model, loader, max_batches=None, device='cuda'):
    model.eval()
    correct, total = 0, 0
    pbar = tqdm(loader, total=max_batches)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(pbar):
            if max_batches and i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})
    return 100.0 * correct / total


def time_model(model, batch_size, device, runs=100):
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def count_tokens(model, device):
    x = torch.randn(1, 3, 224, 224, device=device)
    token_counts = []
    def hook(m, i, o):
        token_counts.append(o.shape[1])
    handles = []
    blocks = model.model.blocks if hasattr(model, 'model') else model.blocks
    for blk in blocks:
        handles.append(blk.register_forward_hook(hook))
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    return token_counts[-1] if token_counts else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_batches', default=None, type=int, help='None = full 50k')
    parser.add_argument('--seeds', default=3, type=int)
    parser.add_argument('--output_dir', default='artifacts')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    
    loader = get_imagenet_loader(batch_size=64)
    total_batches = len(loader) if args.max_batches is None else args.max_batches
    results = []
    
    # ===== BASELINE (deterministic, 1 run) =====
    print("="*70)
    print("BASELINE")
    print("="*70)
    baseline = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    base_acc = evaluate(baseline, loader, args.max_batches)
    base_lat_b1 = time_model(baseline, 1, device)
    base_lat_b64 = time_model(baseline, 64, device)
    base_tokens = count_tokens(baseline, device)
    print(f"Accuracy: {base_acc:.4f}%, Tokens: {base_tokens}")
    print(f"B=1: {base_lat_b1:.2f}ms, B=64: {base_lat_b64:.2f}ms")
    results.append({
        'method': 'baseline',
        'acc_mean': base_acc, 'acc_std': 0,
        'drop_mean': 0, 'drop_std': 0,
        'lat_b1': base_lat_b1, 'lat_b64': base_lat_b64,
        'speedup': 1.0, 'tokens': base_tokens
    })
    del baseline
    torch.cuda.empty_cache()
    
    # ===== STOCHASTIC METHODS (3 seeds) =====
    configs = [
        # ToMe
        {'name': 'tome_r10', 'type': 'tome', 'r': 10},
        {'name': 'tome_r12', 'type': 'tome', 'r': 12},
        # SR-ToMe
        {'name': 'sr_tome_l6_t84', 'type': 'sr', 'late_start': 6, 'target': 84},
        {'name': 'sr_tome_l8_t84', 'type': 'sr', 'late_start': 8, 'target': 84},
    ]
    
    for cfg in configs:
        print(f"\n" + "="*70)
        print(f"{cfg['name'].upper()} ({args.seeds} seeds)")
        print("="*70)
        
        accs = []
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            if cfg['type'] == 'tome':
                model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
                tome.patch.timm(model)
                model.r = cfg['r']
            else:  # sr
                model = create_sr_tome_v3(
                    target_tokens=cfg['target'],
                    late_start=cfg['late_start'],
                    window_size=7
                ).to(device).eval()
            
            acc = evaluate(model, loader, args.max_batches)
            accs.append(acc)
            print(f"Seed {seed}: {acc:.4f}%")
            
            # Latency only on first seed
            if seed == 0:
                lat_b1 = time_model(model, 1, device)
                lat_b64 = time_model(model, 64, device)
                tokens = count_tokens(model, device)
            
            del model
            torch.cuda.empty_cache()
        
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        drop_mean = base_acc - acc_mean
        drop_std = acc_std
        speedup = base_lat_b64 / lat_b64
        
        print(f"\n{cfg['name']}: {acc_mean:.4f}% ± {acc_std:.4f}%")
        print(f"Drop: {drop_mean:.4f}% ± {drop_std:.4f}%")
        print(f"Tokens: {tokens}, Speedup: {speedup:.2f}x, B=1: {lat_b1:.2f}ms")
        
        results.append({
            'method': cfg['name'],
            'acc_mean': acc_mean, 'acc_std': acc_std,
            'drop_mean': drop_mean, 'drop_std': drop_std,
            'lat_b1': lat_b1, 'lat_b64': lat_b64,
            'speedup': speedup, 'tokens': tokens
        })
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("FULL 50k RESULTS (3-seed mean ± std)")
    print("="*70)
    print(f"{'Method':<20} {'Acc':<18} {'Drop':<18} {'B=1':<8} {'Speedup':<8} {'Tok':<4}")
    print("-"*76)
    for r in results:
        acc_str = f"{r['acc_mean']:.2f}±{r['acc_std']:.2f}" if r['acc_std'] > 0 else f"{r['acc_mean']:.2f}"
        drop_str = f"{r['drop_mean']:.2f}±{r['drop_std']:.2f}" if r['drop_std'] > 0 else f"{r['drop_mean']:.2f}"
        print(f"{r['method']:<20} {acc_str:<18} {drop_str:<18} {r['lat_b1']:<8.2f} {r['speedup']:<8.2f}x {r['tokens']:<4}")
    
    # ===== MATCHED-SPEED COMPARISON =====
    print("\n" + "="*70)
    print("MATCHED-SPEED COMPARISON (~1.2-1.3x)")
    print("="*70)
    srL6 = next(r for r in results if r['method'] == 'sr_tome_l6_t84')
    tm10 = next(r for r in results if r['method'] == 'tome_r10')
    tm12 = next(r for r in results if r['method'] == 'tome_r12')
    
    print(f"ToMe r=10 ({tm10['speedup']:.2f}x): {tm10['drop_mean']:.2f}% drop, B=1: {tm10['lat_b1']:.2f}ms")
    print(f"ToMe r=12 ({tm12['speedup']:.2f}x): {tm12['drop_mean']:.2f}% drop, B=1: {tm12['lat_b1']:.2f}ms")
    print(f"SR-ToMe l6 ({srL6['speedup']:.2f}x): {srL6['drop_mean']:.2f}% drop, B=1: {srL6['lat_b1']:.2f}ms")
    
    # B=1 advantage
    b1_advantage = tm10['lat_b1'] / srL6['lat_b1']
    print(f"\nB=1 Latency Advantage: SR-ToMe is {b1_advantage:.1f}x faster")
    
    # ===== SAVE =====
    csv_path = output_dir / 'sr_tome_50k_3seed.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")


if __name__ == '__main__':
    main()
