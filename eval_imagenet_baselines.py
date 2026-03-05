#!/usr/bin/env python3
"""
Unified Baseline Evaluation for Modern Token Merging Methods

Evaluates all training-free baselines under identical conditions:
- Model: DeiT-S/16 (pretrained, timm)
- Dataset: ImageNet-1K val (50k images)
- Timing: warmup=20, measure=100, 3 seeds for stochastic methods  
- Batch sizes: B ∈ {1, 8, 32, 64}

Methods:
- Baseline (no merge)
- ToMe r=10, r=12
- S³-Merge l6_t84, l8_t84
- PiToMe (NeurIPS 2024)
- ATC (ECCV 2024)
- DSM (NeurIPS 2024 Workshop, re-implementation)

Outputs:
- baselines/results/imagenet_baselines.csv
- baselines/results/batch_curve_baselines.csv
- baselines/results/eval_log.txt (stdout for reproducibility)
"""
import argparse
import time
import csv
import sys
import json
import os
import importlib.util
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import timm
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Optional imports for existing baselines (may not be installed in all environments)
try:
    import tome
except ImportError:
    tome = None

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
try:
    from models.sr_tome import create_sr_tome_v3
except ImportError:
    create_sr_tome_v3 = None


# ============================================================
# Data loading (identical to benchmark_50k_3seed.py)
# ============================================================

def get_imagenet_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = load_dataset(
        "ILSVRC/imagenet-1k", split='validation',
        cache_dir=os.environ.get("IMAGENET_CACHE", "./data/imagenet_hf")
    )
    def collate(batch):
        imgs = [transform(x['image'].convert('RGB')) for x in batch]
        labels = [x['label'] for x in batch]
        return torch.stack(imgs), torch.tensor(labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, collate_fn=collate)


# ============================================================
# Evaluation & timing (identical to benchmark_50k_3seed.py)
# ============================================================

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
            # Handle PiToMe which returns (logits, flops) tuple
            if isinstance(out, tuple):
                out = out[0]
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})
    return 100.0 * correct / total


def time_model(model, batch_size, device, warmup=20, runs=100):
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            out = model(x)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            out = model(x)
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def count_tokens(model, device):
    """Count tokens at the last block output."""
    x = torch.randn(1, 3, 224, 224, device=device)
    token_counts = []
    def hook(m, i, o):
        if isinstance(o, tuple):
            token_counts.append(o[0].shape[1])
        else:
            token_counts.append(o.shape[1])
    handles = []
    blocks = getattr(model, 'blocks', None)
    if blocks is None and hasattr(model, 'model'):
        blocks = model.model.blocks
    if blocks is None:
        return -1
    for blk in blocks:
        handles.append(blk.register_forward_hook(hook))
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    return token_counts[-1] if token_counts else -1


# ============================================================
# Model factory
# ============================================================

def create_model(cfg, device):
    """Create a model based on config dict."""
    method = cfg['type']

    if method == 'baseline':
        model = timm.create_model('deit_small_patch16_224', pretrained=True)
        return model.to(device).eval()

    elif method == 'tome':
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        tome.patch.timm(model)
        model.r = cfg['r']
        return model

    elif method == 'sr':
        model = create_sr_tome_v3(
            target_tokens=cfg['target'],
            late_start=cfg['late_start'],
            window_size=7
        )
        return model.to(device).eval()

    elif method == 'pitome':
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        # Use importlib for our timm 1.0-compatible adapter
        _pitome_path = ROOT / 'baselines' / 'pitome_adapter.py'
        _spec = importlib.util.spec_from_file_location('pitome_adapter', _pitome_path)
        _pitome_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_pitome_mod)
        _pitome_mod.apply_pitome_patch(model, trace_source=False, prop_attn=False)
        model.ratio = cfg['ratio']
        return model

    elif method == 'atc':
        # ATC builds its own model class; use importlib to avoid collision with project's models/
        _atc_path = ROOT / 'baselines' / 'ATC' / 'ATC_setup' / 'models' / 'atc.py'
        _spec = importlib.util.spec_from_file_location('atc_model', _atc_path)
        _atc_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_atc_mod)
        import argparse as _ap

        reduction_loc = cfg.get('reduction_loc', list(range(12)))
        args = _ap.Namespace(
            reduction_ratio=[cfg['reduction_ratio']],
            reduction_loc=reduction_loc,
            linkage=cfg.get('linkage', 'single'),
            proportional_attn=True,
        )
        atc_model = _atc_mod.AgglomerativeClusteringVisionTransformer(
            img_size=224, patch_size=16, embed_dim=384, depth=12,
            num_heads=6, mlp_ratio=4., qkv_bias=True,
            num_classes=1000, args=args,
        )
        # Load pretrained DeiT-S weights
        pretrained = timm.create_model('deit_small_patch16_224', pretrained=True)
        state_dict = pretrained.state_dict()
        missing, unexpected = atc_model.load_state_dict(state_dict, strict=False)
        print(f"  ATC weight loading: {len(missing)} missing, {len(unexpected)} unexpected")
        del pretrained
        return atc_model.to(device).eval()

    elif method == 'dsm':
        model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
        _dsm_path = ROOT / 'baselines' / 'dsm_impl.py'
        _spec = importlib.util.spec_from_file_location('dsm_impl', _dsm_path)
        _dsm_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_dsm_mod)
        _dsm_mod.apply_dsm_patch(model,
                        delay_layer=cfg.get('delay_layer', 6),
                        merge_r=cfg.get('merge_r', 12),
                        window_size=cfg.get('window_size', 7))
        return model

    else:
        raise ValueError(f"Unknown method type: {method}")


# ============================================================
# Configurations
# ============================================================

def get_configs(mode='full'):
    """
    Get baseline configurations.
    mode='smoke' → small subset for testing
    mode='full' → all methods
    """
    configs = [
        # ===== Existing baselines =====
        {'name': 'baseline', 'type': 'baseline', 'stochastic': False},
        {'name': 'tome_r10', 'type': 'tome', 'r': 10, 'stochastic': True},
        {'name': 'tome_r12', 'type': 'tome', 'r': 12, 'stochastic': True},
        {'name': 's3merge_l6_t84', 'type': 'sr', 'late_start': 6, 'target': 84, 'stochastic': True},
        {'name': 's3merge_l8_t84', 'type': 'sr', 'late_start': 8, 'target': 84, 'stochastic': True},

        # ===== Modern baselines (2024+) =====
        # PiToMe: ratio = fraction of tokens to keep per layer
        # DeiT-S has 196 patches, 12 blocks
        # ratio=0.9 → ~0.9^12 * 196 ≈ 55 tokens (aggressive)
        # ratio=0.95 → ~0.95^12 * 196 ≈ 107 tokens (mild)
        # We want to match ~84 tokens: ratio ≈ 0.93 → 0.93^12 * 196 ≈ 83
        {'name': 'pitome_r93', 'type': 'pitome', 'ratio': 0.93, 'stochastic': False},
        {'name': 'pitome_r90', 'type': 'pitome', 'ratio': 0.90, 'stochastic': False},

        # ATC: reduction_ratio is used to compute per-loc token counts as:
        #   int(196 * ratio^(idx+1)) for each loc idx
        # ratio=0.65, loc=[6,9]: [int(196*0.65^1), int(196*0.65^2)] = [127, 82] → ~82 final tokens
        # ratio=0.75, loc=[6,9]: [int(196*0.75^1), int(196*0.75^2)] = [147, 110] → ~110 final tokens
        {'name': 'atc_r65_2loc', 'type': 'atc', 'reduction_ratio': 0.65,
         'reduction_loc': [6, 9], 'linkage': 'single', 'stochastic': False},
        {'name': 'atc_r75_2loc', 'type': 'atc', 'reduction_ratio': 0.75,
         'reduction_loc': [6, 9], 'linkage': 'single', 'stochastic': False},

        # DSM (re-implementation): delay merging until layer 6, then merge r tokens/layer
        # delay=6, r=12 → merge 12 tokens/layer for layers 6-11 (6 layers × 12 = 72 merged)
        # Final: 196 - 72 = 124 tokens + CLS
        # For ~84: delay=6, r=19 → 6 × 19 = 114 merged → 196-114 = 82
        {'name': 'dsm_d6_r19', 'type': 'dsm', 'delay_layer': 6, 'merge_r': 19, 'stochastic': False},
        {'name': 'dsm_d4_r12', 'type': 'dsm', 'delay_layer': 4, 'merge_r': 12, 'stochastic': False},
    ]
    return configs


# ============================================================
# Main evaluation loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Unified baseline evaluation')
    parser.add_argument('--max_batches', default=None, type=int,
                        help='Limit batches for smoke testing (None = full 50k)')
    parser.add_argument('--seeds', default=3, type=int)
    parser.add_argument('--output_dir', default='baselines/results')
    parser.add_argument('--batch_sizes', nargs='+', type=int,
                        default=[1, 8, 32, 64])
    parser.add_argument('--methods', nargs='*', default=None,
                        help='Specific methods to run (default: all)')
    parser.add_argument('--skip_batch_curve', action='store_true',
                        help='Skip batch scaling curve measurement')
    args = parser.parse_args()

    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_file = output_dir / f'eval_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    class Tee:
        def __init__(self, logfile):
            self.file = open(logfile, 'w')
            self.stdout = sys.stdout
        def write(self, msg):
            self.stdout.write(msg)
            self.file.write(msg)
        def flush(self):
            self.stdout.flush()
            self.file.flush()

    tee = Tee(log_file)
    sys.stdout = tee

    print(f"={'='*70}")
    print(f"UNIFIED BASELINE EVALUATION")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Max batches: {args.max_batches or 'ALL (50k)'}")
    print(f"Seeds: {args.seeds}")
    print(f"Batch sizes for curves: {args.batch_sizes}")
    print(f"={'='*70}\n")

    loader = get_imagenet_loader(batch_size=64)
    total_batches = len(loader) if args.max_batches is None else args.max_batches

    configs = get_configs()
    if args.methods:
        configs = [c for c in configs if c['name'] in args.methods]

    all_results = []
    batch_curve_results = []

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"{cfg['name'].upper()} (type={cfg['type']})")
        print(f"{'='*70}")

        n_seeds = args.seeds if cfg.get('stochastic', False) else 1
        accs = []

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)

            try:
                model = create_model(cfg, device)
            except Exception as e:
                print(f"ERROR creating model: {e}")
                import traceback
                traceback.print_exc()
                break

            try:
                acc = evaluate(model, loader, args.max_batches, device=device)
                accs.append(acc)
                print(f"Seed {seed}: {acc:.4f}%")
            except Exception as e:
                print(f"ERROR during evaluation: {e}")
                import traceback
                traceback.print_exc()
                break

            # Measure latency and tokens on first seed only
            if seed == 0:
                try:
                    tokens = count_tokens(model, device)
                    print(f"Final tokens: {tokens}")

                    lat_b1 = time_model(model, 1, device)
                    lat_b64 = time_model(model, 64, device)
                    print(f"Latency: B=1 {lat_b1:.2f}ms, B=64 {lat_b64:.2f}ms")

                    # Batch scaling curve
                    if not args.skip_batch_curve:
                        for bs in args.batch_sizes:
                            lat = time_model(model, bs, device)
                            throughput = bs / (lat / 1000.0)
                            batch_curve_results.append({
                                'method': cfg['name'],
                                'batch_size': bs,
                                'latency_ms': round(lat, 3),
                                'throughput_ips': round(throughput, 1),
                            })
                            print(f"  B={bs}: {lat:.2f}ms ({throughput:.0f} img/s)")
                except Exception as e:
                    print(f"ERROR during timing: {e}")
                    import traceback
                    traceback.print_exc()
                    lat_b1 = lat_b64 = tokens = -1

            del model
            torch.cuda.empty_cache()

        if accs:
            base_acc = accs[0] if cfg['name'] == 'baseline' else None
            result = {
                'method': cfg['name'],
                'type': cfg['type'],
                'acc_mean': round(float(np.mean(accs)), 4),
                'acc_std': round(float(np.std(accs)), 4) if len(accs) > 1 else 0.0,
                'n_seeds': len(accs),
                'tokens': tokens,
                'lat_b1_ms': round(lat_b1, 3) if lat_b1 > 0 else -1,
                'lat_b64_ms': round(lat_b64, 3) if lat_b64 > 0 else -1,
            }
            all_results.append(result)
            print(f"\nRESULT: {result}")

    # Save results
    if all_results:
        csv_path = output_dir / 'imagenet_baselines.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {csv_path}")

    if batch_curve_results:
        csv_path = output_dir / 'batch_curve_baselines.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=batch_curve_results[0].keys())
            w.writeheader()
            w.writerows(batch_curve_results)
        print(f"Saved: {csv_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Acc (%)':<15} {'Tokens':<8} {'B=1 (ms)':<10} {'B=64 (ms)':<10}")
    print("-" * 70)
    for r in all_results:
        acc_str = f"{r['acc_mean']:.2f}±{r['acc_std']:.2f}" if r['acc_std'] > 0 else f"{r['acc_mean']:.2f}"
        print(f"{r['method']:<25} {acc_str:<15} {r['tokens']:<8} {r['lat_b1_ms']:<10} {r['lat_b64_ms']:<10}")

    print(f"\n✅ Evaluation complete. Results saved to {output_dir}/")
    sys.stdout = tee.stdout
    tee.file.close()


if __name__ == '__main__':
    main()
