#!/usr/bin/env python3
"""
Model-scale study: evaluate S³-Merge on ViT-B (vs DeiT-S baseline in main paper).
Tests our default configuration on a larger model.

Usage: python model_scale_study.py [--max_batches None]

HISTORICAL SCRIPT, kept for reference. The paper's Table 7 ViT-B/16 row comes from a historical run of this
script with an earlier implementation that had no proportional attention. For that configuration use
r3/table7_vitb16/run_table7_vitb16.py (explicit, asserted PA-off). Here PA is set explicitly to off as well.
"""
import os
import argparse, csv, time, sys
import torch
import timm
from pathlib import Path
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.sr_tome import apply_patch


def get_imagenet_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = load_dataset("ILSVRC/imagenet-1k", split='validation',
                           cache_dir=os.environ.get("IMAGENET_CACHE", "./data/imagenet_hf"))
    def collate(batch):
        imgs = [transform(x['image'].convert('RGB')) for x in batch]
        labels = [x['label'] for x in batch]
        return torch.stack(imgs), torch.tensor(labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, collate_fn=collate), len(dataset)


def evaluate(model, dataloader, device, max_batches=None):
    correct = total = 0
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, (images, targets) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            if (i + 1) % 50 == 0:
                print(f"    batch {i+1}: acc = {100*correct/total:.2f}%")
    return correct / total * 100


def measure_latency(model, dummy_input, n_iter=200, warmup=30):
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(warmup):
            model(dummy_input)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(dummy_input)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--n_iter', type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda')
    dataloader, n_images = get_imagenet_loader(args.batch_size)
    print(f"Dataset: {n_images} images")

    # Model configs: (model_name, display_name, grid_size)
    configs = [
        ('deit_small_patch16_224', 'DeiT-S', 14),
        ('vit_base_patch16_224', 'ViT-B/16', 14),
    ]

    # S³-Merge configs to test
    merge_configs = [
        # (target_tokens, late_start, window_size, label)
        (84, 8, 7, 'l8 r≈0.43'),
        (137, 8, 7, 'l8 r≈0.70'),
    ]

    dummy_b1 = torch.randn(1, 3, 224, 224, device=device)
    dummy_b8 = torch.randn(8, 3, 224, 224, device=device)
    dummy_b64 = torch.randn(64, 3, 224, 224, device=device)

    results = []

    for model_name, display_name, grid_size in configs:
        print(f"\n{'='*60}")
        print(f"Model: {display_name} ({model_name})")
        print(f"{'='*60}")

        # ── Baseline ──
        print(f"\n--- {display_name} Baseline ---")
        model = timm.create_model(model_name, pretrained=True).eval().to(device)
        lat_b1 = measure_latency(model, dummy_b1, args.n_iter)
        lat_b8 = measure_latency(model, dummy_b8, args.n_iter)
        lat_b64 = measure_latency(model, dummy_b64, args.n_iter)
        acc = evaluate(model, dataloader, device, args.max_batches)
        thr_b64 = 64 / (lat_b64 / 1000)
        print(f"  Acc: {acc:.2f}% | B=1: {lat_b1:.2f}ms | B=8: {lat_b8:.2f}ms | B=64 thr: {thr_b64:.0f} img/s")
        results.append({
            'model': display_name, 'config': 'Baseline',
            'acc': acc, 'delta_acc': 0.0,
            'lat_b1': lat_b1, 'lat_b8': lat_b8,
            'thr_b64': thr_b64
        })
        baseline_acc = acc
        del model
        torch.cuda.empty_cache()

        # ── S³-Merge configs ──
        for target_tokens, late_start, ws, label in merge_configs:
            print(f"\n--- {display_name} + S³-Merge {label} ---")
            model = timm.create_model(model_name, pretrained=True).eval().to(device)
            apply_patch(model,
                        target_tokens=target_tokens,
                        late_start=late_start,
                        window_size=ws,
                        prop_attn=False)

            lat_b1 = measure_latency(model, dummy_b1, args.n_iter)
            lat_b8 = measure_latency(model, dummy_b8, args.n_iter)
            lat_b64 = measure_latency(model, dummy_b64, args.n_iter)
            acc = evaluate(model, dataloader, device, args.max_batches)
            thr_b64 = 64 / (lat_b64 / 1000)

            print(f"  Acc: {acc:.2f}% (Δ={acc-baseline_acc:+.2f}%) | B=1: {lat_b1:.2f}ms | "
                  f"B=8: {lat_b8:.2f}ms | B=64 thr: {thr_b64:.0f} img/s")
            results.append({
                'model': display_name, 'config': f'S³-Merge {label}',
                'acc': acc, 'delta_acc': acc - baseline_acc,
                'lat_b1': lat_b1, 'lat_b8': lat_b8,
                'thr_b64': thr_b64
            })
            del model
            torch.cuda.empty_cache()

    # ── Save CSV ──
    out_dir = Path('results_downstream')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / 'model_scale_study.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'config', 'acc', 'delta_acc',
                                                'lat_b1', 'lat_b8', 'thr_b64'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # ── Summary ──
    print("\n" + "=" * 90)
    print(f"{'Model':12s} {'Config':22s} {'Acc':>7s} {'ΔAcc':>7s} {'B=1 ms':>8s} {'B=8 ms':>8s} {'B=64 thr':>10s}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:12s} {r['config']:22s} {r['acc']:7.2f}% {r['delta_acc']:+7.2f}% "
              f"{r['lat_b1']:8.2f} {r['lat_b8']:8.2f} {r['thr_b64']:10.0f}")


if __name__ == '__main__':
    main()
