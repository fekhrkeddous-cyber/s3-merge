#!/usr/bin/env python3
"""
Window-size sweep: accuracy + latency for w ∈ {2, 7, 14} on DeiT-S 224².
Uses HuggingFace ImageNet-1K dataset (Arrow format).

Usage: python sweep_window_size.py [--max_batches 200]
"""
import argparse, csv, time, sys
import torch
import torch.nn.functional as F
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
                print(f"    batch {i+1}: acc so far = {100*correct/total:.2f}%")
    return correct / total * 100 if total > 0 else 0.0


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
    parser.add_argument('--target_tokens', type=int, default=84)
    parser.add_argument('--late_start', type=int, default=8)
    parser.add_argument('--n_iter', type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda')
    grid_size = 14

    window_sizes = [2, 7, 14]
    dataloader, n_images = get_imagenet_loader(args.batch_size)
    print(f"Dataset: {n_images} images")

    dummy_b1 = torch.randn(1, 3, 224, 224, device=device)
    dummy_b8 = torch.randn(8, 3, 224, 224, device=device)
    dummy_b64 = torch.randn(64, 3, 224, 224, device=device)

    results = []

    # ── Baseline ──
    print("\n=== Baseline ===")
    model = timm.create_model('deit_small_patch16_224', pretrained=True).eval().to(device)
    lat_b1 = measure_latency(model, dummy_b1, args.n_iter)
    lat_b8 = measure_latency(model, dummy_b8, args.n_iter)
    lat_b64 = measure_latency(model, dummy_b64, args.n_iter)
    acc = evaluate(model, dataloader, device, args.max_batches)
    thr_b64 = 64 / (lat_b64 / 1000)
    print(f"  Acc: {acc:.2f}% | B=1: {lat_b1:.2f}ms | B=8: {lat_b8:.2f}ms | B=64 thr: {thr_b64:.0f} img/s")
    results.append({
        'window_size': '--', 'method': 'Baseline',
        'acc': acc, 'lat_b1': lat_b1, 'lat_b8': lat_b8,
        'thr_b64': thr_b64, 'tokens': 196
    })
    del model
    torch.cuda.empty_cache()

    # ── S³-Merge with different window sizes ──
    for ws in window_sizes:
        label = f"w={ws}" if ws < 14 else "w=14 (global)"
        print(f"\n=== S³-Merge {label}, T={args.target_tokens}, l{args.late_start} ===")

        n_windows = (grid_size // ws) ** 2
        k_per_win = max(1, args.target_tokens // n_windows)
        actual_T = k_per_win * n_windows

        model = timm.create_model('deit_small_patch16_224', pretrained=True).eval().to(device)
        apply_patch(model,
                    target_tokens=args.target_tokens,
                    late_start=args.late_start,
                    window_size=ws)

        lat_b1 = measure_latency(model, dummy_b1, args.n_iter)
        lat_b8 = measure_latency(model, dummy_b8, args.n_iter)
        lat_b64 = measure_latency(model, dummy_b64, args.n_iter)
        acc = evaluate(model, dataloader, device, args.max_batches)
        thr_b64 = 64 / (lat_b64 / 1000)

        print(f"  Windows: {n_windows}, K/win: {k_per_win}, actual T: {actual_T}")
        print(f"  Acc: {acc:.2f}% | B=1: {lat_b1:.2f}ms | B=8: {lat_b8:.2f}ms | B=64 thr: {thr_b64:.0f} img/s")

        results.append({
            'window_size': ws, 'method': f'S3-Merge w={ws}',
            'acc': acc, 'lat_b1': lat_b1, 'lat_b8': lat_b8,
            'thr_b64': thr_b64, 'tokens': actual_T
        })

        del model
        torch.cuda.empty_cache()

    # ── Save CSV ──
    out_dir = Path('results_downstream')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / 'window_size_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['window_size', 'method', 'acc',
                                                'lat_b1', 'lat_b8', 'thr_b64', 'tokens'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # ── Print summary ──
    print("\n" + "=" * 80)
    print(f"{'Method':25s} {'w':>4s} {'Tok':>5s} {'Acc':>7s} {'B=1 ms':>8s} {'B=8 ms':>8s} {'B=64 thr':>10s}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:25s} {str(r['window_size']):>4s} {r['tokens']:5d} {r['acc']:7.2f}% "
              f"{r['lat_b1']:8.2f} {r['lat_b8']:8.2f} {r['thr_b64']:10.0f}")


if __name__ == '__main__':
    main()
