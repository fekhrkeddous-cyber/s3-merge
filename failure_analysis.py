#!/usr/bin/env python3
"""
Qualitative failure analysis: find images where S³-Merge drops confidence
most significantly compared to baseline.
Uses HuggingFace ImageNet-1K dataset.

Usage: python failure_analysis.py [--max_batches 200] [--top_k 16]
"""
import argparse, sys, os
import torch
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    vis_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = load_dataset("ILSVRC/imagenet-1k", split='validation',
                           cache_dir=os.environ.get("IMAGENET_CACHE", "./data/imagenet_hf"))

    def collate(batch):
        imgs = [transform(x['image'].convert('RGB')) for x in batch]
        labels = [x['label'] for x in batch]
        return torch.stack(imgs), torch.tensor(labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate)
    return dataset, loader, vis_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_batches', type=int, default=200)
    parser.add_argument('--target_tokens', type=int, default=84)
    parser.add_argument('--late_start', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=7)
    parser.add_argument('--output', type=str, default='figures/failure_analysis.pdf')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hf_dataset, dataloader, vis_transform = get_imagenet_loader(args.batch_size)
    print(f"Dataset: {len(hf_dataset)} images, max_batches={args.max_batches}")

    # ── Baseline model ──
    print("Running baseline...")
    model_base = timm.create_model('deit_small_patch16_224', pretrained=True).eval().to(device)

    all_base_probs = []
    all_labels = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, (images, targets) in enumerate(dataloader):
            if args.max_batches and i >= args.max_batches:
                break
            images = images.to(device, non_blocking=True)
            logits = model_base(images)
            probs = F.softmax(logits, dim=-1)
            target_probs = probs[torch.arange(len(targets)), targets.to(device)]
            all_base_probs.append(target_probs.cpu())
            all_labels.append(targets)
            if (i + 1) % 50 == 0:
                print(f"  baseline batch {i+1}/{args.max_batches}")

    all_base_probs = torch.cat(all_base_probs)
    all_labels = torch.cat(all_labels)
    print(f"  Evaluated {len(all_base_probs)} images")

    del model_base
    torch.cuda.empty_cache()

    # ── S³-Merge model ──
    print("Running S³-Merge...")
    model_sr = timm.create_model('deit_small_patch16_224', pretrained=True).eval().to(device)
    apply_patch(model_sr,
                target_tokens=args.target_tokens,
                late_start=args.late_start,
                window_size=args.window_size)

    all_sr_probs = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, (images, targets) in enumerate(dataloader):
            if args.max_batches and i >= args.max_batches:
                break
            images = images.to(device, non_blocking=True)
            logits = model_sr(images)
            probs = F.softmax(logits, dim=-1)
            target_probs = probs[torch.arange(len(targets)), targets.to(device)]
            all_sr_probs.append(target_probs.cpu())
            if (i + 1) % 50 == 0:
                print(f"  S³-Merge batch {i+1}/{args.max_batches}")

    all_sr_probs = torch.cat(all_sr_probs)
    print(f"  Evaluated {len(all_sr_probs)} images")

    del model_sr
    torch.cuda.empty_cache()

    # ── Find biggest confidence drops ──
    confidence_drop = all_base_probs - all_sr_probs
    base_correct = (all_base_probs > 0.5)

    drop_scores = confidence_drop.clone()
    drop_scores[~base_correct] = -1

    top_k = min(args.top_k, len(drop_scores))
    top_indices = drop_scores.topk(top_k).indices

    print(f"\nTop {top_k} confidence drops:")
    print(f"{'Rank':>5s} {'Idx':>6s} {'Base prob':>10s} {'SR prob':>10s} {'Drop':>10s} {'Label':>8s}")
    print("-" * 55)
    for rank, idx in enumerate(top_indices):
        idx_val = idx.item()
        print(f"{rank+1:5d} {idx_val:6d} {all_base_probs[idx_val]:10.4f} {all_sr_probs[idx_val]:10.4f} "
              f"{confidence_drop[idx_val]:10.4f} {all_labels[idx_val].item():8d}")

    # ── Visualize ──
    n_show = min(top_k, 16)
    cols = 4
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for rank in range(n_show):
        ax = axes[rank // cols, rank % cols]
        idx_val = top_indices[rank].item()

        # Load image from HF dataset
        sample = hf_dataset[idx_val]
        img_pil = sample['image'].convert('RGB')
        img_tensor = vis_transform(img_pil)
        img_np = img_tensor.permute(1, 2, 0).numpy()

        ax.imshow(img_np)
        base_p = all_base_probs[idx_val].item()
        sr_p = all_sr_probs[idx_val].item()
        drop = confidence_drop[idx_val].item()

        ax.set_title(f"#{rank+1}: Base={base_p:.2f} → SR={sr_p:.2f}\nΔ={drop:.2f} (cls {all_labels[idx_val].item()})",
                     fontsize=11, fontweight='bold', color='#D32F2F')
        ax.axis('off')

    for rank in range(n_show, rows * cols):
        axes[rank // cols, rank % cols].axis('off')

    plt.suptitle(f'Top-{n_show} Confidence Drops (S³-Merge l{args.late_start}, T={args.target_tokens}, w={args.window_size})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.output}")

    png_path = args.output.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_path}")

    # ── Statistics ──
    print(f"\n=== Overall Statistics ({len(all_base_probs)} images) ===")
    print(f"Mean confidence drop:   {confidence_drop.mean():.4f}")
    print(f"Median confidence drop: {confidence_drop.median():.4f}")
    print(f"Std confidence drop:    {confidence_drop.std():.4f}")
    print(f"Max confidence drop:    {confidence_drop.max():.4f}")
    n_flipped = ((all_base_probs > 0.5) & (all_sr_probs < 0.5)).sum().item()
    print(f"Predictions flipped (conf>0.5 → <0.5): {n_flipped}/{len(all_base_probs)} "
          f"({n_flipped/len(all_base_probs)*100:.2f}%)")

    # ── Histogram of confidence changes ──
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    ax2.hist(confidence_drop.numpy(), bins=100, alpha=0.8, color='steelblue', edgecolor='white')
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No change')
    ax2.set_xlabel('Confidence drop (baseline − S³-Merge)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution of confidence changes ({len(all_base_probs)} images)')
    ax2.legend()
    plt.tight_layout()
    hist_path = args.output.replace('.pdf', '_histogram.pdf')
    plt.savefig(hist_path, dpi=150)
    plt.savefig(hist_path.replace('.pdf', '.png'), dpi=150)
    print(f"Saved: {hist_path}")


if __name__ == '__main__':
    main()
