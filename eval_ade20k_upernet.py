#!/usr/bin/env python3
"""
ADE20K UPerNet Evaluation with SR-ToMe
"""
import argparse
import os
import sys
import time
import csv
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mmseg
    from mmseg.apis import init_model, inference_model
    from mmengine.evaluator import Evaluator
    from mmengine.runner import Runner
    from mmengine.config import Config
    MMSEG_AVAILABLE = True
except ImportError:
    MMSEG_AVAILABLE = False
    print("mmseg not available. Run inside Docker container.")

# =============================================================================
# SR-ToMe Patching Logic for MMSeg ViT
# =============================================================================
try:
    from models.sr_tome_downstream import sr_tome_merge_with_unmerge_cached, sr_tome_unmerge, GridMappingCache
    SRTOME_AVAILABLE = True
except ImportError:
    SRTOME_AVAILABLE = False
    print("Warning: sr_tome_downstream not importable. S3-Merge variants will be skipped.")

class SRToMeManager:
    def __init__(self):
        self.merged = False
        self.restore_idx = None

class SRToMeBlockMMSeg(nn.Module):
    """
    Wraps an MMSeg ViT Layer to apply S³-Merge.
    """
    def __init__(self, layer, ratio, window_size, grid_size, manager):
        super().__init__()
        self.layer = layer # This is the original TransformerEncoderLayer
        self.ratio = ratio
        self.window_size = window_size
        self.grid_size = grid_size
        self.manager = manager 

    def forward(self, x):
        # Original block forward
        x = self.layer(x)
        
        # Apply merge if not already merged
        if not self.manager.merged:
            B, N, C = x.shape
            target_tokens = int(N * self.ratio)
            
            cache = GridMappingCache.get(self.grid_size, self.window_size, x.device)
            
            x, restore_idx = sr_tome_merge_with_unmerge_cached(
                x, target_tokens, cache
            )
            
            self.manager.restore_idx = restore_idx
            self.manager.merged = True
            
        return x

def apply_srtome_mmseg(model, ratio=0.7, late_start=6, window_size=7):
    """
    Patches MMSeg ViT model with S³-Merge.
    """
    backbone = model.backbone
    if hasattr(backbone, 'layers'):
        layers = backbone.layers
    else:
        print("Warning: Could not find backbone.layers.")
        return model

    manager = SRToMeManager()
    
    # Patch the specific layer
    if hasattr(layers[late_start], 'manager'):
        return model
        
    original_layer = layers[late_start]
    
    # ADE20K crop size 512x512 -> 32x32 grid (patch 16)
    grid_size = 32
    
    layers[late_start] = SRToMeBlockMMSeg(
        original_layer, 
        ratio=ratio,
        window_size=window_size,
        grid_size=grid_size,
        manager=manager
    )
    
    # Monkey-patch backbone forward
    if hasattr(backbone, '_original_forward'):
        original_forward = backbone._original_forward
    else:
        original_forward = backbone.forward
        backbone._original_forward = original_forward
    
    def patched_forward(x):
        manager.merged = False
        manager.restore_idx = None
        
        if backbone.with_cls_token:
            cls_token = backbone.cls_token.expand(x.shape[0], -1, -1)
            
        x, embed_hw = backbone.patch_embed(x)
        if backbone.with_cls_token:
            x = torch.cat((cls_token, x), dim=1)
        if backbone.pos_embed is not None:
            # Interpolate pos_embed to match actual sequence length
            pos_embed = backbone.pos_embed  # [1, 1+H0*W0, C]
            if pos_embed.shape[1] != x.shape[1]:
                # Split cls and patch pos embeds
                cls_pe = pos_embed[:, :1, :]   # [1, 1, C]
                patch_pe = pos_embed[:, 1:, :] # [1, H0*W0, C]
                # Stored grid size
                H0 = W0 = int(patch_pe.shape[1] ** 0.5)
                # Target grid size
                H_new, W_new = embed_hw if isinstance(embed_hw, (tuple, list)) else (int(x.shape[1] - 1) ** 0.5, int(x.shape[1] - 1) ** 0.5)
                if isinstance(embed_hw, int):
                    H_new = W_new = int((x.shape[1] - 1) ** 0.5)
                # Interpolate
                patch_pe = patch_pe.reshape(1, H0, W0, -1).permute(0, 3, 1, 2)
                patch_pe = torch.nn.functional.interpolate(
                    patch_pe, size=(H_new, W_new), mode='bicubic', align_corners=False)
                patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, H_new * W_new, -1)
                pos_embed = torch.cat([cls_pe, patch_pe], dim=1)
            x = x + backbone.drop_after_pos(pos_embed)


        # Iterate layers
        outs = []
        for i, layer in enumerate(backbone.layers):
            x = layer(x)
            
            if i == len(backbone.layers) - 1 and backbone.final_norm:
                x = backbone.norm1(x)
                
            if i in backbone.out_indices:
                # We need to reshape/unmerge here
                # If merged, unmerge first to restore grid
                out_x = x
                if manager.restore_idx is not None:
                     # Remove CLS if present?
                     # sr_tome_unmerge expects [B, 1+K, C] input? 
                     # Yes, `merged` arg has CLS.
                     # But `unmerge` returns [B, C, H, W] (without CLS).
                     # MMSeg ViT usually handles CLS handling inside `reshape`?
                     # Or does it strip CLS?
                     # Standard MMSeg ViT:
                     # if self.with_cls_token:
                     #    out = x[:, 1:]
                     # else: out = x
                     # B, N, C = out.shape
                     # out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
                     
                     # Using our unmerge:
                     dense = sr_tome_unmerge(out_x, manager.restore_idx, grid_size=grid_size)
                     outs.append(dense)
                else:
                     # Standard reshape
                     if backbone.with_cls_token:
                         out = out_x[:, 1:]
                     else:
                         out = out_x
                     B, N, C = out.shape
                     H = W = int(N**0.5)
                     out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                     outs.append(out)
                     
        return tuple(outs)

    backbone.forward = patched_forward
    return model

# =============================================================================
# Benchmarking Utils
# =============================================================================

def measure_time(model, batch_size=1, n_warmup=10, n_runs=100, mode='latency'):
    """Measure forward time (latency or throughput)."""
    device = "cuda"
    model.eval()
    H, W = 512, 512
    x = torch.randn(batch_size, 3, H, W, device=device)
    # Provide required metadata for MMSeg
    batch_img_metas = [{'img_shape': (H, W, 3), 'ori_shape': (H, W, 3), 'pad_shape': (H, W, 3), 'scale_factor': 1.0} for _ in range(batch_size)]

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.encode_decode(x, batch_img_metas)
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model.encode_decode(x, batch_img_metas)
            torch.cuda.synchronize()
            
    elapsed_sec = (time.perf_counter() - t0)
    
    if mode == 'latency':
        return (elapsed_sec * 1000) / n_runs # ms / img
    else:
        return (batch_size * n_runs) / elapsed_sec # img / s

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/mmseg/vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py')
    parser.add_argument('--checkpoint', default='https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_160k_ade20k/upernet_vit-b16_mln_512x512_160k_ade20k_20210624_130547-0403cee1.pth')
    parser.add_argument('--output_dir', default='results_downstream')
    parser.add_argument('--ade_root', default=os.environ.get('ADE20K_ROOT', './data/ade20k'))
    parser.add_argument('--latency_only', action='store_true')
    parser.add_argument('--run_full_eval', action='store_true')
    args = parser.parse_args()
    
    if not MMSEG_AVAILABLE:
        print("MMSeg needed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading config: {args.config_file}")
    cfg = Config.fromfile(args.config_file)
    cfg.work_dir = str(output_dir)
    
    # Build helper
    def build_model_custom(patch_kwargs=None):
        try:
            model = init_model(cfg, args.checkpoint, device='cuda')
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}). Using random weights.")
            model = init_model(cfg, checkpoint=None, device='cuda')
            
        if patch_kwargs:
            apply_srtome_mmseg(model, **patch_kwargs)
        return model

    # 1. Latency & Throughput
    if args.latency_only:
        print("\n=== Latency & Throughput Benchmark ===")
        variants = [
             ("Baseline", None),
             ("S3-Merge-l6", {"ratio": 0.7, "late_start": 6}),
             ("S3-Merge-l8", {"ratio": 0.7, "late_start": 8}),
        ]
        
        results = []
        
        for name, kwargs in variants:
            print(f"Benchmarking {name}...")
            model = build_model_custom(kwargs)
            
            # Latency (B=1)
            lat = measure_time(model, batch_size=1, mode='latency')
            
            # Throughput (B=8)
            thr = measure_time(model, batch_size=8, mode='throughput')
            
            print(f"  Latency: {lat:.2f} ms | Throughput: {thr:.1f} img/s")
            results.append({
                "method": name, 
                "latency_b1_ms": lat, 
                "throughput_b8_imgs": thr
            })
            del model
            torch.cuda.empty_cache()

        csv_path = output_dir / 'ade20k_latency.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["method", "latency_b1_ms", "throughput_b8_imgs"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {csv_path}")

    # 2. Full Evaluation (mIoU)
    # Using 'mmengine.runner.Runner.test'
    if args.run_full_eval:
        print("\n=== Full ADE20K Evaluation ===")
        # We need to construct a Runner or use Evaluator manually
        # Runner is easiest as it handles dataloader construction from config
        
        # Override data root
        cfg.data_root = args.ade_root
        cfg.train_dataloader.dataset.data_root = args.ade_root
        cfg.val_dataloader.dataset.data_root = args.ade_root
        cfg.test_dataloader.dataset.data_root = args.ade_root
        
        # Override config paths for dataset
        # The variables in config usually use data_root
        
        variants = [
             ("Baseline", None),
             ("S3-Merge-l6", {"ratio": 0.7, "late_start": 6}),
        ]
        
        miou_results = []
        
        for name, kwargs in variants:
            print(f"Evaluating {name}...")
            # We can't easily pass 'model' object to Runner if we init it outside.
            # Runner builds its own model.
            # We need a hook or custom model build function?
            # Or we patch the Runner's model after build.
            
            runner = Runner.from_cfg(cfg)
            
            # Patch model
            if kwargs:
                apply_srtome_mmseg(runner.model, **kwargs)
            
            runner.load_checkpoint(args.checkpoint)
            
            # Run test
            metrics = runner.test()
            print(metrics)
            
            # Extract mIoU
            # metrics is dict like {'coco/bbox_mAP': ...}
            # For ADE20K/MMSeg: {'mIoU': 45.x, ...}
            miou = metrics.get('mIoU', 0.0)
            if 'mIoU' not in metrics:
                 # Try finding keys containing IoU
                 for k, v in metrics.items():
                      if 'mIoU' in k:
                           miou = v
                           break
            
            miou_results.append({"method": name, "mIoU": miou})
            del runner
            torch.cuda.empty_cache()
            
        csv_path = output_dir / 'ade20k_miou.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["method", "mIoU"])
            writer.writeheader()
            writer.writerows(miou_results)
        print(f"Saved {csv_path}")

if __name__ == '__main__':
    main()
