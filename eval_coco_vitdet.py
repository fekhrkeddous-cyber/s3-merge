#!/usr/bin/env python3
"""
COCO ViTDet Evaluation with SR-ToMe
"""
import argparse
import os
import sys
import time
import logging
import csv
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Detectron2 imports
try:
    import detectron2
    from detectron2.config import LazyConfig, instantiate
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.structures import ImageList
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("detectron2 not available. Run inside Docker container.")


# =============================================================================
# Monkey Patch for Model Zoo Config Loading
# =============================================================================
# ViTDet configs use "common/..." paths that are relative to D2 root configs
# We redirect these to our local d2_repo clone.
if DETECTRON2_AVAILABLE:
    from detectron2 import model_zoo
    original_get_config = model_zoo.get_config_file
    
    def custom_get_config_file(config_path):
        if config_path.startswith("common/"):
            # Map to local d2_repo/configs
            local_path = os.path.join(
                "configs/vitdet",
                config_path
            )
            if os.path.exists(local_path):
                return local_path
        # Also try standard lookup
        try:
             return original_get_config(config_path)
        except RuntimeError:
             # Fallback: check d2_repo/configs
             p = os.path.join("configs/vitdet", config_path)
             if os.path.exists(p):
                 return p
             raise

    # LazyConfig uses get_config_file internally? No, it uses get_config usually.
    # Wait, `mask_rcnn_vitdet_b_100ep.py` calls `model_zoo.get_config("common/models/mask_rcnn_vitdet.py")`.
    # `get_config` calls `get_config_file` then `LazyConfig.load`.
    # So patching `get_config_file` is enough IF `get_config` uses it.
    
    # Check D2 source (simulated):
    # def get_config(config_path):
    #     filename = get_config_file(config_path)
    #     if filename.endswith(".py"): return LazyConfig.load(filename)
    
    model_zoo.get_config_file = custom_get_config_file

def setup_logger(output_dir):
    logger = logging.getLogger("detectron2")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_dir, "coco_eval.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger

# =============================================================================
# SR-ToMe Patching Logic for Detectron2 ViT
# =============================================================================
from models.sr_tome_downstream import sr_tome_merge_with_unmerge_cached, sr_tome_unmerge, GridMappingCache

def apply_dynamic_window_patch(model):
    """
    Monkey-patches ViT blocks to support dynamic window sizes.
    If input resolution is not divisible by window_size, finds the nearest 
    window size that divides it and interpolates relative position embeddings.
    """
    vit = model.backbone.net
    if hasattr(vit, '_dynamic_window_patched'):
        return
    
    print("Applying Dynamic Window Size patching...")
    
    import torch.nn.functional as F
    
    def interpolate_rel_pos(rel_pos, new_size):
        # rel_pos: [2*M-1, C]
        # new_size: 2*M_new-1
        C = rel_pos.shape[1]
        x = rel_pos.permute(1, 0).unsqueeze(0) # [1, C, 2*M-1]
        # Use linear interpolation
        x_new = F.interpolate(x, size=new_size, mode='linear', align_corners=False)
        return x_new.squeeze(0).permute(1, 0).contiguous() # [2*M_new-1, C]

    def make_forward_wrapper(block):
        original_forward = block.forward
        original_ws = block.window_size
        
        def dynamic_forward(x):
            H, W = x.shape[1], x.shape[2]
            
            # Check if padding would be needed
            need_pad = (H % block.window_size != 0) or (W % block.window_size != 0)
            
            if not need_pad:
                return original_forward(x)
            
            # Find best new window size
            candidates = []
            for ws in range(original_ws - 4, original_ws + 5):
                if ws < 2: continue
                if H % ws == 0 and W % ws == 0:
                    candidates.append(ws)
            
            if not candidates:
                # Fallback to original (padding)
                return original_forward(x)
                
            # Pick closest to original
            new_ws = min(candidates, key=lambda w: abs(w - original_ws))
            
            # Monkey-patch temporary state
            old_ws = block.window_size
            block.window_size = new_ws
            
            # Interpolate weights if using rel_pos
            restored = False
            if block.attn.use_rel_pos:
                old_h = block.attn.rel_pos_h
                old_w = block.attn.rel_pos_w
                
                new_h_size = 2 * new_ws - 1
                new_w_size = 2 * new_ws - 1
                
                # Create temporary parameters
                # Note: We need to set them as nn.Parameter to work with F.multi_head_std... if used?
                # But D2 uses custom Attention. We just need to replace the attribute.
                block.attn.rel_pos_h = torch.nn.Parameter(interpolate_rel_pos(old_h, new_h_size))
                block.attn.rel_pos_w = torch.nn.Parameter(interpolate_rel_pos(old_w, new_w_size))
                
                restored = True
                
            try:
                out = original_forward(x)
            finally:
                # Restore state
                block.window_size = old_ws
                if restored:
                    block.attn.rel_pos_h = old_h
                    block.attn.rel_pos_w = old_w
            
            return out
            
        return dynamic_forward

    count = 0
    for block in vit.blocks:
        if block.window_size > 0:
            block.forward = make_forward_wrapper(block)
            count += 1
            
    vit._dynamic_window_patched = True
    print(f"Patched {count} blocks for dynamic window sizing.")

def apply_srtome_detectron2(model, ratio=0.7, late_start=6, merge_ws=8, weighted_merge=False,  # Eq. 1 uniform averaging (paper default)
                            keep_hw=None, target_policy="random"):
    """
    Patches D2 ViTDet model with S³-Merge (spatially-preserving variant).
    
    D2's ViT operates on [B, H, W, C] spatial tensors with window attention.
    After merging, we reshape the merged tokens back into a smaller spatial grid
    so that window attention continues to work in post-merge blocks.
    
    Strategy:
    - Merge with within-window merging (merge_ws × merge_ws windows).
    - Each window keeps k_per_window tokens.
    - For post-merge blocks: reshape [B, K_total, C] -> [B, H_m, W_m, C]
      where H_m = n_windows_h * k_h, W_m = n_windows_w * k_w.
    - This requires k_per_window to be a perfect square.
    - After all blocks: unmerge back to original [B, H, W, C].
    
    Window size arithmetic:
      grid_size=64, merge_ws=8 -> n_windows = 8×8 = 64, tokens_per_window = 64
      ratio=0.766 -> k_per_window = 49 = 7×7
      merged_grid = 8*7 × 8*7 = 56×56 = 3136 tokens
      56 is divisible by ViTDet's attention window_size=14 ✓
    """
    import math

    # Navigate to the actual ViT backbone
    vit = model.backbone.net
    if not hasattr(vit, 'blocks') or not hasattr(vit, 'patch_embed'):
        print("Warning: Could not find ViT at model.backbone.net")
        return model

    blocks = vit.blocks
    n_blocks = len(blocks)
    grid_size = 64  # 1024/16 = 64 for ViTDet-B
    
    if hasattr(vit, '_original_forward'):
        print("Warning: Model already patched, skipping re-patch.")
        return model

    original_forward = vit.forward
    vit._original_forward = original_forward
    
    # Compute merge geometry
    n_windows_per_dim = grid_size // merge_ws  # 8
    n_windows = n_windows_per_dim ** 2          # 64
    tokens_per_window = merge_ws ** 2           # 64
    
    # Calculate keep-grid dimensions per window
    if keep_hw is not None:
        k_h, k_w = keep_hw
        k_per_window = k_h * k_w
    else:
        # Legacy: largest perfect square ≤ tokens_per_window * ratio
        raw_k = int(tokens_per_window * ratio)
        k_sqrt = int(math.sqrt(raw_k))
        k_per_window = k_sqrt ** 2
        k_h = k_w = k_sqrt
    
    K_total = n_windows * k_per_window
    merged_h = n_windows_per_dim * k_h
    merged_w = n_windows_per_dim * k_w
    
    target_tokens = K_total
    actual_ratio = K_total / (grid_size ** 2)
    
    print(f"  S³-Merge geometry: {grid_size}×{grid_size} → {merged_h}×{merged_w} "
          f"(ratio={actual_ratio:.3f}, k={k_per_window}={k_h}×{k_w}, "
          f"policy={target_policy}, weighted={weighted_merge})")
    
    # Enable dynamic window patching if merged grid isn't 14-divisible
    vitdet_ws = 14  # ViTDet local attention window size
    needs_dynamic_ws = (merged_h % vitdet_ws != 0) or (merged_w % vitdet_ws != 0)
    if needs_dynamic_ws:
        print(f"  Note: {merged_h}×{merged_w} not divisible by {vitdet_ws}, enabling dynamic window patch")
        apply_dynamic_window_patch(model)
    
    def patched_vit_forward(x):
        """Modified ViT forward with S³-Merge preserving spatial grid."""
        # 1) Patch embed + pos embed
        x = vit.patch_embed(x)
        if vit.pos_embed is not None:
            from detectron2.modeling.backbone.vit import get_abs_pos
            x = x + get_abs_pos(
                vit.pos_embed, vit.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
        
        B, H, W, C = x.shape
        
        # 2) Pre-merge blocks
        for i in range(late_start + 1):
            x = blocks[i](x)
        
        # 3) Merge
        x_flat = x.reshape(B, H * W, C)
        cls_dummy = x_flat.mean(dim=1, keepdim=True)
        x_with_cls = torch.cat([cls_dummy, x_flat], dim=1)
        
        cache = GridMappingCache.get(grid_size, merge_ws, x.device)
        
        x_merged, restore_idx, _ = sr_tome_merge_with_unmerge_cached(
            x_with_cls, target_tokens, cache,
            weighted_merge=weighted_merge,
            keep_hw=(k_h, k_w) if keep_hw is not None else None,
            target_policy=target_policy,
        )
        # x_merged: [B, 1+K_total, C]
        
        # 4) Drop CLS, reshape to spatial grid [B, merged_h, merged_w, C]
        merged_patches = x_merged[:, 1:]  # [B, K_total, C]
        
        # The merge function returns tokens ordered by window:
        # [win_0_tok_0, ..., win_0_tok_k, win_1_tok_0, ..., win_1_tok_k, ...]
        # Reshape to [B, n_win_h, n_win_w, k_h, k_w, C] then permute to spatial grid
        merged_patches = merged_patches.view(
            B, n_windows_per_dim, n_windows_per_dim, k_h, k_w, C
        )
        # [B, nwh, nww, kh, kw, C] -> [B, nwh*kh, nww*kw, C]
        merged_patches = merged_patches.permute(0, 1, 3, 2, 4, 5).reshape(
            B, merged_h, merged_w, C
        )
        
        # 5) Post-merge blocks with window attention on smaller grid
        x_post = merged_patches
        for i in range(late_start + 1, n_blocks):
            x_post = blocks[i](x_post)
        
        # 6) Flatten back and prepend CLS for unmerge
        x_flat_post = x_post.reshape(B, merged_h * merged_w, C)
        
        # Reverse the spatial rearrangement back to window-order
        x_post_win = x_post.view(
            B, n_windows_per_dim, k_h, n_windows_per_dim, k_w, C
        )
        x_post_win = x_post_win.permute(0, 1, 3, 2, 4, 5).reshape(
            B, K_total, C
        )
        
        # Prepend CLS dummy for unmerge
        cls_post = x_post_win.mean(dim=1, keepdim=True)
        x_with_cls_post = torch.cat([cls_post, x_post_win], dim=1)
        
        # 7) Unmerge back to spatial grid
        features = sr_tome_unmerge(x_with_cls_post, restore_idx, grid_size=grid_size)
        
        outputs = {vit._out_features[0]: features}
        return outputs

    vit.forward = patched_vit_forward
    return model

# =============================================================================
# Benchmarking Utils
# =============================================================================

def measure_latency(model, batch_size=1, n_warmup=20, n_runs=100):
    """Measure pure model forward latency."""
    device = "cuda"
    model.eval()
    H, W = 1024, 1024
    images = torch.randn(batch_size, 3, H, W, device=device)
    
    from detectron2.structures import ImageList
    image_list = ImageList(images, [(H, W)] * batch_size)
    
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            features = model.backbone(image_list.tensor)
            proposals, _ = model.proposal_generator(image_list, features, None)
            _ = model.roi_heads(image_list, features, proposals, None)
            torch.cuda.synchronize()
        
        # Measure
        t0 = time.perf_counter()
        for _ in range(n_runs):
            features = model.backbone(image_list.tensor)
            proposals, _ = model.proposal_generator(image_list, features, None)
            _ = model.roi_heads(image_list, features, proposals, None)
            torch.cuda.synchronize()
            
    elapsed = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/vitdet/COCO/mask_rcnn_vitdet_b_100ep.py')
    parser.add_argument('--init_checkpoint', default='detectron2://ViTDet/COCO/mask_rcnn_vitdet_b_100ep.py')
    parser.add_argument('--output_dir', default='results_downstream')
    parser.add_argument('--coco_root', default=os.environ.get('COCO_ROOT', './data/coco'))
    parser.add_argument('--latency_only', action='store_true')
    parser.add_argument('--run_full_eval', action='store_true')
    parser.add_argument('--calibration', action='store_true', help='Run B* calibration')
    parser.add_argument('--skip_checkpoint', action='store_true', help='Skip checkpoint loading (use random weights for latency)')
    args = parser.parse_args()
    
    if not DETECTRON2_AVAILABLE:
        print("Detectron2 needed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(output_dir))
    
    print(f"Loading config: {args.config_file}")
    
    # Helper to build model
    skip_ckpt = args.skip_checkpoint or args.latency_only or args.calibration
    def build_vitdet(patch_kwargs=None):
        cfg = LazyConfig.load(args.config_file)
        cfg.train.init_checkpoint = args.init_checkpoint
        model = instantiate(cfg.model)
        model.to("cuda")
        if not skip_ckpt:
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        else:
            print("  (skipping checkpoint — using random weights for timing)")
        model.eval()
        if patch_kwargs:
            apply_srtome_detectron2(model, **patch_kwargs)
        return model

    # 1. Latency & Calibration
    if args.latency_only or args.calibration:
        print("\n=== Latency Benchmark ===")
        variants = [
             ("Baseline", None),
             ("l8-87-rand", {"late_start": 8, "keep_hw": (8, 7), "target_policy": "random"}),
             ("l8-87-det",  {"late_start": 8, "keep_hw": (8, 7), "target_policy": "deterministic"}),
        ]
        
        results = []
        
        # Run B=1, B=4, B=8 for all variants
        for batch_size in [1, 4, 8]:
            for name, kwargs in variants:
                print(f"Benchmarking {name} (B={batch_size})...")
                model = build_vitdet(kwargs)
                lat = measure_latency(model, batch_size=batch_size)
                throughput = batch_size / (lat / 1000.0)  # img/s
                print(f"  Latency: {lat:.2f} ms  Throughput: {throughput:.1f} img/s")
                results.append({"method": name, "batch": batch_size, "latency_ms": lat, "throughput_ips": throughput, "type": "main"})
                del model
                torch.cuda.empty_cache()
            
        # Calibration (Safe Mode)
        if args.calibration:
            print("\n=== Safe Mode Calibration (B*) ===")
            b_star = 0
            for b in [1, 2, 4, 8]:
                print(f"Checking B={b}...")
                
                # Baseline
                model = build_vitdet(None)
                base_lat = measure_latency(model, batch_size=b)
                del model
                
                # S3-Merge
                model = build_vitdet({"ratio": 0.5, "late_start": 6})
                sr_lat = measure_latency(model, batch_size=b)
                del model
                
                print(f"  Base: {base_lat:.2f} ms | SR: {sr_lat:.2f} ms")
                if sr_lat < base_lat:
                    print(f"  -> S3-Merge FASTER at B={b}")
                    if b_star == 0: b_star = b
                else:
                    print(f"  -> S3-Merge SLOWER/EQUAL at B={b}")
            
            print(f"\nDetermined B* = {b_star}")
            results.append({"method": "SafeMode_Calibrated", "b_star": b_star, "note": "Calibration result"})

        # Save Latency CSV
        csv_path = output_dir / 'coco_latency.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["method", "batch", "latency_ms", "throughput_ips", "type", "b_star", "note"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {csv_path}")

    # 2. Full Evaluation (mAP)
    if args.run_full_eval:
        print("\n=== Full COCO evaluation ===")
        # We need to register dataset if not registered
        from detectron2.data.datasets import register_coco_instances
        
        dataset_name = "coco_2017_val_custom"
        coco_val = os.path.join(args.coco_root, 'val2017')
        coco_ann = os.path.join(args.coco_root, 'annotations', 'instances_val2017.json')
        
        # We need to use DataCatalog properly
        from detectron2.data import DatasetCatalog
        if dataset_name not in DatasetCatalog.list():
             register_coco_instances(dataset_name, {}, coco_ann, coco_val)
        
        # Load config once to setup dataloader config
        cfg = LazyConfig.load(args.config_file)
        cfg.train.init_checkpoint = args.init_checkpoint
        


        # Adjust config for evaluation
        cfg.dataloader.test.dataset.names = dataset_name
        cfg.dataloader.test.mapper.is_train = False
        


        variants = [
             ("l8-87-rand", {"late_start": 8, "keep_hw": (8, 7), "target_policy": "random"}),
             ("l8-87-det",  {"late_start": 8, "keep_hw": (8, 7), "target_policy": "deterministic"}),
        ]
        
        map_results = []
        
        
        for name, kwargs in variants:
            print(f"Evaluating {name}...")
            model = build_vitdet(kwargs)
            
            evaluator = COCOEvaluator(dataset_name, output_dir=str(output_dir))
            loader = instantiate(cfg.dataloader.test)
            
            ret = inference_on_dataset(model, loader, evaluator)
            
            bbox = ret.get('bbox', {})
            segm = ret.get('segm', {})
            row = {
                "method": name,
                "box_AP": bbox.get('AP', 0.0),
                "box_AP50": bbox.get('AP50', 0.0),
                "box_AP75": bbox.get('AP75', 0.0),
                "box_AP_S": bbox.get('APs', 0.0),
                "box_AP_M": bbox.get('APm', 0.0),
                "box_AP_L": bbox.get('APl', 0.0),
                "mask_AP": segm.get('AP', 0.0),
            }
            map_results.append(row)
            print(f"  AP={row['box_AP']:.1f}  AP_S={row['box_AP_S']:.1f}  "
                  f"AP_M={row['box_AP_M']:.1f}  AP_L={row['box_AP_L']:.1f}  "
                  f"mask_AP={row['mask_AP']:.1f}")
            del model
            torch.cuda.empty_cache()
            
        csv_path = output_dir / 'coco_map.csv'
        fieldnames = ["method", "box_AP", "box_AP50", "box_AP75",
                       "box_AP_S", "box_AP_M", "box_AP_L", "mask_AP"]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(map_results)
        print(f"Saved {csv_path}")

if __name__ == '__main__':
    main()
