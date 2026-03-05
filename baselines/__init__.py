"""
Baselines module: unified API for modern token merging methods.

Usage:
    from baselines import apply_method
    model = apply_method(model, 'pitome', ratio=0.55)
"""
import sys
from pathlib import Path

# Add baseline repos to path
_base = Path(__file__).parent
sys.path.insert(0, str(_base / 'PiToMe'))
sys.path.insert(0, str(_base / 'ATC'))


def apply_method(model, method_name, **kwargs):
    """Apply a token merging method to a timm ViT model.
    
    Args:
        model: A timm VisionTransformer (e.g., deit_small_patch16_224)
        method_name: One of 'pitome', 'atc', 'dsm'
        **kwargs: Method-specific arguments
    
    Returns:
        Patched model ready for inference
    """
    if method_name == 'pitome':
        return _apply_pitome(model, **kwargs)
    elif method_name == 'atc':
        return _apply_atc(model, **kwargs)
    elif method_name == 'dsm':
        return _apply_dsm(model, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def _apply_pitome(model, ratio=0.55, **kwargs):
    """Apply PiToMe (NeurIPS 2024) to a timm model.
    
    Args:
        ratio: Fraction of tokens to KEEP per layer (0-1).
               e.g., 0.55 keeps 55% of tokens per layer.
    """
    from algo.pitome.patch.deit import apply_patch
    apply_patch(model, trace_source=False, prop_attn=False)
    model.ratio = ratio
    return model


def _apply_atc(model, reduction_ratio=0.5, linkage='single',
               reduction_loc=None, **kwargs):
    """Apply ATC (ECCV 2024) to a timm model.
    
    ATC requires rebuilding the model with custom blocks.
    We load the pretrained weights into the ATC architecture.
    
    Args:
        reduction_ratio: Fraction of tokens to merge at each reduction layer
        linkage: Clustering linkage type ('single', 'average', 'complete')
        reduction_loc: List of block indices where reduction occurs
    """
    import argparse
    import timm
    import torch
    
    sys.path.insert(0, str(_base / 'ATC' / 'ATC_setup'))
    from models.atc import AgglomerativeClusteringVisionTransformer
    
    if reduction_loc is None:
        reduction_loc = list(range(12))  # all blocks
    
    args = argparse.Namespace(
        reduction_ratio=[reduction_ratio],
        reduction_loc=reduction_loc,
        linkage=linkage,
        proportional_attn=True,
    )
    
    atc_model = AgglomerativeClusteringVisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12,
        num_heads=6, mlp_ratio=4., qkv_bias=True,
        num_classes=1000, args=args,
    )
    
    # Load pretrained DeiT-S weights (skip mismatched keys from block structure)
    state_dict = model.state_dict()
    missing, unexpected = atc_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"ATC: {len(missing)} missing keys (expected for custom blocks)")
    
    atc_model = atc_model.to(next(model.parameters()).device)
    atc_model.eval()
    return atc_model


def _apply_dsm(model, delay_layer=6, merge_r=12, window_size=7, **kwargs):
    """Apply DSM (NeurIPS 2024 Workshop) - re-implementation from paper.
    
    'Training-Free Visual Token Compression via Delayed Spatial Merging'
    Key ideas:
    1. Delayed onset: skip merging in early layers where attention hasn't converged
    2. Hierarchical spatial merging after delay point
    3. Outlier-aware: exclude high-activation tokens from merge candidates
    
    Args:
        delay_layer: Skip merging before this layer (paper uses ~layer 6)
        merge_r: Number of tokens to merge per layer after delay
        window_size: Spatial window for local merging
    
    NOTE: Re-implementation, no official code available.
    """
    from baselines.dsm_impl import apply_dsm_patch
    apply_dsm_patch(model, delay_layer=delay_layer, merge_r=merge_r,
                    window_size=window_size)
    return model
