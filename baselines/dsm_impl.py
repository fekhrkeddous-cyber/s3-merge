"""
DSM: Delayed Spatial Merging — Re-implementation from paper (vectorized).

Paper: "Training-Free Visual Token Compression via Delayed Spatial Merging"
       NeurIPS 2024 Workshop on ML & Compression
       Heo et al., 2024

Key ideas:
1. Delayed onset: skip merging in early layers (attention not yet converged)
2. Similarity-based merging using importance ranking after delay point  
3. Outlier-aware: exclude high-activation tokens from merge candidates

NOTE: This is a faithful re-implementation; no official code is available.
      Batched operations are optimized to avoid per-sample Python loops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
import math


def dsm_merge(x, r, outlier_threshold=2.0):
    """
    Delayed Spatial Merging: merge r tokens based on similarity,
    excluding activation outliers. Fully vectorized across batch.
    
    Args:
        x: [B, N, C] with CLS at index 0
        r: number of tokens to reduce by
        outlier_threshold: std multiplier to identify outlier tokens
    
    Returns:
        merged x: [B, N-r, C]
    """
    B, N, C = x.shape
    
    if r <= 0 or N <= 2:
        return x
    
    # Separate CLS token
    cls = x[:, :1]       # [B, 1, C]
    patch = x[:, 1:]     # [B, n_patches, C] 
    n_patches = patch.shape[1]
    
    r = min(r, n_patches - 1)
    if r <= 0:
        return x
    n_keep = n_patches - r
    
    # Detect outlier tokens (high activation norm — protect from merging)
    norms = patch.norm(dim=-1)  # [B, n_patches]
    mean_norm = norms.mean(dim=-1, keepdim=True)
    std_norm = norms.std(dim=-1, keepdim=True) + 1e-6
    is_outlier = norms > (mean_norm + outlier_threshold * std_norm)  # [B, n_patches]
    
    # Compute pairwise cosine similarity
    patch_norm = F.normalize(patch, dim=-1)  # [B, n_patches, C]
    sim = torch.bmm(patch_norm, patch_norm.transpose(-1, -2))  # [B, n_patches, n_patches]
    
    # Mask self-similarity
    eye = torch.eye(n_patches, device=x.device, dtype=torch.bool).unsqueeze(0)
    sim = sim.masked_fill(eye, -float('inf'))
    
    # Mask outlier tokens as merge sources (rows)
    outlier_src = is_outlier.unsqueeze(-1).expand_as(sim)
    sim = sim.masked_fill(outlier_src, -float('inf'))
    
    # Split into src (to merge away) and dst (to keep) by importance
    _, sort_idx = norms.sort(dim=-1)  # ascending: least important first
    src_idx = sort_idx[:, :r]     # [B, r]
    dst_idx = sort_idx[:, r:]     # [B, n_keep]
    
    # For each src, find its best dst match
    # Gather src rows from sim matrix: [B, r, n_patches]
    src_sims = torch.gather(sim, 1, src_idx.unsqueeze(-1).expand(-1, -1, n_patches))
    
    # Mask non-dst columns
    dst_mask = torch.zeros(B, n_patches, device=x.device, dtype=torch.bool)
    dst_mask.scatter_(1, dst_idx, True)
    src_sims = src_sims.masked_fill(~dst_mask.unsqueeze(1).expand(-1, r, -1), -float('inf'))
    
    # Best dst (in original patch index space) for each src
    best_dst_abs = src_sims.argmax(dim=-1)  # [B, r]
    
    # === Vectorized merge: scatter_reduce src tokens into dst positions ===
    # Build assignment: for each patch position, map to a "cluster" position
    # dst tokens keep their position (relative to dst_idx ordering)
    # src tokens are assigned to their best_dst's position in the dst ordering
    
    # Convert best_dst from absolute patch index to position within dst_idx
    # Create inverse mapping: for each absolute index, its position in dst_idx
    # Use argsort of dst_idx to get this mapping
    dst_sorted, dst_order = dst_idx.sort(dim=-1)  # [B, n_keep]
    
    # For absolute indices in best_dst_abs, find their rank within dst_sorted
    # Since dst_sorted is sorted, use searchsorted
    best_dst_rank = torch.searchsorted(dst_sorted, best_dst_abs)  # [B, r]
    best_dst_rank = best_dst_rank.clamp(max=n_keep - 1)
    
    # Build full cluster assignment: each patch → cluster index
    cluster = torch.zeros(B, n_patches, device=x.device, dtype=torch.long)
    
    # dst tokens: map to their position in the output [0, n_keep)
    # Need to assign dst positions sequentially
    dst_positions = torch.arange(n_keep, device=x.device).unsqueeze(0).expand(B, -1)
    cluster.scatter_(1, dst_sorted, dst_positions)
    
    # src tokens: map to their best_dst's position in the output
    cluster.scatter_(1, src_idx, best_dst_rank)
    
    # Weighted merge using scatter_reduce with mean
    # Unlike simple scatter_reduce('mean'), we want merge_wavg behavior
    # where dst contributes with weight 1 and src contributes with weight 1
    # scatter_reduce with 'mean' over the cluster assignment handles this correctly
    
    cluster_exp = cluster.unsqueeze(-1).expand(-1, -1, C)  # [B, n_patches, C]
    
    # Initialize output with zeros, scatter_reduce with mean
    merged = torch.zeros(B, n_keep, C, device=x.device, dtype=patch.dtype)
    merged = merged.scatter_reduce(1, cluster_exp, patch, reduce='mean', include_self=False)
    
    return torch.cat([cls, merged], dim=1)


class DSMBlock(Block):
    """Modified ViT Block with Delayed Spatial Merging.
    
    Overrides forward to:
    1. Run standard attention + MLP
    2. Apply DSM merge if this block is past the delay point
    """
    
    def forward(self, x, **kwargs):
        # Standard ViT block forward (timm 1.0+)
        if hasattr(self, 'ls1'):
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        elif hasattr(self, 'drop_path1'):
            x = x + self.drop_path1(self.attn(self.norm1(x)))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        
        # Apply DSM merge after this block if enabled
        if hasattr(self, '_dsm_merge_r') and self._dsm_merge_r > 0 and self._dsm_should_merge:
            x = dsm_merge(x, self._dsm_merge_r)
        
        return x


def apply_dsm_patch(model, delay_layer=6, merge_r=12, window_size=7):
    """
    Patch a timm ViT model with Delayed Spatial Merging.
    
    Args:
        model: timm VisionTransformer
        delay_layer: First layer to apply merging (0-indexed)
        merge_r: Tokens to merge per active layer
        window_size: Spatial window constraint (reserved for future use)
    """
    for i, block in enumerate(model.blocks):
        block.__class__ = DSMBlock
        block._dsm_should_merge = (i >= delay_layer)
        block._dsm_merge_r = merge_r if (i >= delay_layer) else 0
