"""
SR-ToMe v3: Fixed Token Budget via Random K-Target Selection

Key fix: Instead of even/odd split (which limits output to |dst|),
we randomly select exactly K targets per window and merge all others into them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import time

from typing import Tuple, Optional


def sr_tome_merge_exact(
    x: torch.Tensor,        # [B, N, C] with CLS at index 0
    target_tokens: int,     # Exact number of patch tokens to output
    window_size: int = 7,   # Window size
    grid_size: int = 14,
    weighted_merge: bool = False,   # Use similarity-weighted fusion
    tau: float = 0.3,               # Temperature for weight softmax
    content_adaptive: bool = False, # CAMR: adapt K per window based on redundancy
    track_sizes: bool = False,      # Return token sizes for proportional attention
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    SR-ToMe with exact token budget via random K-target selection.
    
    For each window:
    1. Randomly select K tokens as targets (K = target_tokens / n_windows)
       - If content_adaptive: K varies per window based on redundancy
    2. Merge remaining tokens into nearest target via cosine similarity
    3. Optionally fuse via similarity-weighted averaging
    
    Returns:
        merged_x: [B, 1+target_tokens, C] — CLS + merged patches
        token_sizes: [B, 1+target_tokens] if track_sizes else None  
                     — number of original tokens each output represents
    """
    B, N, C = x.shape
    
    cls_token = x[:, :1]
    patches = x[:, 1:]
    n_patches = N - 1
    
    H = W = grid_size
    if patches.shape[1] != H * W:
        return x, None
    
    if window_size >= grid_size:
        return x, None
    
    ws = window_size
    n_windows_h = H // ws
    n_windows_w = W // ws
    n_windows = n_windows_h * n_windows_w
    tokens_per_window = ws * ws
    
    # Calculate base K (targets to keep) per window
    k_per_window = target_tokens // n_windows
    k_per_window = max(1, k_per_window)
    k_per_window = min(k_per_window, tokens_per_window)
    
    # ===== Reshape into windows =====
    nH, nW = H // ws, W // ws
    grid = patches.view(B, H, W, C)
    windows = grid.view(B, nH, ws, nW, ws, C)
    windows = windows.permute(0, 1, 3, 2, 4, 5)  # [B, nH, nW, ws, ws, C]
    flat_windows = windows.reshape(B * n_windows, tokens_per_window, C)
    
    BnW = flat_windows.shape[0]
    
    if content_adaptive and n_windows > 1:
        # ===== CAMR: Content-Adaptive Merge Ratio =====
        # Compute per-window redundancy: mean cosine sim of tokens to window mean
        win_mean = flat_windows.mean(dim=1, keepdim=True)  # [BnW, 1, C]
        win_mean_n = F.normalize(win_mean, dim=-1)
        tokens_n = F.normalize(flat_windows, dim=-1)
        redundancy = torch.bmm(tokens_n, win_mean_n.transpose(1, 2)).squeeze(-1)  # [BnW, w²]
        redundancy_score = redundancy.mean(dim=1)  # [BnW] — per-window scalar
        
        # Reshape to [B, n_windows] for per-image budget allocation
        r = redundancy_score.view(B, n_windows)
        complexity = 1.0 - r  # high = complex = keep more tokens
        
        K_base = k_per_window
        K_min = max(1, K_base - 1)
        K_max = min(tokens_per_window - 1, K_base + 1)
        
        total_budget = target_tokens
        base_total = K_base * n_windows
        excess = total_budget - base_total
        
        # Vectorized budget allocation (no Python loops)
        K_per_win = torch.full((B, n_windows), K_base, device=x.device, dtype=torch.long)
        _, sorted_idx = complexity.sort(dim=1, descending=True)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1)  # [B, 1]
        
        if excess > 0:
            n_upgrade = min(excess, n_windows)
            K_per_win[batch_idx.expand(-1, n_upgrade), sorted_idx[:, :n_upgrade]] = K_max
        elif excess < 0:
            n_downgrade = min(-excess, n_windows)
            K_per_win[batch_idx.expand(-1, n_downgrade), sorted_idx[:, -n_downgrade:]] = K_min
        
        # Process each K-level group (at most 3 iterations: K-1, K, K+1)
        K_flat = K_per_win.view(-1)  # [BnW]
        unique_Ks = K_flat.unique()
        
        all_merged = torch.zeros(BnW, K_max, C, device=x.device)
        all_K = K_flat  # track actual K per window
        
        for K_val in unique_Ks:
            K_int = K_val.item()
            mask = (K_flat == K_val)
            if not mask.any():
                continue
            
            group_windows = flat_windows[mask]  # [G, w², C]
            G = group_windows.shape[0]
            n_src = tokens_per_window - K_int
            
            if n_src <= 0:
                all_merged[mask, :K_int] = group_windows[:, :K_int]
                continue
            
            # Random target selection
            rs = torch.rand(G, tokens_per_window, device=x.device)
            _, t_idx = torch.topk(rs, K_int, dim=1, sorted=False)
            _, s_idx = torch.topk(rs, n_src, dim=1, largest=False, sorted=False)
            
            tgt = group_windows.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, C))
            src = group_windows.gather(1, s_idx.unsqueeze(-1).expand(-1, -1, C))
            
            # Similarity matching + merge
            tgt_n = F.normalize(tgt, dim=-1)
            src_n = F.normalize(src, dim=-1)
            sim = torch.bmm(src_n, tgt_n.transpose(1, 2))
            _, best_t = sim.max(dim=-1)
            
            if weighted_merge:
                sim_a = sim.gather(2, best_t.unsqueeze(-1)).squeeze(-1).clamp(-1.0, 1.0)
                sw = ((sim_a - 1.0) / tau).exp()
                ws_sum = torch.ones(G, K_int, device=x.device)
                ws_sum.scatter_add_(1, best_t, sw)
                w_src = src * sw.unsqueeze(-1)
                sc_idx = best_t.unsqueeze(-1).expand(-1, -1, C)
                merged = tgt.clone()
                merged.scatter_add_(1, sc_idx, w_src)
                merged = merged / ws_sum.unsqueeze(-1)
            else:
                sc_idx = best_t.unsqueeze(-1).expand(-1, -1, C)
                merged = tgt.scatter_reduce(1, sc_idx, src, reduce='mean')
            
            all_merged[mask, :K_int] = merged
        
        # Vectorized packing: build a [BnW, K_max] → [B, target_tokens] index map
        # Create a mask of valid positions across all windows
        pos_range = torch.arange(K_max, device=x.device).unsqueeze(0)  # [1, K_max]
        valid_mask = pos_range < all_K.unsqueeze(1)  # [BnW, K_max], True where valid
        
        # Reshape to [B, n_windows * K_max] and select valid
        valid_mask_flat = valid_mask.view(B, n_windows * K_max)
        all_merged_flat = all_merged.view(B, n_windows * K_max, C)
        
        # Gather valid tokens — each batch has exactly target_tokens valid
        merged_patches = torch.zeros(B, target_tokens, C, device=x.device)
        for b in range(B):
            valid_idx = valid_mask_flat[b].nonzero(as_tuple=True)[0][:target_tokens]
            merged_patches[b] = all_merged_flat[b, valid_idx]
        
        return torch.cat([cls_token, merged_patches], dim=1), None
    
    else:
        # ===== Uniform K: original path =====
        K = k_per_window
        n_sources = tokens_per_window - K
        
        if n_sources <= 0:
            return x, None
        
        # Random K-target selection
        rand_scores = torch.rand(BnW, tokens_per_window, device=x.device)
        _, target_idx = torch.topk(rand_scores, K, dim=1, sorted=False)
        _, source_idx = torch.topk(rand_scores, n_sources, dim=1, largest=False, sorted=False)
        
        targets = flat_windows.gather(1, target_idx.unsqueeze(-1).expand(-1, -1, C))
        sources = flat_windows.gather(1, source_idx.unsqueeze(-1).expand(-1, -1, C))
        
        # Similarity matching
        targets_n = F.normalize(targets, dim=-1)
        sources_n = F.normalize(sources, dim=-1)
        sim = torch.bmm(sources_n, targets_n.transpose(1, 2))
        _, best_target = sim.max(dim=-1)
        
        # Merge
        if weighted_merge:
            sim_assigned = sim.gather(2, best_target.unsqueeze(-1)).squeeze(-1)
            sim_assigned = sim_assigned.clamp(-1.0, 1.0)
            source_weights = ((sim_assigned - 1.0) / tau).exp()
            weight_sum = torch.ones(BnW, K, device=x.device)
            weight_sum.scatter_add_(1, best_target, source_weights)
            weighted_sources = sources * source_weights.unsqueeze(-1)
            scatter_idx_C = best_target.unsqueeze(-1).expand(-1, -1, C)
            merged_targets = targets.clone()
            merged_targets.scatter_add_(1, scatter_idx_C, weighted_sources)
            merged_targets = merged_targets / weight_sum.unsqueeze(-1)
        else:
            scatter_idx = best_target.unsqueeze(-1).expand(-1, -1, C)
            merged_targets = targets.scatter_reduce(1, scatter_idx, sources, reduce='mean')
        
        # Reconstruct
        merged_windows = merged_targets.view(B, n_windows, K, C)
        merged_patches = merged_windows.view(B, -1, C)
        
        # Compute token sizes
        sizes = None
        if track_sizes:
            # Each target gets 1 (itself) + count of sources merged into it
            target_counts = torch.ones(BnW, K, device=x.device)
            src_ones = torch.ones(BnW, n_sources, device=x.device)
            target_counts.scatter_add_(1, best_target, src_ones)
            # Reshape: [B, n_windows*K]
            patch_sizes = target_counts.view(B, -1)
            # Prepend CLS size = 1
            cls_size = torch.ones(B, 1, device=x.device)
            sizes = torch.cat([cls_size, patch_sizes], dim=1)  # [B, 1+target_tokens]
        
        return torch.cat([cls_token, merged_patches], dim=1), sizes


class SRToMeBlock(nn.Module):
    """
    Wraps a timm Block to apply S³-Merge after the block's forward pass.
    When prop_attn=True, stores token sizes on parent model for downstream use.
    """
    def __init__(self, block: nn.Module, target_tokens: int, window_size: int = 7,
                 weighted_merge: bool = False, tau: float = 0.3,
                 content_adaptive: bool = False, prop_attn: bool = True):
        super().__init__()
        self.block = block
        self.target_tokens = target_tokens
        self.window_size = window_size
        self.weighted_merge = weighted_merge
        self.tau = tau
        self.content_adaptive = content_adaptive
        self.prop_attn = prop_attn
        self._token_sizes = None  # populated during forward
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x, sizes = sr_tome_merge_exact(
            x, self.target_tokens, 
            window_size=self.window_size,
            grid_size=14,
            weighted_merge=self.weighted_merge,
            tau=self.tau,
            content_adaptive=self.content_adaptive,
            track_sizes=self.prop_attn,
        )
        self._token_sizes = sizes  # store for PropAttnBlock to read
        return x


class PropAttnBlock(nn.Module):
    """
    Wraps a timm Block to apply proportional attention bias.
    Reads token sizes from the merge block and injects log(size) as
    an additive bias to attended keys (before softmax).
    """
    def __init__(self, block: nn.Module, merge_block: SRToMeBlock):
        super().__init__()
        self.block = block
        self.merge_block = merge_block  # reference to read _token_sizes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes = self.merge_block._token_sizes
        if sizes is not None:
            # Build attention mask: log(size) broadcast across heads and queries
            # sizes: [B, N], need [B, 1, 1, N] for broadcast with [B, heads, Q, K]
            attn_mask = sizes.log()[:, None, None, :]  # [B, 1, 1, N]
            return self.block(x, attn_mask=attn_mask)
        else:
            return self.block(x)


def apply_patch(
    model: nn.Module,
    target_tokens: int = 84,
    late_start: int = 8,  # paper defaults: T=84, l8
    window_size: int = 7,
    weighted_merge: bool = False,
    tau: float = 0.3,
    content_adaptive: bool = False,
    prop_attn: bool = True,
):
    """
    Applies S³-Merge patch in-place to the model.
    - Replaces blocks[late_start] with SRToMeBlock wrapper.
    - If prop_attn: wraps blocks[late_start+1:] with PropAttnBlock for
      proportional attention (log-size bias on attention logits).
    """
    blocks = model.blocks if hasattr(model, 'blocks') else model.model.blocks
    
    if isinstance(blocks[late_start], SRToMeBlock):
        return model

    # Wrap the merge layer
    original_block = blocks[late_start]
    merge_block = SRToMeBlock(
        original_block, 
        target_tokens=target_tokens, 
        window_size=window_size,
        weighted_merge=weighted_merge,
        tau=tau,
        content_adaptive=content_adaptive,
        prop_attn=prop_attn,
    )
    blocks[late_start] = merge_block
    
    # Wrap post-merge blocks with proportional attention
    if prop_attn:
        for i in range(late_start + 1, len(blocks)):
            if not isinstance(blocks[i], PropAttnBlock):
                blocks[i] = PropAttnBlock(blocks[i], merge_block)
    
    return model


def create_sr_tome_v3(
    model_name: str = 'deit_small_patch16_224',
    target_tokens: int = 84,
    late_start: int = 8,  # paper defaults: T=84, l8
    window_size: int = 7,
    pretrained: bool = True,
    weighted_merge: bool = False,
    tau: float = 0.3,
    content_adaptive: bool = False,
    prop_attn: bool = True,
) -> nn.Module:
    base = timm.create_model(model_name, pretrained=pretrained)
    apply_patch(base, target_tokens=target_tokens, late_start=late_start,
                window_size=window_size, weighted_merge=weighted_merge, tau=tau,
                content_adaptive=content_adaptive, prop_attn=prop_attn)
    return base


# Keep v2 for backward compatibility
create_sr_tome_v2 = create_sr_tome_v3


if __name__ == '__main__':
    device = torch.device('cuda')
    
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
        return token_counts
    
    def bench(model, x, name, runs=100):
        model.eval()
        with torch.no_grad():
            for _ in range(20):
                _ = model(x)
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                _ = model(x)
                torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / runs * 1000
        print(f"{name}: {lat:.2f} ms")
        return lat
    
    x = torch.randn(1, 3, 224, 224, device=device)
    x64 = torch.randn(64, 3, 224, 224, device=device)
    
    # Test token counts
    print("="*60)
    print("TOKEN COUNT VERIFICATION")
    print("="*60)
    
    # SR-ToMe v3 7x7 target=87
    sr87 = create_sr_tome_v3(target_tokens=87, late_start=6, window_size=7).to(device).eval()
    counts87 = count_tokens(sr87, device)
    print(f"SR-ToMe 7x7 target=87: {counts87}")
    print(f"  Final tokens: {counts87[-1]} (expected: 88 = 87 patches + 1 CLS)")
    del sr87
    
    # SR-ToMe v3 2x2 target=87
    sr87_2x2 = create_sr_tome_v3(target_tokens=87, late_start=6, window_size=2).to(device).eval()
    counts87_2x2 = count_tokens(sr87_2x2, device)
    print(f"SR-ToMe 2x2 target=87: {counts87_2x2}")
    print(f"  Final tokens: {counts87_2x2[-1]} (expected: 88 = 87 patches + 1 CLS)")
    del sr87_2x2
    
    torch.cuda.empty_cache()
    
    # Latency test
    print("\n" + "="*60)
    print("LATENCY BENCHMARK")
    print("="*60)
    
    baseline = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    b_lat = bench(baseline, x64, "Baseline B=64")
    del baseline
    torch.cuda.empty_cache()
    
    sr = create_sr_tome_v3(target_tokens=87, late_start=6, window_size=7).to(device).eval()
    sr_lat = bench(sr, x64, "SR-ToMe 7x7 B=64")
    print(f"  Speedup: {b_lat/sr_lat:.2f}x")
    del sr
    torch.cuda.empty_cache()
    
    import tome
    tome_m = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    tome.patch.timm(tome_m)
    tome_m.r = 10
    t_counts = count_tokens(tome_m, device)
    print(f"\nToMe r=10 tokens: {t_counts}")
    print(f"  Final: {t_counts[-1]} (77 patches + 1 CLS = 78)")
    t_lat = bench(tome_m, x64, "ToMe r=10 B=64")
    print(f"  Speedup: {b_lat/t_lat:.2f}x")
