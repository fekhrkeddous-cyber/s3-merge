"""
SR-ToMe with Unmerge Bookkeeping for Downstream Tasks (COCO/ADE20K)
OPTIMIZED: Cached grid mappings for B=1 latency reduction

Key addition: restore_idx that maps each original patch position to its merged token index.
This allows restoring a dense spatial grid for detection/segmentation heads.

Flow:
  tokens (grid) → merge (save restore_idx) → reduced tokens through late blocks →
  unmerge to grid → reshape → detection/segmentation head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional, Dict


class GridMappingCache:
    """
    Cached grid-to-window mappings.
    Precomputed once per (H, W, window_size, device) combination.
    Supports rectangular grids (H != W).
    """
    _cache: Dict[Tuple, 'GridMappingCache'] = {}
    
    def __init__(self, H: int, W: int, window_size: int, device: torch.device):
        ws = window_size
        n_windows_h = H // ws
        n_windows_w = W // ws
        tokens_per_window = ws * ws
        
        # Precompute grid position indices
        h_idx = torch.arange(H, device=device).view(H, 1).expand(H, W)
        w_idx = torch.arange(W, device=device).view(1, W).expand(H, W)
        
        wh = h_idx // ws
        ph = h_idx % ws
        ww = w_idx // ws
        pw = w_idx % ws
        
        # Window index (0..n_windows-1) for each grid position
        self.local_win_idx = (wh * n_windows_w + ww).view(-1)  # [H*W]
        
        # Position within window for each grid position
        self.pos_in_window = (ph * ws + pw).view(-1)  # [H*W]
        
        self.tokens_per_window = tokens_per_window
        self.n_windows = n_windows_h * n_windows_w
        self.n_windows_h = n_windows_h
        self.n_windows_w = n_windows_w
        self.H = H
        self.W = W
        self.grid_size = H  # backward compat for square grids
        self.window_size = window_size
    
    @classmethod
    def get(cls, H: int, W_or_ws: int = None, device: torch.device = None,
            *, W: int = None, window_size: int = None) -> 'GridMappingCache':
        """Flexible constructor: get(H, W, ws, device) or get(grid_size, ws, device)."""
        if W is not None and window_size is not None:
            # New-style: get(H, W=W, window_size=ws, device=device)
            key = (H, W, window_size, device)
            if key not in cls._cache:
                cls._cache[key] = cls(H, W, window_size, device)
            return cls._cache[key]
        else:
            # Old-style: get(grid_size, window_size, device)
            ws = W_or_ws
            key = (H, H, ws, device)
            if key not in cls._cache:
                cls._cache[key] = cls(H, H, ws, device)
            return cls._cache[key]


def sr_tome_merge_with_unmerge_cached(
    x: torch.Tensor,        # [B, N, C] with CLS at index 0
    target_tokens: int,     # Exact number of patch tokens to output (used if keep_hw is None)
    cache: GridMappingCache,
    weighted_merge: bool = False,  # Eq. 1 uniform averaging (paper default; set True for the similarity-weighted ablation)   # Use similarity-weighted fusion
    tau: float = 0.1,              # Temperature for weight softmax
    track_sizes: bool = False,     # Return token sizes for proportional attention
    keep_hw: tuple = None,         # (kh, kw) rectangular keep-grid per window
    target_policy: str = "random", # "random" or "stratified"
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    SR-ToMe merge with restore_idx for unmerging (CACHED version).
    
    Returns:
        merged: [B, 1 + K_total, C] - CLS + merged patch tokens
        restore_idx: [B, N-1] - maps each original patch (0..N-2) to merged index (0..K_total-1)
        token_sizes: [B, 1 + K_total] if track_sizes else None
    """
    B, N, C = x.shape
    device = x.device
    
    cls_token = x[:, :1]
    patches = x[:, 1:]
    n_patches = N - 1
    
    H = cache.H
    W = cache.W
    ws = cache.window_size
    n_windows = cache.n_windows
    tokens_per_window = cache.tokens_per_window
    
    if patches.shape[1] != H * W:
        restore_idx = torch.arange(n_patches, device=device).unsqueeze(0).expand(B, -1)
        return x, restore_idx, None
    
    if ws >= H or ws >= W:
        restore_idx = torch.arange(n_patches, device=device).unsqueeze(0).expand(B, -1)
        return x, restore_idx, None
    
    # Calculate K (targets to keep) per window
    if keep_hw is not None:
        kh, kw = keep_hw
        K = kh * kw
    else:
        K = target_tokens // n_windows
        K = max(1, K)
        K = min(K, tokens_per_window)
    
    n_sources = tokens_per_window - K
    
    if n_sources <= 0:
        restore_idx = torch.arange(n_patches, device=device).unsqueeze(0).expand(B, -1)
        return x, restore_idx, None
    
    K_total = n_windows * K
    n_windows_h = H // ws
    n_windows_w = W // ws
    
    # ===== Reshape and unfold =====
    grid = patches.view(B, H, W, C)
    grid_t = grid.permute(0, 3, 1, 2)  # [B, C, H, W]
    windows = grid_t.unfold(2, ws, ws).unfold(3, ws, ws)  # [B, C, nH, nW, ws, ws]
    windows = windows.permute(0, 2, 3, 4, 5, 1)  # [B, nH, nW, ws, ws, C]
    windows = windows.reshape(B, n_windows, tokens_per_window, C)
    flat_windows = windows.reshape(B * n_windows, tokens_per_window, C)
    
    BnW = flat_windows.shape[0]
    
    # ===== Target selection =====
    rand_scores = torch.rand(BnW, tokens_per_window, device=device)
    
    if target_policy == "stratified" and keep_hw is not None:
        kh, kw = keep_hw
        # Compute cell assignments: partition ws×ws window into kh×kw cells
        r = torch.arange(ws, device=device).unsqueeze(1).expand(ws, ws).reshape(-1)
        c = torch.arange(ws, device=device).unsqueeze(0).expand(ws, ws).reshape(-1)
        h_cell = (r * kh) // ws  # cell row for each position
        w_cell = (c * kw) // ws  # cell col for each position
        cell_id = h_cell * kw + w_cell  # [tokens_per_window], values in [0, K)
        
        # Offset scores by cell so topk picks exactly one per cell
        cell_offset = cell_id.float() * 2.0  # non-overlapping ranges
        stratified_scores = rand_scores + cell_offset.unsqueeze(0)  # [BnW, tpw]
        _, target_idx = torch.topk(stratified_scores, K, dim=1, sorted=False)

        # Source indices: mark targets, pick complement
        source_marker = rand_scores.clone()
        source_marker.scatter_(1, target_idx, -1.0)  # targets get -1, sources stay [0,1)
        _, source_idx = torch.topk(source_marker, n_sources, dim=1, largest=True, sorted=False)
    elif target_policy == "deterministic" and keep_hw is not None:
        kh, kw = keep_hw
        r = torch.arange(ws, device=device).unsqueeze(1).expand(ws, ws).reshape(-1)
        c = torch.arange(ws, device=device).unsqueeze(0).expand(ws, ws).reshape(-1)
        h_cell = (r * kh) // ws  # cell row for each position
        w_cell = (c * kw) // ws  # cell col for each position
        cell_id = h_cell * kw + w_cell  # [tokens_per_window], values in [0, K)
        cell_offset = cell_id.float() * 2.0  # non-overlapping ranges
        # Deterministic tie-breaker: prefer top-left of each cell (smallest linear index)
        det_scores = -torch.arange(tokens_per_window, device=device).float() / (tokens_per_window * 2.0)
        stratified_scores = det_scores.unsqueeze(0) + cell_offset.unsqueeze(0)
        stratified_scores = stratified_scores.expand(BnW, -1)  # Expand to [BnW, tpw]
        _, target_idx = torch.topk(stratified_scores, K, dim=1, sorted=False)
        source_marker = det_scores.unsqueeze(0).expand(BnW, -1).clone()
        source_marker.scatter_(1, target_idx, -100.0)
        _, source_idx = torch.topk(source_marker, n_sources, dim=1, largest=True, sorted=False)
    else:
        # Random selection (original behavior)
        _, target_idx = torch.topk(rand_scores, K, dim=1, sorted=False)
        _, source_idx = torch.topk(rand_scores, n_sources, dim=1, largest=False, sorted=False)
    
    # ===== SPATIAL SORTING: sort targets by position within window =====
    target_idx, _ = target_idx.sort(dim=1)  # [BnW, K] sorted ascending
    
    # Gather targets and sources
    targets = flat_windows.gather(1, target_idx.unsqueeze(-1).expand(-1, -1, C))
    sources = flat_windows.gather(1, source_idx.unsqueeze(-1).expand(-1, -1, C))
    
    # ===== Similarity matching =====
    targets_n = F.normalize(targets, dim=-1)
    sources_n = F.normalize(sources, dim=-1)
    sim = torch.bmm(sources_n, targets_n.transpose(1, 2))
    
    _, best_target_local = sim.max(dim=-1)  # [BnW, n_sources]
    
    # ===== Build restore_idx =====
    restore_idx_window = torch.zeros(BnW, tokens_per_window, dtype=torch.long, device=device)
    
    window_idx = torch.arange(BnW, device=device)
    win_idx_flat = window_idx % n_windows
    window_base = win_idx_flat * K
    
    target_local_rank = torch.arange(K, device=device).unsqueeze(0).expand(BnW, -1)
    target_global_idx = window_base.unsqueeze(1) + target_local_rank
    
    restore_idx_window.scatter_(1, target_idx, target_global_idx)
    source_global_idx = target_global_idx.gather(1, best_target_local)
    restore_idx_window.scatter_(1, source_idx, source_global_idx)
    
    # ===== Use CACHED grid mapping to convert back to original patch order =====
    local_win_idx = cache.local_win_idx  # [N-1] - precomputed
    pos_in_window = cache.pos_in_window  # [N-1] - precomputed
    
    batch_offset = torch.arange(B, device=device).view(B, 1) * n_windows
    flat_window_idx = batch_offset + local_win_idx.unsqueeze(0)
    pos_in_window_expanded = pos_in_window.unsqueeze(0).expand(B, -1)
    
    # Optimized gather using indexing
    restore_idx_window_flat = restore_idx_window.view(-1)
    gather_idx = flat_window_idx * tokens_per_window + pos_in_window_expanded
    restore_idx = restore_idx_window_flat[gather_idx.view(-1)].view(B, n_patches)
    
    # ===== Merge: weighted fusion or uniform averaging =====
    if weighted_merge:
        # Extract similarity of each source to its assigned target
        sim_assigned = sim.gather(2, best_target_local.unsqueeze(-1)).squeeze(-1)  # [BnW, n_sources]
        sim_assigned = sim_assigned.clamp(-1.0, 1.0)
        
        # Compute weights: exp((sim - 1) / tau), target self-weight = exp(0) = 1
        source_weights = ((sim_assigned - 1.0) / tau).exp()  # [BnW, n_sources], all <= 1.0
        
        # Sum of weights per target: 1.0 (self) + sum(source weights)
        weight_sum = torch.ones(BnW, K, device=device)
        weight_sum.scatter_add_(1, best_target_local, source_weights)
        
        # Weighted source contributions
        weighted_sources = sources * source_weights.unsqueeze(-1)
        
        # Accumulate: target (weight=1) + weighted sources
        scatter_idx_C = best_target_local.unsqueeze(-1).expand(-1, -1, C)
        merged_targets = targets.clone()
        merged_targets.scatter_add_(1, scatter_idx_C, weighted_sources)
        
        # Normalize by total weight
        merged_targets = merged_targets / weight_sum.unsqueeze(-1)
    else:
        # Original uniform averaging
        scatter_idx = best_target_local.unsqueeze(-1).expand(-1, -1, C)
        merged_targets = targets.scatter_reduce(1, scatter_idx, sources, reduce='mean')
    
    # ===== Reconstruct =====
    merged_windows = merged_targets.view(B, n_windows, K, C)
    merged_patches = merged_windows.view(B, -1, C)
    
    merged = torch.cat([cls_token, merged_patches], dim=1)
    
    # ===== Token sizes for proportional attention =====
    token_sizes = None
    if track_sizes:
        # Count how many original patches map to each merged token.
        # restore_idx: [B, N-1] with values in [0, K_total-1]
        src_ones = torch.ones(B, n_patches, device=device)
        patch_sizes = torch.zeros(B, K_total, device=device)
        patch_sizes.scatter_add_(1, restore_idx, src_ones)
        # Prepend CLS size = 1
        cls_size = torch.ones(B, 1, device=device)
        token_sizes = torch.cat([cls_size, patch_sizes], dim=1)  # [B, 1 + K_total]
    
    return merged, restore_idx, token_sizes


def sr_tome_unmerge(
    merged: torch.Tensor,       # [B, 1 + K_total, C]
    restore_idx: torch.Tensor,  # [B, N-1] values in [0..K_total-1]
    grid_size: int = 14,
    grid_h: int = None,
    grid_w: int = None,
) -> torch.Tensor:
    """
    Unmerge tokens back to dense spatial grid.
    Supports rectangular grids via grid_h, grid_w params.
    
    Returns:
        restored_grid: [B, C, H, W] - dense feature map for detection/segmentation heads
    """
    B, _, C = merged.shape
    H = grid_h if grid_h is not None else grid_size
    W = grid_w if grid_w is not None else grid_size
    
    merged_patches = merged[:, 1:]  # [B, K_total, C]
    
    restore_idx_expanded = restore_idx.unsqueeze(-1).expand(-1, -1, C)
    restored_patches = merged_patches.gather(1, restore_idx_expanded)
    
    restored_grid = restored_patches.view(B, H, W, C).permute(0, 3, 1, 2)
    
    return restored_grid


class SRToMeDownstream(nn.Module):
    """
    SR-ToMe wrapper for downstream tasks (detection/segmentation).
    OPTIMIZED: Caches grid mappings to eliminate per-forward overhead.
    
    Returns dense feature grid instead of CLS logits.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_tokens: int = 84,
        late_start: int = 8,
        window_size: int = 7,
        grid_size: int = 14,
        weighted_merge: bool = False,  # Eq. 1 uniform averaging (paper default; set True for the similarity-weighted ablation)
        tau: float = 0.1,
        prop_attn: bool = False,
    ):
        super().__init__()
        self.model = model
        self.target_tokens = target_tokens
        self.late_start = late_start
        self.window_size = window_size
        self.grid_size = grid_size
        self.weighted_merge = weighted_merge
        self.tau = tau
        self.prop_attn = prop_attn
        
        self._restore_idx: Optional[torch.Tensor] = None
        self._cache: Optional[GridMappingCache] = None
        
    def _get_cache(self, device: torch.device) -> GridMappingCache:
        """Lazy initialization of cached grid mapping."""
        if self._cache is None or self._cache.local_win_idx.device != device:
            self._cache = GridMappingCache.get(self.grid_size, self.window_size, device)
        return self._cache
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            features: [B, C, H, W] - dense feature grid for downstream heads
        """
        cache = self._get_cache(x.device)
        
        x = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        
        merged = False
        attn_mask = None  # PA: only set after merge
        
        for i, blk in enumerate(self.model.blocks):
            if merged and attn_mask is not None:
                x = blk(x, attn_mask=attn_mask)
            else:
                x = blk(x)
            
            if i == self.late_start and not merged:
                x, self._restore_idx, token_sizes = sr_tome_merge_with_unmerge_cached(
                    x, self.target_tokens, cache,
                    weighted_merge=self.weighted_merge, tau=self.tau,
                    track_sizes=self.prop_attn,
                )
                merged = True
                # Build PA mask: log(size) broadcast for [B, heads, Q, K]
                if self.prop_attn and token_sizes is not None:
                    attn_mask = token_sizes.log()[:, None, None, :]  # [B, 1, 1, N]
        
        x = self.model.norm(x)
        
        if self._restore_idx is not None:
            features = sr_tome_unmerge(x, self._restore_idx, grid_size=self.grid_size)
        else:
            patches = x[:, 1:]
            features = patches.view(x.shape[0], self.grid_size, self.grid_size, -1)
            features = features.permute(0, 3, 1, 2)
        
        return features
    
    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """For classification (returns CLS logits)."""
        cache = self._get_cache(x.device)
        
        x = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        
        merged = False
        attn_mask = None
        
        for i, blk in enumerate(self.model.blocks):
            if merged and attn_mask is not None:
                x = blk(x, attn_mask=attn_mask)
            else:
                x = blk(x)
            if i == self.late_start and not merged:
                x, _, token_sizes = sr_tome_merge_with_unmerge_cached(
                    x, self.target_tokens, cache,
                    weighted_merge=self.weighted_merge, tau=self.tau,
                    track_sizes=self.prop_attn,
                )
                merged = True
                if self.prop_attn and token_sizes is not None:
                    attn_mask = token_sizes.log()[:, None, None, :]
        
        x = self.model.norm(x)
        return self.model.head(x[:, 0])


def create_sr_tome_downstream(
    model_name: str = 'deit_small_patch16_224',
    target_tokens: int = 84,
    late_start: int = 8,
    window_size: int = 7,
    pretrained: bool = True,
    weighted_merge: bool = False,  # Eq. 1 uniform averaging (paper default; set True for the similarity-weighted ablation)
    tau: float = 0.1,
    prop_attn: bool = False,
) -> SRToMeDownstream:
    base = timm.create_model(model_name, pretrained=pretrained)
    return SRToMeDownstream(
        base, 
        target_tokens=target_tokens, 
        late_start=late_start, 
        window_size=window_size,
        grid_size=14,  # For patch16/224
        weighted_merge=weighted_merge,
        tau=tau,
        prop_attn=prop_attn,
    )


# ===== Test =====
if __name__ == '__main__':
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("SR-ToMe Downstream: OPTIMIZED Merge + Unmerge Test")
    print("=" * 60)
    
    # Create model
    model = create_sr_tome_downstream(
        target_tokens=84,
        late_start=8,
        window_size=7
    ).to(device).eval()
    
    # Test forward
    x = torch.randn(4, 3, 224, 224, device=device)
    
    with torch.no_grad():
        features = model(x)
        print(f"\nInput: {x.shape}")
        print(f"Features: {features.shape}")
        
        logits = model.forward_cls(x)
        print(f"Logits: {logits.shape}")
    
    # Latency test
    print("\n" + "=" * 60)
    print("Latency Test (B=1 and B=4)")
    print("=" * 60)
    
    baseline = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    
    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        with torch.no_grad():
            # Warmup
            for _ in range(20):
                _ = model(x)
                _ = baseline(x)
                torch.cuda.synchronize()
            
            # SR-ToMe
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                _ = model(x)
                torch.cuda.synchronize()
            sr_lat = (time.perf_counter() - t0) / 100 * 1000
            
            # Baseline
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                _ = baseline.forward_features(x)
                torch.cuda.synchronize()
            base_lat = (time.perf_counter() - t0) / 100 * 1000
        
        print(f"\nB={batch_size}:")
        print(f"  Baseline: {base_lat:.2f} ms")
        print(f"  SR-ToMe:  {sr_lat:.2f} ms")
        print(f"  Ratio:    {sr_lat / base_lat:.2f}x")
    
    print("\n✅ SR-ToMe Downstream (Optimized) ready")
