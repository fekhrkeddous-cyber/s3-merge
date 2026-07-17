"""Adaptive-K: variable per-window keep counts under an exact global token budget.

Implements the paper's Adaptive-K allocation (Section 3, Step 2, regime iii) and the
packed exact-coordinate execution path used for the revised COCO evaluation.
Merge semantics follow the paper's Eq. 1 (hard cosine argmax assignment, uniform
normalized averaging, first-max tie-break); no softmax weighting and no temperature
in the primary path. `merge='weighted_static'` reproduces the similarity-weighted
Static-K variant as an isolated ablation only.

With a constant K_w (allocate_uniform) and sel_scores=None this module also provides
the deterministic variance-aware Static-K selection policy; sel_scores accepts any
per-token score tensor, giving a policy-neutral selection interface.

Representation: packed-uniform. The global budget T_g is exact PER IMAGE, so every image
packs to exactly T_g tokens -> static shapes, zero padding, no attention masks.
No data-dependent Python loops; no .item()/.cpu()/.tolist() in the measured path;
integer sort keys only (no FP BIG constants).
"""
import torch
import torch.nn.functional as F


# ============================================================================
# Geometry (built once per (H, W, ws, device); pure config, no data dependence)
# ============================================================================

class PackGeometry:
    _cache = {}

    def __init__(self, H, W, ws, device):
        self.H, self.W, self.ws = H, W, ws
        assert H % ws == 0 and W % ws == 0, "grid must divide by merge window"
        self.nWh, self.nWw = H // ws, W // ws
        self.n_windows = self.nWh * self.nWw
        self.tpw = ws * ws
        dev = torch.device(device)
        w_idx = torch.arange(self.n_windows, device=dev)
        wh, ww = w_idx // self.nWw, w_idx % self.nWw
        t = torch.arange(self.tpw, device=dev)
        r, c = t // ws, t % ws
        # original grid coordinates of token t of window w
        self.coord_r = wh[:, None] * ws + r[None, :]            # (nW, tpw)
        self.coord_c = ww[:, None] * ws + c[None, :]            # (nW, tpw)
        # flat position in the H*W grid (a permutation of [0, H*W))
        self.pos_map = self.coord_r * W + self.coord_c          # (nW, tpw)

    @classmethod
    def get(cls, H, W, ws, device):
        key = (H, W, ws, str(device))
        if key not in cls._cache:
            cls._cache[key] = cls(H, W, ws, device)
        return cls._cache[key]


# ============================================================================
# Scores
# ============================================================================

def window_scores(win_tokens):
    """win_tokens: (B, nW, tpw, C) -> per-token centred-L2 (B, nW, tpw), window score (B, nW)."""
    mu = win_tokens.mean(dim=2, keepdim=True)
    tok_var = (win_tokens - mu).pow(2).sum(dim=-1)
    s_w = tok_var.mean(dim=2)
    return tok_var, s_w


# ============================================================================
# Allocation policies (all return integer K_w (B, nW) with sum == T_g exactly)
# ============================================================================

def allocate_uniform(s_w, K_min, K_max, T_g):
    """Forced-uniform control. Requires T_g divisible by n_windows."""
    B, nW = s_w.shape
    assert T_g % nW == 0, "uniform control needs divisible budget"
    K = T_g // nW
    assert K_min <= K <= K_max
    return torch.full((B, nW), K, dtype=torch.long, device=s_w.device)


def allocate_slot_auction(s_w, K_min, K_max, T_g, taper="linear"):
    """Slot-auction allocation (candidate policy; no optimality claim).

    Every window holds K_min guaranteed slots plus c = K_max - K_min auction slots.
    Slot j of window w bids priority  s_hat_w * taper(j)  with taper strictly
    decreasing in j (diminishing marginal value of extra tokens in one window; also
    prevents a single window from monopolising slots unless its score dominates
    every other window's FIRST slot). The E = T_g - nW*K_min globally best slots win.
    E is a CONFIG constant -> static shapes. Scores are robustly normalized per image
    by the per-image median so the taper acts on comparable magnitudes.
    Deterministic tie-break: stable sort => lower (window, slot) index wins ties.
    """
    B, nW = s_w.shape
    c = K_max - K_min
    E = T_g - nW * K_min
    assert 0 <= E <= nW * c, "infeasible (K_min, K_max, T_g)"
    if E == 0:
        return torch.full((B, nW), K_min, dtype=torch.long, device=s_w.device)
    med = s_w.median(dim=1, keepdim=True).values.clamp_min(1e-12)
    s_hat = s_w / med                                            # robust per-image scale
    j = torch.arange(c, device=s_w.device, dtype=s_w.dtype)
    if taper == "linear":
        tap = 1.0 - j / c                                        # (c,), in (0, 1]
    elif taper == "harmonic":
        tap = 1.0 / (1.0 + j)
    else:
        raise ValueError(taper)
    P = s_hat[:, :, None] * tap[None, None, :]                   # (B, nW, c)
    flat = P.reshape(B, nW * c)
    order = torch.argsort(flat, dim=1, descending=True, stable=True)
    win_of_slot = order[:, :E] // c                              # (B, E)
    e = torch.zeros(B, nW, dtype=torch.long, device=s_w.device)
    e.scatter_add_(1, win_of_slot, torch.ones_like(win_of_slot))
    return K_min + e


ALLOCATORS = {"auction": allocate_slot_auction, "uniform": allocate_uniform}


# ============================================================================
# Selection, merge (Eq. 2), packing, restore — one fused stage, static shapes
# ============================================================================

def adaptive_merge_pack(x_grid, geom, K_w, merge="eq2", tau=0.3, sel_scores=None):
    """x_grid: (B, H, W, C) -> packed (B, T_g, C), coords (B,T_g,2), restore (B,H*W), stats.

    All merge arms share this code path; K_w decides fixed vs variable allocation.
    merge='eq2'            : manuscript Eq. 2 (primary).
    merge='weighted_static': public Static-K weighted variant (ablation ONLY).
    sel_scores (B,nW,tpw)  : optional selection scores replacing per-token variance
                             (e.g. random scores for the Static-K Eq.2 arm).
    """
    B, H, W, C = x_grid.shape
    nW, tpw = geom.n_windows, geom.tpw
    dev = x_grid.device
    T_g = int(K_w.sum(dim=1)[0])  # config-static by allocator contract (asserted in tests)

    win = (x_grid.view(B, geom.nWh, geom.ws, geom.nWw, geom.ws, C)
           .permute(0, 1, 3, 2, 4, 5).reshape(B, nW, tpw, C))
    tok_var, _ = window_scores(win)
    rank_scores = tok_var if sel_scores is None else sel_scores

    # ---- within-window target selection: top-K_w by score, stable ties ----
    order = torch.argsort(rank_scores, dim=2, descending=True, stable=True)  # (B,nW,tpw)
    rank = torch.empty_like(order)
    rank.scatter_(2, order, torch.arange(tpw, device=dev).expand(B, nW, tpw))
    target_mask = rank < K_w[:, :, None]                                  # (B,nW,tpw)

    # ---- packing indices: window-major, position-order inside window ----
    kept_rank_local = target_mask.long().cumsum(dim=2) - 1                # valid where mask
    base = K_w.cumsum(dim=1) - K_w                                        # exclusive cumsum (B,nW)
    dst = torch.where(target_mask, base[:, :, None] + kept_rank_local,
                      torch.full_like(kept_rank_local, T_g))              # trash slot = T_g

    # ---- merge: hard argmax assignment + Eq. 2 averaging ----
    xn = F.normalize(win, dim=-1)
    sim = xn @ xn.transpose(2, 3)                                         # (B,nW,tpw,tpw)
    sim = sim.masked_fill(~target_mask[:, :, None, :], float("-inf"))
    best_tgt = sim.argmax(dim=-1)                                         # first-max tie-break
    src_mask = ~target_mask

    # Deterministic aggregation: float scatter_add_ on CUDA uses atomicAdd whose
    # accumulation order is non-deterministic (breaks bit-reproducibility).
    # Build a one-hot source->target assignment matrix instead (scatter_ of a constant
    # into row-unique positions is deterministic) and aggregate via matmul.
    onehot = win.new_zeros(B, nW, tpw, tpw)                               # [.., src, tgt]
    onehot.scatter_(3, best_tgt[..., None], 1.0)
    if merge == "eq2":
        w_src = src_mask.to(win.dtype)                                    # weight 1 per source
    elif merge == "weighted_static":
        sim_assigned = sim.gather(3, best_tgt[..., None]).squeeze(-1).clamp(-1.0, 1.0)
        w_src = ((sim_assigned - 1.0) / tau).exp() * src_mask.to(win.dtype)
    else:
        raise ValueError(merge)
    onehot = onehot * w_src[..., None]
    acc = torch.einsum("bwst,bwsc->bwtc", onehot, win)                    # weighted source sums
    den = 1.0 + onehot.sum(dim=2)                                         # 1 + count / 1 + sum(w)
    merged_at_tok = (win + acc) / den[..., None]

    # ---- scatter into packed tensors (one extra trash slot, then slice) ----
    dstF = dst.reshape(B, nW * tpw)
    x_pack = win.new_zeros(B, T_g + 1, C)
    x_pack.scatter_(1, dstF[..., None].expand(-1, -1, C), merged_at_tok.reshape(B, nW * tpw, C))
    x_pack = x_pack[:, :T_g]

    cr = geom.coord_r.reshape(1, nW * tpw).expand(B, -1)
    cc = geom.coord_c.reshape(1, nW * tpw).expand(B, -1)
    coord_pack = win.new_zeros(B, T_g + 1, 2, dtype=torch.long)
    coord_pack.scatter_(1, dstF[..., None].expand(-1, -1, 2),
                        torch.stack([cr, cc], dim=-1))
    coord_pack = coord_pack[:, :T_g]                                       # (B,T_g,2)

    # ---- restore: every original token -> packed slot of its target ----
    ar = torch.arange(tpw, device=dev).expand(B, nW, tpw)
    tgt_tok = torch.where(target_mask, ar, best_tgt)                       # own slot if target
    restore_local = dst.gather(2, tgt_tok)                                 # (B,nW,tpw), in [0,T_g)
    restore_idx = torch.empty(B, H * W, dtype=torch.long, device=dev)
    restore_idx.scatter_(1, geom.pos_map.reshape(1, nW * tpw).expand(B, -1),
                         restore_local.reshape(B, nW * tpw))

    stats = {"K_w": K_w, "physical_tokens": T_g, "logical_tokens": T_g, "padding_rate": 0.0}
    return x_pack, coord_pack, restore_idx, stats


def unpack_restore(x_pack, restore_idx, H, W):
    """(B, T_g, C) + (B, H*W) -> (B, H, W, C) full grid (gather; same as sr_tome_unmerge)."""
    B, _, C = x_pack.shape
    out = x_pack.gather(1, restore_idx[..., None].expand(-1, -1, C))
    return out.view(B, H, W, C)


# ============================================================================
# Packed global-attention block (exact generalization of the official Block)
# ============================================================================

def _packed_rel_bias(attn, q, rel_pos, coord, size, num_heads):
    """Add one decomposed-RPE axis for packed tokens with arbitrary coordinates.

    attn: (B*nh, T, T)  q: (B*nh, T, hd)  coord: (B, T) int  rel_pos: (2*size-1, hd)
    bias[b,h,i,j] = q[b,h,i] . rel_pos_lut[coord_i, coord_j]
    Exact generalization of detectron2 add_decomposed_rel_pos (grid = special case).
    """
    from d2vit_extract import get_rel_pos
    Bnh, T, hd = q.shape
    B = Bnh // num_heads
    lut = get_rel_pos(size, size, rel_pos)                    # (size, size, hd) official LUT
    lut_i = lut[coord]                                        # (B, T, size, hd)
    qb = q.view(B, num_heads, T, hd)
    G = torch.einsum("bhic,bixc->bhix", qb, lut_i)            # (B, nh, T, size)
    idx = coord[:, None, None, :].expand(B, num_heads, T, T)  # coord_j along last dim
    attn.view(B, num_heads, T, T).add_(G.gather(3, idx))
    return attn


def packed_block_forward(block, x_seq, coords, H, W):
    """Run an official global-attention Block on packed tokens (B, T, C).

    Mirrors Block.forward / Attention.forward line-for-line (same ops, same order),
    with the (H*W) flatten replaced by the packed length T and grid RPE replaced by
    the exact coordinate-gather generalization. drop_path is identity at eval.
    """
    assert getattr(block, "window_size", None) == 0, "packed path requires a global block"
    attn_mod = block.attn
    B, T, C = x_seq.shape
    nh = attn_mod.num_heads

    shortcut = x_seq
    x = block.norm1(x_seq)
    qkv = attn_mod.qkv(x).reshape(B, T, 3, nh, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * nh, T, -1).unbind(0)
    attn = (q * attn_mod.scale) @ k.transpose(-2, -1)
    if attn_mod.use_rel_pos:
        attn = _packed_rel_bias(attn, q, attn_mod.rel_pos_h, coords[..., 0], H, nh)
        attn = _packed_rel_bias(attn, q, attn_mod.rel_pos_w, coords[..., 1], W, nh)
    attn = attn.softmax(dim=-1)
    x = (attn @ v).view(B, nh, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)
    x = attn_mod.proj(x)
    x = shortcut + block.drop_path(x)
    x = x + block.drop_path(block.mlp(block.norm2(x)))
    return x


# ============================================================================
# Architecture guard + end-to-end adaptive stage
# ============================================================================

def assert_post_merge_blocks_global(blocks, late_start):
    """Refuse any configuration with a window-attention block after the merge point."""
    for i in range(late_start + 1, len(blocks)):
        wsz = getattr(blocks[i], "window_size", None)
        if wsz is None:
            raise RuntimeError(f"post-merge block {i}: cannot determine window_size; refusing")
        if wsz != 0:
            raise RuntimeError(
                f"post-merge block {i} uses window attention (window_size={wsz}); "
                f"true Adaptive-K requires global attention after the merge point"
            )


def adaptive_stage_forward(blocks, late_start, x_grid, *, budget, K_min, K_max,
                           allocator="auction", merge="eq2", ws=8, taper="linear"):
    """Merge -> packed post-merge blocks -> restore. x_grid: (B, H, W, C) at layer L_s output.

    Pre-merge blocks are NOT run here (they are identical across arms); this is the
    adaptive stage only, so it can be exercised standalone with real Block instances.
    """
    assert_post_merge_blocks_global(blocks, late_start)
    B, H, W, C = x_grid.shape
    geom = PackGeometry.get(H, W, ws, x_grid.device)
    win = (x_grid.view(B, geom.nWh, geom.ws, geom.nWw, geom.ws, C)
           .permute(0, 1, 3, 2, 4, 5).reshape(B, geom.n_windows, geom.tpw, C))
    _, s_w = window_scores(win)
    if allocator == "uniform":
        K_w = allocate_uniform(s_w, K_min, K_max, budget)
    elif allocator == "auction":
        K_w = allocate_slot_auction(s_w, K_min, K_max, budget, taper=taper)
    else:
        raise ValueError(allocator)
    x_pack, coords, restore_idx, stats = adaptive_merge_pack(x_grid, geom, K_w, merge=merge)
    for i in range(late_start + 1, len(blocks)):
        x_pack = packed_block_forward(blocks[i], x_pack, coords, H, W)
    out = unpack_restore(x_pack, restore_idx, H, W)
    return out, stats
