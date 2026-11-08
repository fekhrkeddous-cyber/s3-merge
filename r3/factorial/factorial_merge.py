"""Factorial merge for S3-Merge (paper Table 4, 2x2 schedule x scope study): arms SW, SG, PW, PG.

- DeiT-S/16 224^2, grid 14x14, w=7 -> 4 windows x 49 patches; final T=84 (K=21 per window).
- ONE priority draw per forward, at the first merge event: torch.rand(B*4, 49) on the default CUDA
  generator. The call, shape and window layout are identical to the canonical
  models/sr_tome.py::sr_tome_merge_exact (uniform-K path).
- Nested removal: per window, current survivors are kept in priority-descending (rank) order; each
  stage removes the k lowest-priority survivors (sources = complement of the kept candidates).
- Scope = candidate mask only: windowed arms mask candidates outside the source's ORIGINAL window
  (the window of the representative's original patch coordinate); global arms use no mask.
- Aggregation = size-weighted, mass-conserving mean, computed as a dense one-hot matmul
  (deterministic). For single-shot merges every size is 1, so it equals Eq. 1.
- Argmax ties are resolved by the lowest original patch coordinate among the maxima; ties are counted.
- After the final stage, every arm reorders its 21 survivors per window into the canonical
  torch.topk(p, 21, sorted=False) order, so the output layout equals the canonical S3-Merge Core.
- PA is OFF: post-merge blocks are plain timm Blocks called without attn_mask.
The canonical module models/sr_tome.py is not modified; this module only wraps timm blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 14
WS = 7
NWIN = 4          # (14/7)^2
TPW = 49          # tokens per window
N_PATCH = 196
K_FINAL = 21

ARM_SPECS = {
    # arm: (merge-after block indices, removals per window per stage, scope)
    "SW": ((8,), (28,), "windowed"),
    "SG": ((8,), (28,), "global"),
    "PW": ((5, 6, 7, 8), (7, 7, 7, 7), "windowed"),
    "PG": ((5, 6, 7, 8), (7, 7, 7, 7), "global"),
}


def coord_table(device):
    """coord[w, q] = row-major original patch index of within-window index q in window w.
    Identical layout to the canonical reshape: view(B,nH,ws,nW,ws,C).permute(0,1,3,2,4,5)."""
    w = torch.arange(NWIN, device=device).view(NWIN, 1)
    q = torch.arange(TPW, device=device).view(1, TPW)
    hw, ww = w // 2, w % 2
    r, c = q // WS, q % WS
    return (hw * WS + r) * GRID + (ww * WS + c)          # [4, 49]


def window_of_coord(device):
    i = torch.arange(N_PATCH, device=device)
    return (i // GRID // WS) * (GRID // WS) + (i % GRID) // WS   # [196]


def _is_compiling():
    try:
        return bool(torch._dynamo.is_compiling())
    except Exception:
        return False


class FactorialState:
    """Per-forward state shared by the merge stages of one model. Reset by FactorialModel.forward."""

    def __init__(self, record=False):
        self.record = record
        self.reset()

    def reset(self):
        self.p = None            # [B*4, 49] priority draw (canonical layout)
        self.rep = None          # [B, n] original coord of each patch slot (window-major, rank order)
        self.size = None         # [B, n] float32 token mass (number of original patches)
        self.owner = None        # [B, 196] representative coord of each original patch
        self.m = None            # current survivors per window
        self.stage_calls = []    # block indices at which a merge stage executed
        self.logs = []           # per-stage records (only when record=True)
        self.final = None        # final record (only when record=True)
        self.inforward_checks = 0


def merge_stage(x, state, block_index, k, scope, is_last):
    """One merge stage after block `block_index`: remove k lowest-priority survivors per window."""
    # ---- in-forward assertions ----
    assert x.dtype == torch.float32, f"dtype {x.dtype} != float32"
    assert not torch.is_autocast_enabled(), "autocast enabled inside forward"
    assert not _is_compiling(), "torch.compile active inside forward"
    state.inforward_checks += 1

    B, N, C = x.shape
    dev = x.device
    cls_tok = x[:, :1]

    if state.p is None:                       # first merge event of this forward
        assert N == 1 + N_PATCH, f"first merge expects {1 + N_PATCH} tokens, got {N}"
        # SAME call/shape as canonical models/sr_tome.py::sr_tome_merge_exact (uniform-K path):
        p = torch.rand(B * NWIN, TPW, device=dev)
        state.p = p
        ctab = coord_table(dev)                                        # [4,49]
        order = torch.sort(p, dim=1, descending=True, stable=True).indices   # [B*4,49] rank order (q)
        rep0 = ctab.unsqueeze(0).expand(B, -1, -1).reshape(B * NWIN, TPW).gather(1, order)  # coords
        rep0 = rep0.view(B, NWIN * TPW)                                # window-major, rank order
        patches = x[:, 1:]                                             # raster order
        xs = patches.gather(1, rep0.unsqueeze(-1).expand(-1, -1, C))   # rank-ordered tokens
        state.rep = rep0
        state.size = torch.ones(B, N_PATCH, device=dev, dtype=torch.float32)
        state.owner = torch.arange(N_PATCH, device=dev).unsqueeze(0).expand(B, -1).clone()
        state.m = TPW
    else:
        xs = x[:, 1:]
    m = state.m
    assert xs.shape[1] == NWIN * m, f"expected {NWIN * m} patch tokens, got {xs.shape[1]}"
    assert 0 < k < m

    win_of = window_of_coord(dev)
    xs4 = xs.view(B, NWIN, m, C)
    rep4 = state.rep.view(B, NWIN, m)
    size4 = state.size.view(B, NWIN, m)
    # storage-window == coordinate-window (no compaction-based reassignment)
    slot_win = torch.arange(NWIN, device=dev).view(1, NWIN, 1).expand(B, NWIN, m)
    assert torch.equal(win_of[rep4], slot_win), "slot window differs from original-coordinate window"

    mc = m - k
    cand = xs4[:, :, :mc].reshape(B, NWIN * mc, C)
    rem = xs4[:, :, mc:].reshape(B, NWIN * k, C)
    rep_c = rep4[:, :, :mc].reshape(B, NWIN * mc)
    rep_r = rep4[:, :, mc:].reshape(B, NWIN * k)
    size_c = size4[:, :, :mc].reshape(B, NWIN * mc)
    size_r = size4[:, :, mc:].reshape(B, NWIN * k)

    sim = torch.bmm(F.normalize(rem, dim=-1), F.normalize(cand, dim=-1).transpose(1, 2))  # [B,4k,4mc]
    if scope == "windowed":
        wr = torch.arange(NWIN, device=dev).repeat_interleave(k)
        wc = torch.arange(NWIN, device=dev).repeat_interleave(mc)
        sim = sim.masked_fill((wr.view(-1, 1) != wc.view(1, -1)).unsqueeze(0), float("-inf"))
    elif scope != "global":
        raise ValueError(scope)
    maxv = sim.max(dim=-1, keepdim=True).values
    ismax = sim == maxv
    n_ties = int((ismax.sum(-1) > 1).sum()) if state.record else -1
    big = torch.iinfo(torch.long).max
    a = torch.where(ismax, rep_c.unsqueeze(1).expand(-1, NWIN * k, -1),
                    torch.full_like(rep_c.unsqueeze(1).expand(-1, NWIN * k, -1), big)).argmin(-1)  # [B,4k]

    A = F.one_hot(a, NWIN * mc).to(xs.dtype)                           # [B,4k,4mc]
    At = A.transpose(1, 2)                                             # [B,4mc,4k]
    num = cand * size_c.unsqueeze(-1) + torch.bmm(At, rem * size_r.unsqueeze(-1))
    den = size_c + torch.bmm(At, size_r.unsqueeze(-1)).squeeze(-1)
    new_cand = num / den.unsqueeze(-1)

    assigned_coord = rep_c.gather(1, a)                                # [B,4k]
    remap = torch.arange(N_PATCH, device=dev).unsqueeze(0).expand(B, -1).clone()
    remap.scatter_(1, rep_r, assigned_coord)
    state.owner = remap.gather(1, state.owner)

    rec = None
    if state.record:
        cross = (win_of[rep_r] != win_of[assigned_coord])
        rec = {
            "block": block_index, "m_before": m, "k": k, "m_after": mc, "scope": scope,
            "n_removed": NWIN * k, "n_candidates": NWIN * mc,
            "mass_sum": den.sum(1).detach().cpu(),                                  # [B]
            "cross_window_assignments": cross.sum(1).detach().cpu(),               # [B]
            "exact_argmax_ties": n_ties,
            "removed_coords": rep_r.detach().cpu(), "assigned_coords": assigned_coord.detach().cpu(),
        }

    rep_new, size_new = rep_c, den
    out_patches = new_cand
    if is_last:
        assert mc == K_FINAL
        ctab = coord_table(dev)
        canon_q = torch.topk(state.p, K_FINAL, dim=1, sorted=False).indices        # canonical order
        canon_coord = ctab.unsqueeze(0).expand(B, -1, -1).reshape(B * NWIN, TPW).gather(1, canon_q)
        canon_coord = canon_coord.view(B, NWIN * K_FINAL)
        slot_of = torch.full((B, N_PATCH), -1, device=dev, dtype=torch.long)
        slot_of.scatter_(1, rep_new, torch.arange(NWIN * K_FINAL, device=dev).unsqueeze(0).expand(B, -1))
        perm = slot_of.gather(1, canon_coord)
        assert bool((perm >= 0).all()), "final survivors differ from canonical topk target set"
        out_patches = out_patches.gather(1, perm.unsqueeze(-1).expand(-1, -1, C))
        rep_new = canon_coord
        size_new = size_new.gather(1, perm)
        if state.record:
            slot_final = torch.full((B, N_PATCH), -1, device=dev, dtype=torch.long)
            slot_final.scatter_(1, rep_new, torch.arange(NWIN * K_FINAL, device=dev).unsqueeze(0).expand(B, -1))
            restore = slot_final.gather(1, state.owner)
            state.final = {
                "rep": rep_new.detach().cpu(), "size": size_new.detach().cpu(),
                "owner": state.owner.detach().cpu(), "restore": restore.detach().cpu(),
                "p": state.p.detach().cpu(),
            }

    state.rep, state.size, state.m = rep_new, size_new, mc
    state.stage_calls.append(block_index)
    if rec is not None:
        rec["mass_sum_after"] = size_new.sum(1).detach().cpu()
        state.logs.append(rec)
    return torch.cat([cls_tok, out_patches], dim=1)


class FactorialMergeBlock(nn.Module):
    """Runs the wrapped timm Block, then one factorial merge stage (block-then-merge, as canonical)."""

    def __init__(self, block, block_index, k, scope, is_last, state):
        super().__init__()
        self.block = block
        self.block_index, self.k, self.scope, self.is_last = block_index, k, scope, is_last
        self._state = [state]        # list: keep the state object out of nn.Module registration

    def forward(self, x):
        x = self.block(x)
        return merge_stage(x, self._state[0], self.block_index, self.k, self.scope, self.is_last)


class FactorialModel(nn.Module):
    """Wraps a timm DeiT; resets the per-forward state so the priority draw happens inside forward."""

    def __init__(self, model, arm, record=False):
        super().__init__()
        if arm not in ARM_SPECS:
            raise ValueError(arm)
        blocks_at, ks, scope = ARM_SPECS[arm]
        self.model = model
        self.arm = arm
        self.state = FactorialState(record=record)
        for j, (b, k) in enumerate(zip(blocks_at, ks)):
            model.blocks[b] = FactorialMergeBlock(model.blocks[b], b, k, scope, j == len(ks) - 1, self.state)

    def forward(self, x):
        assert not torch.is_autocast_enabled(), "autocast enabled"
        assert not _is_compiling(), "torch.compile active"
        self.state.reset()
        out = self.model(x)
        blocks_at = ARM_SPECS[self.arm][0]
        assert self.state.stage_calls == list(blocks_at), f"merge stages {self.state.stage_calls} != {blocks_at}"
        return out
