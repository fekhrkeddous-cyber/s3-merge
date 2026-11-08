"""Release unit tests for S3-Merge (synthetic tensors only; no ImageNet, COCO or checkpoint downloads).

SOFTWARE TESTS ONLY: they check shapes, token counts, geometry validation and determinism. Nothing here is a
paper measurement. Run from the repository root:

    python tests/test_release.py            # plain runner (prints PASS/FAIL, exits non-zero on failure)
    python -m pytest tests/test_release.py   # also works where pytest is installed
"""
import importlib
import pathlib
import sys

import torch
import timm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.sr_tome import (apply_patch, resolve_merge_geometry, sr_tome_merge_exact,  # noqa: E402
                            PropAttnBlock, SRToMeBlock)
from models.sr_tome_downstream import (GridMappingCache, SRToMeDownstream,  # noqa: E402
                                       sr_tome_merge_with_unmerge_cached, sr_tome_unmerge)
from models.adaptive_k import (PackGeometry, adaptive_merge_pack, allocate_slot_auction,  # noqa: E402
                               allocate_uniform, unpack_restore, window_scores)


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _deit(img=224):
    torch.manual_seed(0)
    return timm.create_model("deit_small_patch16_224", pretrained=False, img_size=img).eval()


# ---------------------------------------------------------------------------- imports
def test_imports():
    for mod in ("models.sr_tome", "models.sr_tome_downstream", "models.adaptive_k", "models.safemode",
                "models.triton_merge"):
        importlib.import_module(mod)


# ---------------------------------------------------------------------------- grids and windows
def test_grid_14x14_window_7():
    g = resolve_merge_geometry(196, 84, 7)
    assert (g["grid"], g["n_windows"], g["k_per_window"], g["patch_tokens_out"]) == (14, 4, 21, 84)
    m = apply_patch(_deit(224), target_tokens=84, late_start=8, window_size=7, prop_attn=True)
    assert m.s3merge_config["grid"] == 14 and m.s3merge_config["total_tokens_out"] == 85


def test_grid_24x24_window_8():
    g = resolve_merge_geometry(576, 360, 8)
    assert (g["grid"], g["n_windows"], g["k_per_window"], g["patch_tokens_out"]) == (24, 9, 40, 360)
    m = apply_patch(_deit(384), target_tokens=360, late_start=8, window_size=8, prop_attn=False)
    assert m.s3merge_config["grid"] == 24
    with torch.no_grad():
        m(torch.randn(1, 3, 384, 384))
    geo = m.blocks[8].last_geometry
    assert geo["merges"] and geo["grid"] == 24 and geo["total_tokens_out"] == 361


def test_384_merge_executes_not_skipped():
    m = apply_patch(_deit(384), target_tokens=288, late_start=8, window_size=8, prop_attn=False)
    seen = {}
    m.blocks[8].register_forward_hook(lambda mod, i, o: seen.__setitem__("n", o.shape[1]))  # hook must return None
    with torch.no_grad():
        m(torch.randn(2, 3, 384, 384))
    assert seen["n"] == 1 + 9 * 32 < 577


def test_native_10x10_window_geometry():
    # LW-DETR-medium style geometry: 40x40 grid, native 10x10 windows, K=75 of 100 per window (packed path and
    # regular per-window path). The LW-DETR wrapper itself is not part of this release.
    geom = PackGeometry(40, 40, 10, torch.device("cpu"))
    x = torch.randn(2, 40, 40, 32)
    win = x.view(2, 4, 10, 4, 10, 32).permute(0, 1, 3, 2, 4, 5).reshape(2, 16, 100, 32)
    _, s_w = window_scores(win)
    K_w = allocate_uniform(s_w, 75, 75, 16 * 75)
    packed, coords, restore, _ = adaptive_merge_pack(x, geom, K_w)
    assert packed.shape == (2, 1200, 32) and coords.shape == (2, 1200, 2) and restore.shape == (2, 1600)
    assert unpack_restore(packed, restore, 40, 40).shape == (2, 40, 40, 32)
    cache = GridMappingCache.get(40, W=40, window_size=10, device=torch.device("cpu"))
    xm, ridx, _ = sr_tome_merge_with_unmerge_cached(torch.randn(2, 1601, 32), 1200, cache)
    assert xm.shape == (2, 1201, 32) and sr_tome_unmerge(xm, ridx, grid_h=40, grid_w=40).shape == (2, 32, 40, 40)


# ---------------------------------------------------------------------------- token accounting
def test_cls_preserved():
    x = torch.randn(3, 197, 16)
    torch.manual_seed(1)
    y, sizes = sr_tome_merge_exact(x, 84, window_size=7, track_sizes=True)
    assert torch.equal(y[:, 0], x[:, 0]) and torch.all(sizes[:, 0] == 1)


def test_table7_tokens_136_137():
    x = torch.randn(2, 197, 16)
    y, _ = sr_tome_merge_exact(x, 136, window_size=7)
    assert y.shape[1] == 137
    assert resolve_merge_geometry(196, 137, 7)["patch_tokens_out"] == 136  # historical argument 137 -> same K=34


def test_nominal_171_gives_168_169():
    g = resolve_merge_geometry(196, 171, 7)
    assert (g["k_per_window"], g["patch_tokens_out"], g["total_tokens_out"]) == (42, 168, 169)
    assert sr_tome_merge_exact(torch.randn(1, 197, 8), 171, window_size=7)[0].shape[1] == 169


def test_targets_per_window_and_sizes():
    x = torch.randn(2, 197, 16)
    y, sizes = sr_tome_merge_exact(x, 84, window_size=7, track_sizes=True)
    per_window = sizes[:, 1:].view(2, 4, 21)            # 4 windows x K=21 targets
    assert torch.all(per_window.sum(dim=-1) == 49)      # every window's tokens are accounted for
    assert torch.all(sizes[:, 1:] >= 1)


# ---------------------------------------------------------------------------- configuration
def test_table7_config_resolves_pa_off():
    sys.path.insert(0, str(ROOT / "r3" / "table7_vitb16"))
    t7 = importlib.import_module("run_table7_vitb16")
    cfg = t7.load_config(t7.DEFAULT_CONFIG)
    m = t7.build_merged(cfg, "cpu", pretrained=False)
    rc = m.s3merge_config
    assert rc["prop_attn"] is False and rc["L_s"] == 9 and rc["patch_tokens_out"] == 136 and rc["total_tokens_out"] == 137
    assert not any(isinstance(b, PropAttnBlock) for b in m.blocks)
    import tempfile, yaml
    for merge in ({k: v for k, v in cfg["merge"].items() if k != "prop_attn"}, dict(cfg["merge"], prop_attn=True)):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(dict(cfg, merge=merge), f)
        assert _raises(lambda: t7.load_config(f.name))      # PA missing or on -> explicit error


def test_pa_on_off_distinct_resolved_configs():
    on = apply_patch(_deit(), target_tokens=84, late_start=8, window_size=7, prop_attn=True)
    off = apply_patch(_deit(), target_tokens=84, late_start=8, window_size=7, prop_attn=False)
    assert on.s3merge_config != off.s3merge_config
    assert on.s3merge_config["prop_attn"] and not off.s3merge_config["prop_attn"]
    assert any(isinstance(b, PropAttnBlock) for b in on.blocks)
    assert not any(isinstance(b, PropAttnBlock) for b in off.blocks)


# ---------------------------------------------------------------------------- restore / exact-coordinate paths
def test_restore_mapping_shapes():
    for img, g in ((224, 14), (384, 24)):
        torch.manual_seed(0)
        base = timm.create_model("deit_small_patch16_224", pretrained=False, img_size=img).eval()
        T = 84 if g == 14 else 360
        ws = 7 if g == 14 else 8
        dm = SRToMeDownstream(base, target_tokens=T, late_start=8, window_size=ws)
        with torch.no_grad():
            f = dm(torch.randn(2, 3, img, img))
        assert f.shape == (2, 384, g, g), f.shape


def test_exact_coordinate_path_shapes():
    geom = PackGeometry(64, 56, 8, torch.device("cpu"))
    x = torch.randn(2, 64, 56, 32)
    win = x.view(2, 8, 8, 7, 8, 32).permute(0, 1, 3, 2, 4, 5).reshape(2, 56, 64, 32)
    _, s_w = window_scores(win)
    K_w = allocate_slot_auction(s_w, K_min=32, K_max=64, T_g=3136)
    assert torch.all(K_w.sum(dim=1) == 3136)
    packed, coords, restore, _ = adaptive_merge_pack(x, geom, K_w)
    assert packed.shape == (2, 3136, 32) and coords.shape == (2, 3136, 2)
    assert unpack_restore(packed, restore, 64, 56).shape == (2, 64, 56, 32)


def test_packed_block_with_relative_positions():
    try:
        from detectron2.modeling.backbone.vit import Block
    except Exception:
        return "SKIP (detectron2 not installed)"
    from models.adaptive_k import packed_block_forward
    blk = Block(dim=64, num_heads=4, use_rel_pos=True, window_size=0, input_size=(16, 16)).eval()
    geom = PackGeometry(16, 16, 8, torch.device("cpu"))
    x = torch.randn(1, 16, 16, 64)
    win = x.view(1, 2, 8, 2, 8, 64).permute(0, 1, 3, 2, 4, 5).reshape(1, 4, 64, 64)
    _, s_w = window_scores(win)
    packed, coords, restore, _ = adaptive_merge_pack(x, geom, allocate_uniform(s_w, 48, 48, 192))
    with torch.no_grad():
        y = packed_block_forward(blk, packed, coords, 16, 16)
    assert y.shape == packed.shape and torch.isfinite(y).all()


# ---------------------------------------------------------------------------- explicit failures, no silent skip
def test_unsupported_geometry_raises():
    assert _raises(lambda: resolve_merge_geometry(195, 84, 7))            # not a square grid
    assert _raises(lambda: resolve_merge_geometry(196, 84, 5))            # window does not divide 14
    assert _raises(lambda: resolve_merge_geometry(196, 84, 7, grid_size=24))  # declared grid mismatch
    assert _raises(lambda: resolve_merge_geometry(196, 40, 2))            # T < number of windows (49)
    assert _raises(lambda: sr_tome_merge_exact(torch.randn(1, 197, 8), 84, window_size=5))
    assert _raises(lambda: apply_patch(_deit(), target_tokens=84, late_start=8, window_size=5))
    assert _raises(lambda: apply_patch(_deit(), target_tokens=84, late_start=8, window_size=7, grid_size=24))
    assert _raises(lambda: PackGeometry(64, 56, 9, torch.device("cpu")))
    cache = GridMappingCache.get(14, W=14, window_size=7, device=torch.device("cpu"))
    assert _raises(lambda: sr_tome_merge_with_unmerge_cached(torch.randn(1, 577, 8), 84, cache))


def test_no_silent_merge_skip():
    # configurations that perform no merge by construction must be requested explicitly
    assert _raises(lambda: resolve_merge_geometry(196, 84, 14))                    # w >= grid
    assert _raises(lambda: resolve_merge_geometry(196, 84, 1))                     # w = 1
    assert _raises(lambda: sr_tome_merge_exact(torch.randn(1, 197, 8), 84, window_size=14))
    assert _raises(lambda: apply_patch(_deit(), target_tokens=84, late_start=8, window_size=14))
    m = apply_patch(_deit(), target_tokens=84, late_start=8, window_size=14, keep_all_control=True, prop_attn=False)
    assert m.s3merge_config["merges"] is False and "keep-all" in m.s3merge_config["reason"]
    x = torch.randn(1, 197, 8)
    y, s = sr_tome_merge_exact(x, 84, window_size=14, keep_all_control=True)
    assert torch.equal(x, y) and s is None
    # every valid configuration actually reduces the token count
    for T, w, n in ((84, 7, 196), (136, 7, 196), (171, 7, 196), (84, 2, 196), (360, 8, 576)):
        assert sr_tome_merge_exact(torch.randn(1, n + 1, 8), T, window_size=w)[0].shape[1] < n + 1
    # a block bound to a 14x14 grid refuses a 24x24 input instead of returning it unchanged
    assert _raises(lambda: sr_tome_merge_exact(torch.randn(1, 577, 8), 84, window_size=7, grid_size=14))


def test_double_patch_raises():
    m = apply_patch(_deit(), target_tokens=84, late_start=8, window_size=7)
    assert _raises(lambda: apply_patch(m, target_tokens=84, late_start=8, window_size=7), RuntimeError)


# ---------------------------------------------------------------------------- determinism
def test_seeded_random_selection_deterministic():
    x = torch.randn(2, 197, 16)
    torch.manual_seed(123); a, _ = sr_tome_merge_exact(x, 84, window_size=7)
    torch.manual_seed(123); b, _ = sr_tome_merge_exact(x, 84, window_size=7)
    torch.manual_seed(124); c, _ = sr_tome_merge_exact(x, 84, window_size=7)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_variance_aware_selection_deterministic():
    geom = PackGeometry(14, 14, 7, torch.device("cpu"))
    x = torch.randn(1, 14, 14, 16)
    win = x.view(1, 2, 7, 2, 7, 16).permute(0, 1, 3, 2, 4, 5).reshape(1, 4, 49, 16)
    _, s_w = window_scores(win)
    K_w = allocate_uniform(s_w, 21, 21, 84)
    p1 = adaptive_merge_pack(x, geom, K_w)[0]
    torch.manual_seed(999)
    p2 = adaptive_merge_pack(x, geom, K_w)[0]
    assert torch.equal(p1, p2)


def test_table7_runtime_token_check():
    sys.path.insert(0, str(ROOT / "r3" / "table7_vitb16"))
    t7 = importlib.import_module("run_table7_vitb16")
    cfg = t7.load_config(t7.DEFAULT_CONFIG)
    m = t7.build_merged(cfg, "cpu", pretrained=False)
    tokens, geom = t7.check_runtime_tokens(m, cfg, "cpu")
    assert tokens == 137 and geom["grid"] == 14 and geom["k_per_window"] == 34


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            r = fn()
            print(f"[PASS] {name}" + (f" {r}" if isinstance(r, str) else ""))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
