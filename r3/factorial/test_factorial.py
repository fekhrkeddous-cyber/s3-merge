"""Unit tests for the factorial arms (paper Table 4) on synthetic tensors.

SOFTWARE TESTS ONLY: CPU, no dataset, no checkpoint, no network; nothing here is a paper measurement.
Run from the repository root: python r3/factorial/test_factorial.py   (exit code 0 = all tests passed)
"""
import sys
sys.dont_write_bytecode = True
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from factorial_merge import (ARM_SPECS, FactorialModel, FactorialState, merge_stage, coord_table,  # noqa: E402
                             window_of_coord, N_PATCH, NWIN, TPW, K_FINAL)

DEV = torch.device("cpu")
C_DIM = 32
TESTS, RESULTS = [], []
S_TRAJ = [197] * 9 + [85] * 3
P_TRAJ = [197] * 6 + [169, 141, 113] + [85] * 3


def check(name):
    """Register a test; each test returns (ok, detail)."""
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def run_stages(x, arm, record=False):
    """Apply all merge stages of `arm` back to back (identity blocks in between). Returns (out, state, counts)."""
    blocks, ks, scope = ARM_SPECS[arm]
    st = FactorialState(record=record)
    counts = []
    for j, (b, k) in enumerate(zip(blocks, ks)):
        x = merge_stage(x, st, b, k, scope, j == len(ks) - 1)
        counts.append(int(x.shape[1]))
    return x, st, counts


def tiny_vit():
    from timm.models.vision_transformer import VisionTransformer
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        m = VisionTransformer(img_size=224, patch_size=16, embed_dim=48, depth=12, num_heads=3, num_classes=10)
    return m.eval().float()


def trajectory(model, inner, x):
    traj = []
    hs = [blk.register_forward_pre_hook(lambda mod, args: traj.append(int(args[0].shape[1]))) for blk in inner]
    with torch.no_grad():
        out = model(x)
    for h in hs:
        h.remove()
    return traj, out


# ------------------------------------------------------------------------------------------------- tests
@check("arm specifications (schedule, 28 removals per window, scope)")
def _():
    ok = (ARM_SPECS["SW"] == ((8,), (28,), "windowed") and ARM_SPECS["SG"] == ((8,), (28,), "global")
          and ARM_SPECS["PW"] == ((5, 6, 7, 8), (7, 7, 7, 7), "windowed")
          and ARM_SPECS["PG"] == ((5, 6, 7, 8), (7, 7, 7, 7), "global")
          and all(sum(ks) == TPW - K_FINAL == 28 for _, ks, _ in ARM_SPECS.values())
          and NWIN * (TPW - K_FINAL) == 112 and NWIN * K_FINAL == 84)
    return ok, ARM_SPECS


@check("token counts after every schedule step (84 patch tokens + CLS at the end)")
def _():
    torch.manual_seed(0)
    x = torch.randn(2, 1 + N_PATCH, C_DIM)
    got = {arm: run_stages(x.clone(), arm)[2] for arm in ARM_SPECS}
    exp = {"SW": [85], "SG": [85], "PW": [169, 141, 113, 85], "PG": [169, 141, 113, 85]}
    return got == exp, got


@check("per-block token trajectory inside a 12-block timm ViT (factorial arms + baseline)")
def _():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    res = {}
    base = tiny_vit()
    res["baseline"] = trajectory(base, list(base.blocks), x)[0] == [197] * 12
    for arm in ARM_SPECS:
        base = tiny_vit()
        inner = list(base.blocks)
        model = FactorialModel(base, arm, record=False)
        traj, out = trajectory(model, inner, x)
        res[arm] = (traj == (S_TRAJ if arm[0] == "S" else P_TRAJ) and out.shape == (2, 10)
                    and model.state.stage_calls == list(ARM_SPECS[arm][0]))
    return all(res.values()), res


@check("canonical Core control via models.sr_tome.apply_patch (PA off): trajectory, no PropAttnBlock, state hash")
def _():
    import factorial_common as C
    base = tiny_vit()
    inner = list(base.blocks)
    sha_pre = C.state_sha(list(base.state_dict().items()))
    model = C.apply_patch(base, **C.CORE_KWARGS)
    sha_post = C.state_sha(C.normalized_items(model))
    torch.manual_seed(0)
    traj, _ = trajectory(model, inner, torch.randn(2, 3, 224, 224))
    no_pa = not any(isinstance(m, C.PropAttnBlock) for m in model.modules())
    cfg = model.s3merge_config
    ok = (traj == C.EXPECT_TRAJ["core"] and no_pa and sha_pre == sha_post and cfg["prop_attn"] is False
          and cfg["total_tokens_out"] == 85 and cfg["L_s"] == 9)
    return ok, (traj, no_pa, sha_pre == sha_post, cfg)


@check("state hash unchanged by the factorial wrappers (normalized state_dict keys)")
def _():
    import factorial_common as C
    res = {}
    for arm in ARM_SPECS:
        base = tiny_vit()
        pre = C.state_sha(list(base.state_dict().items()))
        res[arm] = pre == C.state_sha(C.normalized_items(FactorialModel(base, arm)))
    return all(res.values()), res


@check("window-scope masking: windowed arms never assign across original 7x7 windows; global arms do")
def _():
    torch.manual_seed(1)
    x = torch.randn(3, 1 + N_PATCH, C_DIM)
    win = window_of_coord(DEV)
    res = {}
    for arm in ARM_SPECS:
        torch.manual_seed(5)
        _, st, _ = run_stages(x.clone(), arm, record=True)
        cross_stage = sum(int(r["cross_window_assignments"].sum()) for r in st.logs)
        cross_final = int((win[st.owner] != win[torch.arange(N_PATCH)].unsqueeze(0)).sum())
        rep_in_window = torch.equal(win[st.rep], torch.arange(NWIN).repeat_interleave(K_FINAL).unsqueeze(0).expand(3, -1))
        if ARM_SPECS[arm][2] == "windowed":
            res[arm] = cross_stage == 0 and cross_final == 0 and rep_in_window
        else:
            res[arm] = cross_stage > 0 and cross_final > 0 and rep_in_window
    return all(res.values()), res


@check("size-weighted, mass-conserving aggregation: final tokens = mean of their member patches")
def _():
    torch.manual_seed(2)
    x = torch.randn(2, 1 + N_PATCH, C_DIM)
    res = {}
    for arm in ARM_SPECS:
        torch.manual_seed(7)
        out, st, _ = run_stages(x.clone(), arm, record=True)
        B = x.shape[0]
        slot = torch.full((B, N_PATCH), -1, dtype=torch.long)
        slot.scatter_(1, st.rep, torch.arange(NWIN * K_FINAL).unsqueeze(0).expand(B, -1))
        restore = slot.gather(1, st.owner)                                  # [B,196] output slot of every patch
        cnt = torch.zeros(B, NWIN * K_FINAL).scatter_add_(1, restore, torch.ones(B, N_PATCH))
        ssum = torch.zeros(B, NWIN * K_FINAL, C_DIM).scatter_add_(1, restore.unsqueeze(-1).expand(-1, -1, C_DIM), x[:, 1:])
        expected = ssum / cnt.unsqueeze(-1)
        res[arm] = (bool((restore >= 0).all()) and torch.equal(cnt, st.size) and bool((st.size.sum(1) == N_PATCH).all())
                    and torch.allclose(out[:, 1:], expected, atol=1e-5, rtol=1e-5) and torch.equal(out[:, :1], x[:, :1]))
    return all(res.values()), res


@check("determinism under a fixed seed; different seeds give different targets")
def _():
    torch.manual_seed(3)
    x = torch.randn(2, 1 + N_PATCH, C_DIM)
    res = {}
    for arm in ARM_SPECS:
        torch.manual_seed(0)
        a, sa, _ = run_stages(x.clone(), arm)
        torch.manual_seed(0)
        b, sb, _ = run_stages(x.clone(), arm)
        torch.manual_seed(1)
        _, sc, _ = run_stages(x.clone(), arm)
        res[arm] = torch.equal(a, b) and torch.equal(sa.rep, sb.rep) and not torch.equal(sa.rep, sc.rep)
    return all(res.values()), res


@check("nested random priority: one draw per forward, identical final targets across SW/SG/PW/PG for a seed")
def _():
    torch.manual_seed(4)
    x = torch.randn(2, 1 + N_PATCH, C_DIM)
    reps, ps = {}, {}
    for arm in ARM_SPECS:
        torch.manual_seed(11)
        _, st, _ = run_stages(x.clone(), arm)
        reps[arm], ps[arm] = st.rep, st.p
    torch.manual_seed(11)
    p_ref = torch.rand(2 * NWIN, TPW)                                       # the single priority draw
    canon_q = torch.topk(p_ref, K_FINAL, dim=1, sorted=False).indices
    canon = coord_table(DEV).unsqueeze(0).expand(2, -1, -1).reshape(2 * NWIN, TPW).gather(1, canon_q).view(2, -1)
    ok = all(torch.equal(reps[a], canon) for a in reps) and all(torch.equal(ps[a], p_ref) for a in ps)
    return ok, {a: torch.equal(reps[a], canon) for a in reps}


@check("SW equals the canonical S3-Merge Core merge (models.sr_tome.sr_tome_merge_exact, same seed)")
def _():
    from models.sr_tome import sr_tome_merge_exact
    torch.manual_seed(6)
    x = torch.randn(3, 1 + N_PATCH, C_DIM)
    torch.manual_seed(21)
    canon, sizes = sr_tome_merge_exact(x.clone(), 84, window_size=7, grid_size=14, weighted_merge=False, tau=0.3,
                                       content_adaptive=False, track_sizes=False, keep_all_control=False)
    torch.manual_seed(21)
    sw, _, _ = run_stages(x.clone(), "SW")
    return canon.shape == sw.shape == (3, 85, C_DIM) and torch.allclose(canon, sw, atol=1e-6, rtol=1e-5), \
        float((canon - sw).abs().max())


@check("argmax tie rule: ties go to the lowest original patch coordinate (windowed and global)")
def _():
    x = torch.ones(1, 1 + N_PATCH, C_DIM)                                   # every similarity is equal
    res = {}
    for arm in ("SW", "SG"):
        torch.manual_seed(9)
        _, st, _ = run_stages(x.clone(), arm, record=True)
        r = st.logs[0]
        cand = st.rep.view(NWIN, K_FINAL)                                   # candidates = final targets (single-shot)
        removed = r["removed_coords"].view(-1)
        assigned = r["assigned_coords"].view(-1)
        if arm == "SW":
            win = window_of_coord(DEV)
            exp = torch.stack([cand[int(win[c])].min() for c in removed])
        else:
            exp = torch.full_like(assigned, int(cand.min()))
        res[arm] = torch.equal(assigned, exp) and r["exact_argmax_ties"] == NWIN * (TPW - K_FINAL)
    return all(res.values()), res


@check("in-forward guards: non-FP32 input is rejected")
def _():
    st = FactorialState()
    try:
        merge_stage(torch.randn(1, 1 + N_PATCH, C_DIM, dtype=torch.float64), st, 8, 28, "windowed", True)
    except AssertionError:
        return True, ""
    return False, "float64 input accepted"


@check("timing arm orders reproduce the paper's randomized order manifest (random.Random(20261103))")
def _():
    import timing
    o = timing.make_orders()
    exp_latency = {"0": ["baseline", "SW", "PG", "SG", "PW"], "1": ["PG", "SW", "baseline", "PW", "SG"],
                   "2": ["SW", "PW", "baseline", "PG", "SG"], "3": ["SG", "PG", "SW", "PW", "baseline"],
                   "4": ["SW", "PG", "SG", "baseline", "PW"], "5": ["baseline", "SG", "PG", "SW", "PW"],
                   "6": ["baseline", "PW", "SW", "PG", "SG"], "7": ["PW", "SW", "baseline", "PG", "SG"],
                   "8": ["baseline", "PG", "SW", "PW", "SG"], "9": ["SW", "PW", "baseline", "PG", "SG"]}
    exp_profile = {"0": ["baseline", "SW", "SG", "PG", "PW"], "1": ["baseline", "PG", "SG", "SW", "PW"],
                   "2": ["SW", "SG", "baseline", "PW", "PG"]}
    exp_micro = {"0": ["SW", "PW", "baseline", "PG", "SG"], "1": ["PG", "SW", "SG", "PW", "baseline"],
                 "2": ["PW", "SG", "SW", "baseline", "PG"], "3": ["baseline", "PG", "PW", "SW", "SG"],
                 "4": ["baseline", "SG", "SW", "PW", "PG"]}
    ok = o["latency"] == exp_latency and o["profile"] == exp_profile and o["merge_micro"] == exp_micro
    return ok, o


@check("config_table4.yaml mirrors the command-line defaults")
def _():
    try:
        import yaml
    except ImportError:
        return True, "SKIPPED (pyyaml not installed)"
    import factorial_common as C
    import accuracy, timing, analyze_accuracy, analyze_timing
    y = yaml.safe_load((HERE / "config_table4.yaml").read_text())
    acc = accuracy.parse_args(["--arm", "SW", "--seed", "0", "--data-dir", "."])
    tim = timing.parse_args(["--invocation", "0"])
    aa, at = analyze_accuracy.parse_args([]), analyze_timing.parse_args([])
    checks = {
        "ckpt_sha": y["model"]["checkpoint_file_sha256"] == C.CKPT_FILE_SHA == acc.expected_sha256 == tim.expected_sha256,
        "state_sha": y["model"]["state_sha256"] == C.STATE_SHA == acc.expected_state_sha256,
        "core": y["arms"]["core"]["apply_patch"] == C.CORE_KWARGS,
        "arms": all(tuple(y["arms"][a]["merge_after_block_index"]) == ARM_SPECS[a][0]
                    and tuple(y["arms"][a]["removals_per_window"]) == ARM_SPECS[a][1]
                    and y["arms"][a]["scope"] == ARM_SPECS[a][2] for a in ARM_SPECS),
        "loader": (y["accuracy"]["batch_size"], y["accuracy"]["num_workers"]) == (accuracy.BATCH_SIZE, accuracy.NUM_WORKERS),
        "shards": y["accuracy"]["shard_pattern"] == acc.shard_pattern,
        "seeds": y["accuracy"]["seeds"] == aa.seeds,
        "timing": (y["timing"]["batch_sizes"] == tim.batch_sizes and y["timing"]["arms"] == tim.arms
                   and (y["timing"]["warmup"], y["timing"]["timed"], y["timing"]["profiled_forwards"])
                   == (timing.WARMUP, timing.TIMED, timing.PROFILED)
                   and (y["timing"]["select_seed"], y["timing"]["input_seed"], y["timing"]["order_seed"])
                   == (timing.SELECT_SEED, timing.INPUT_SEED, timing.ORDER_SEED)
                   and y["timing"]["invocations"] == timing.INVOCATIONS),
        "analysis": (y["analysis"]["accuracy"]["margin_pp"] == aa.margin_pp and y["analysis"]["timing"]["margin_pct"] == at.margin_pct
                     and y["analysis"]["timing"]["bootstrap"] == at.bootstrap and at.n_invocations == timing.INVOCATIONS["latency"]),
    }
    return all(checks.values()), checks


def main():
    for name, fn in TESTS:
        try:
            ok, detail = fn()
        except Exception:
            ok, detail = False, traceback.format_exc(limit=3)
        RESULTS.append((name, bool(ok)))
        note = f"  ({detail})" if ok and isinstance(detail, str) and detail.startswith("SKIPPED") else ""
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{note}" + ("" if ok else f"  {detail}"), flush=True)
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} tests passed (software tests on synthetic tensors; not paper measurements)")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
