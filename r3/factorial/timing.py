"""Factorial-study timing driver (paper Table 4 latency columns; Supplementary Table S19). ONE process invocation per call.

Arms: baseline (unmerged DeiT-S), SW, SG, PW, PG (FactorialModel, record=False; PA off; FP32; no autocast; no
torch.compile). Batch sizes B in {1, 8, 32}. Fixed device-resident synthetic inputs: torch.randn float32
[B,3,224,224] from a CPU torch.Generator seeded 20260718, drawn for B = 1, 8, 32 in that order; identical for every
arm and invocation. Arm order per invocation: random.Random(20261103), drawn in the fixed sequence latency
invocations 0..9, profile 0..2, merge_micro 0..4, each a random.sample of the five arms; within an arm, B runs in
the order 1, 8, 32.

Modes
  latency      (paper) For each arm in the randomized order of this invocation, build the model, then for each B:
               torch.manual_seed(1000); 30 warm-up forwards (synchronize after each); 200 timed forwards, each:
               synchronize; t0 = perf_counter_ns(); model(x); synchronize; t1 = perf_counter_ns(), under no_grad.
               Timing boundary = the whole forward incl. priority draw, matching, aggregation and all bookkeeping.
               nvidia-smi telemetry is sampled before and after every (arm, B) cell, never inside a timed loop.
               No profiler in this mode.
  profile      (supplementary, not in Table 4) torch.profiler (CPU + CUDA activities). merge_stage is wrapped in
               record_function('factorial_merge_stage') for EVERY arm (identical instrumentation; baseline never
               merges); each forward is wrapped in record_function('factorial_forward'). 30 warm-up, then 20 profiled
               forwards per (arm, B). Kernel sums are never mixed with wall-clock latency.
  merge_micro  (supplementary, not in Table 4) Isolated merge cost: the input and the FactorialState before every
               merge stage are captured from one forward; each stage is then timed alone (30 warm-up + 200 timed,
               synchronize before/after; state restored outside the timed region). Per-forward merge cost = sum
               over stages. Never a decomposition of end-to-end latency.

Output: <output-dir>/<mode>_inv<NN>.json (never overwritten).

--smoke is for SOFTWARE TESTING ONLY: reduced counts (3 warm-up, 10 timed, 3 profiled) plus a correctness check
(timing-path logits bitwise equal to the accuracy-path logits; trajectories; FP32); CPU allowed. Smoke outputs
are marked "smoke": true; their timings are not paper measurements.
"""
import sys
sys.dont_write_bytecode = True
import argparse, copy, hashlib, json, random, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factorial_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import factorial_merge as FM  # noqa: E402

ARMS5 = ["baseline", "SW", "SG", "PW", "PG"]          # list order matters for random.sample (order draws)
BS_DEFAULT = [1, 8, 32]
ORDER_SEED, INPUT_SEED, SELECT_SEED = 20261103, 20260718, 1000
INVOCATIONS = {"latency": 10, "profile": 3, "merge_micro": 5}   # draw sequence of the randomized arm orders
WARMUP, TIMED, PROFILED = 30, 200, 20                            # paper protocol
SMOKE_WARMUP, SMOKE_TIMED, SMOKE_PROFILED = 3, 10, 3             # software test only
# sha256 of the CPU-generated input tensors (torch 2.1.2) for the default seed and batch-size list [1, 8, 32]
INPUT_TENSOR_SHA256 = {1: "49167eef0df80828c3ea073066eaaebdc5aca87c3e2c4a3f7587509abd542a74",
                       8: "4c455aee2a20fbd51ab739aaf18f7f35ab8c29ade756a9b0f8d02cc87c9f206d",
                       32: "11967863a78b6535a7619104329d21e6640d19dd5a1c720d3148c2db1afbe86d"}


def make_orders(seed=ORDER_SEED):
    rng = random.Random(seed)
    return {mode: {str(i): rng.sample(ARMS5, 5) for i in range(n)} for mode, n in INVOCATIONS.items()}


def make_inputs(batch_sizes, seed=INPUT_SEED):
    """CPU tensors, drawn in the order of batch_sizes from one generator; returns (dict B->tensor, sha256 dict)."""
    g = torch.Generator().manual_seed(seed)
    xs = {b: torch.randn(b, 3, 224, 224, generator=g, dtype=torch.float32) for b in batch_sizes}
    return xs, {b: hashlib.sha256(v.numpy().tobytes()).hexdigest() for b, v in xs.items()}


def build_timing_model(arm, ckpt, dev, pretrained, expected_state_sha):
    """Same builders as the accuracy path WITHOUT hooks (hooks would add per-block host overhead to timing)."""
    base = C.build_base(ckpt, dev, pretrained)
    sha_pre = C.state_sha(list(base.state_dict().items()))
    model = base if arm == "baseline" else C.FactorialModel(base, arm, record=False)
    sha_post = C.state_sha(C.normalized_items(model))
    assert sha_pre == sha_post and (expected_state_sha is None or sha_post == expected_state_sha), f"state sha mismatch {arm}"
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert not any(isinstance(m, C.PropAttnBlock) for m in model.modules())
    train = {type(m).__name__ for m in model.modules() if m.training}
    assert train <= {"FactorialModel", "FactorialMergeBlock"}, train
    return model


def time_cell(model, x, warm, iters, dev):
    sync = C.sync_fn(dev)                                     # torch.cuda.synchronize on the default CUDA device
    torch.manual_seed(SELECT_SEED)
    t_w, t = [], []
    with torch.no_grad():
        for _ in range(warm):
            s = time.perf_counter_ns(); model(x); sync(); t_w.append((time.perf_counter_ns() - s) / 1e6)
        for _ in range(iters):
            sync()
            s = time.perf_counter_ns()
            model(x)
            sync()
            t.append((time.perf_counter_ns() - s) / 1e6)
    return t_w, t


def run_latency(build, xs, order, warm, iters, dev):
    res = {"arm_order": order, "cells": {}}
    for arm in order:
        model = build(arm)
        for B in xs:
            pre = C.gpu_snapshot()
            u0 = C.utc()
            tw, t = time_cell(model, xs[B], warm, iters, dev)
            res["cells"][f"{arm}|{B}"] = {"arm": arm, "B": B, "start_utc": u0, "end_utc": C.utc(), "warmup_ms": tw,
                                          "times_ms": t, "mean_ms": float(np.mean(t)), "median_ms": float(np.median(t)),
                                          "telemetry_before": pre, "telemetry_after": C.gpu_snapshot()}
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return res


def _kernels(evt):
    out = list(getattr(evt, "kernels", []) or [])
    for ch in evt.cpu_children:
        out += _kernels(ch)
    return out


def run_profile(build, xs, order, warm, n_prof, dev):
    from torch.profiler import profile, ProfilerActivity, record_function
    orig = FM.merge_stage

    def wrapped(*a, **k):
        with record_function("factorial_merge_stage"):
            return orig(*a, **k)
    FM.merge_stage = wrapped                                   # identical for every arm (module global, read at call)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    res = {"arm_order": order, "cells": {}, "activities": [str(a) for a in acts]}
    try:
        for arm in order:
            model = build(arm)
            for B in xs:
                x = xs[B]
                torch.manual_seed(SELECT_SEED)
                with torch.no_grad():
                    for _ in range(warm):
                        model(x); C.sync(dev)
                    with profile(activities=acts) as prof:
                        for _ in range(n_prof):
                            with record_function("factorial_forward"):
                                model(x)
                            C.sync(dev)
                ev = prof.events()
                fw = [e for e in ev if e.name == "factorial_forward"]
                ms = [e for e in ev if e.name == "factorial_merge_stage"]
                k_fw = [_kernels(e) for e in fw]
                k_ms = [k for e in ms for k in _kernels(e)]
                names = {}
                for ks in k_fw:
                    for k in ks:
                        names[k.name] = names.get(k.name, 0) + 1
                attn = sorted({n for n in names if any(s in n.lower() for s in ("fmha", "flash", "attention", "efficient", "softmax"))})
                res["cells"][f"{arm}|{B}"] = {
                    "arm": arm, "B": B, "n_profiled_forwards": len(fw),
                    "kernel_launches_per_forward": float(np.mean([len(k) for k in k_fw])),
                    "kernel_time_us_per_forward": float(np.mean([sum(k.duration for k in ks) for ks in k_fw])),
                    "merge_stage_calls_per_forward": len(ms) / max(1, len(fw)),
                    "merge_kernel_launches_per_forward": len(k_ms) / max(1, len(fw)),
                    "merge_kernel_time_us_per_forward": sum(k.duration for k in k_ms) / max(1, len(fw)),
                    "profiled_forward_cpu_wall_us_mean": float(np.mean([e.cpu_time_total for e in fw])),
                    "attention_related_kernel_names": attn,
                    "top_kernels": sorted(names.items(), key=lambda kv: -kv[1])[:15]}
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        FM.merge_stage = orig
    return res


def _snap_state(st):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in vars(st).items()}


def run_merge_micro(build, xs, order, warm, iters, dev):
    orig = FM.merge_stage
    sync = C.sync_fn(dev)
    res = {"arm_order": [a for a in order if a != "baseline"], "cells": {}}
    for arm in res["arm_order"]:
        model = build(arm)
        for B in xs:
            cap = []

            def capture(x, state, block_index, k, scope, is_last):
                cap.append((x.clone(), _snap_state(state), block_index, k, scope, is_last))
                return orig(x, state, block_index, k, scope, is_last)
            FM.merge_stage = capture
            torch.manual_seed(SELECT_SEED)
            try:
                with torch.no_grad():
                    model(xs[B])
            finally:
                FM.merge_stage = orig
            C.sync(dev)
            stages = []
            for (x, snap, b, k, scope, last) in cap:
                tt = []
                with torch.no_grad():
                    for it in range(warm + iters):
                        st = FM.FactorialState(record=False)
                        for kk, vv in snap.items():
                            setattr(st, kk, vv.clone() if torch.is_tensor(vv) else copy.deepcopy(vv))
                        sync()
                        s = time.perf_counter_ns()
                        orig(x, st, b, k, scope, last)
                        sync()
                        if it >= warm:
                            tt.append((time.perf_counter_ns() - s) / 1e6)
                stages.append({"after_block": b, "k": k, "scope": scope, "first_stage_includes_priority_draw": snap["p"] is None,
                               "times_ms": tt, "mean_ms": float(np.mean(tt)), "median_ms": float(np.median(tt))})
            res["cells"][f"{arm}|{B}"] = {"arm": arm, "B": B, "n_stages": len(stages), "stages": stages,
                                          "sum_of_stage_means_ms": float(sum(s["mean_ms"] for s in stages))}
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return res


def smoke_correctness(build, xs, arms, ckpt, dev, pretrained, expected_state_sha):
    """Timing-path model == accuracy-path model (bitwise logits), trajectories, FP32, no autocast."""
    out = {}
    for arm in arms:
        tm = build(arm)
        ref = C.build_model(arm, ckpt, dev, pretrained, expected_state_sha)   # accuracy path (hooks + assertions)
        ok = True
        for B in xs:
            torch.manual_seed(SELECT_SEED)
            with torch.no_grad():
                a = tm(xs[B])
            torch.manual_seed(SELECT_SEED)
            b = C.forward(ref, xs[B])                          # raises on trajectory/mask/dtype violation
            ok &= torch.equal(a, b) and a.dtype == torch.float32
        out[arm] = {"timing_path_logits_bitwise_equal_accuracy_path": bool(ok), "trajectory": C.EXPECT_TRAJ[arm],
                    "autocast": torch.is_autocast_enabled()}
        del tm, ref
    out["PASS"] = all(v["timing_path_logits_bitwise_equal_accuracy_path"] and not v["autocast"]
                      for k, v in out.items() if k != "PASS")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Factorial-study timing: ONE process invocation. Defaults = paper protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=list(INVOCATIONS), default="latency",
                    help="latency = paper measurement; profile / merge_micro = supplementary diagnostics")
    ap.add_argument("--invocation", type=int, required=True,
                    help="invocation index (selects the randomized arm order): latency 0-9, profile 0-2, merge_micro 0-4")
    ap.add_argument("--output-dir", default="runs/r3_factorial/timing", help="directory for <mode>_inv<NN>.json")
    ap.add_argument("--arms", nargs="+", choices=ARMS5, default=ARMS5,
                    help="arms to time (always run in this invocation's randomized order)")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=BS_DEFAULT,
                    help="batch sizes, timed in this order; inputs are drawn in this order")
    ap.add_argument("--warmup", type=int, default=None, help="warm-up forwards per (arm, B); default 30 (3 with --smoke)")
    ap.add_argument("--timed", type=int, default=None, help="timed forwards per (arm, B); default 200 (10 with --smoke)")
    ap.add_argument("--profiled-forwards", type=int, default=None, help="profile mode; default 20 (3 with --smoke)")
    ap.add_argument("--checkpoint", default=None,
                    help=f"local model.safetensors (default: Hugging Face hub {C.HF_HUB_ID})")
    ap.add_argument("--expected-sha256", default=C.CKPT_FILE_SHA, help="checkpoint file sha256 ('none' disables the check)")
    ap.add_argument("--expected-state-sha256", default=C.STATE_SHA, help="sha256 of the loaded weights ('none' disables)")
    ap.add_argument("--device", default="cuda", help="torch device (the paper used one H100 NVL)")
    ap.add_argument("--smoke", action="store_true",
                    help="SOFTWARE TEST ONLY: reduced counts + correctness check; timings are not paper measurements")
    ap.add_argument("--no-pretrained", action="store_true", help="random (seeded) weights; only allowed with --smoke")
    a = ap.parse_args(argv)
    if a.no_pretrained and not a.smoke:
        ap.error("--no-pretrained is only allowed with --smoke")
    if not 0 <= a.invocation < INVOCATIONS[a.mode]:
        ap.error(f"--invocation for mode {a.mode} must be in 0..{INVOCATIONS[a.mode] - 1}")
    if len(set(a.batch_sizes)) != len(a.batch_sizes):
        ap.error("--batch-sizes must be distinct")
    return a


def main(argv=None):
    a = parse_args(argv)
    warm = a.warmup if a.warmup is not None else (SMOKE_WARMUP if a.smoke else WARMUP)
    iters = a.timed if a.timed is not None else (SMOKE_TIMED if a.smoke else TIMED)
    n_prof = a.profiled_forwards if a.profiled_forwards is not None else (SMOKE_PROFILED if a.smoke else PROFILED)
    orders = make_orders(ORDER_SEED)
    order = [arm for arm in orders[a.mode][str(a.invocation)] if arm in a.arms]
    cfg = {"script": "r3/factorial/timing.py", "mode": a.mode, "invocation": a.invocation, "smoke": bool(a.smoke),
           "output_dir": str(Path(a.output_dir).resolve()), "device": a.device, "arm_order": order,
           "batch_sizes": list(a.batch_sizes), "warmup": warm, "timed": iters,
           "profiled_forwards": n_prof if a.mode == "profile" else None,
           "select_seed": SELECT_SEED, "order_seed": ORDER_SEED, "input_seed": INPUT_SEED,
           "inputs": "torch.randn float32 [B,3,224,224], CPU generator seeded 20260718, drawn in --batch-sizes order",
           "boundary": "synchronize; t0 = perf_counter_ns(); model(x); synchronize; t1 (no_grad, whole forward)",
           "model": {"name": C.MODEL_NAME, "weights": "none (random, smoke)" if a.no_pretrained else C.HF_HUB_ID,
                     "checkpoint": a.checkpoint, "expected_sha256": C.none_or(a.expected_sha256),
                     "expected_state_sha256": C.none_or(a.expected_state_sha256), "pretrained": not a.no_pretrained},
           "arm_definitions": {arm: {"merge_after_block_index": list(C.ARM_SPECS[arm][0]),
                                     "removals_per_window": list(C.ARM_SPECS[arm][1]), "scope": C.ARM_SPECS[arm][2]}
                               for arm in order if arm != "baseline"},
           "inference": {"precision": "FP32", "autocast": False, "torch_compile": False, "cuda_graphs": False,
                         "tf32_matmul": False, "tf32_cudnn": True, "cudnn_benchmark": False,
                         "proportional_attention": False}}
    C.print_config(cfg)
    if a.smoke:
        print(C.SMOKE_NOTICE, flush=True)
    out_dir = Path(a.output_dir)
    out = out_dir / f"{a.mode}_inv{a.invocation:02d}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    dev = torch.device(a.device)
    C.set_backends()
    assert not torch.is_autocast_enabled()
    rec = {"mode": a.mode, "invocation": a.invocation, "smoke": bool(a.smoke), "started_utc": C.utc(),
           "argv": sys.argv, "config": cfg, "source_sha256": C.source_hashes(__file__)}
    if a.smoke:
        rec["notice"] = C.SMOKE_NOTICE
    apps = C.compute_apps() if dev.type == "cuda" else None        # before CUDA initialization
    rec["compute_apps_at_start_before_cuda_init"] = apps
    if apps:
        print(f"WARNING: other compute processes are present on the GPU (pid,used_memory): {apps!r}. "
              "Timing on a shared GPU is not comparable to the paper's exclusive-GPU measurements.",
              file=sys.stderr, flush=True)
    ckpt, ck_sha = C.resolve_checkpoint(a.checkpoint, C.none_or(a.expected_sha256), pretrained=not a.no_pretrained)
    rec.update({"checkpoint": ckpt, "checkpoint_file_sha256": ck_sha})
    exp_state = None if a.no_pretrained else C.none_or(a.expected_state_sha256)

    def build(arm):
        return build_timing_model(arm, ckpt, dev, not a.no_pretrained, exp_state)
    xs_cpu, in_sha = make_inputs(a.batch_sizes, INPUT_SEED)
    rec["inputs"] = {"tensor_sha256": {str(b): h for b, h in in_sha.items()},
                     "match_paper_inputs": ({str(b): in_sha[b] == INPUT_TENSOR_SHA256[b] for b in in_sha}
                                            if list(a.batch_sizes) == BS_DEFAULT else None)}
    xs = {b: v.to(dev) for b, v in xs_cpu.items()}                 # device-resident during timing
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        rec["correctness"] = smoke_correctness(build, xs, order, ckpt, dev, not a.no_pretrained, exp_state)
    if a.mode == "latency":
        rec["result"] = run_latency(build, xs, order, warm, iters, dev)
    elif a.mode == "profile":
        rec["result"] = run_profile(build, xs, order, warm, n_prof, dev)
    else:
        rec["result"] = run_merge_micro(build, xs, order, warm, iters, dev)
    rec["env"] = C.env_record(dev)
    rec["finished_utc"] = C.utc()
    C.write_atomic_json(out, rec)
    print(json.dumps({"out": str(out), "sha256": C.sha256_file(out), "smoke": bool(a.smoke),
                      "correctness_PASS": rec.get("correctness", {}).get("PASS")}, indent=1))
    if a.smoke:
        print(C.SMOKE_NOTICE, flush=True)
        if not rec["correctness"]["PASS"]:
            raise SystemExit("SMOKE CORRECTNESS FAILURE (timing path != accuracy path)")


if __name__ == "__main__":
    main()
