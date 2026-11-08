#!/usr/bin/env python3
"""Unified timing driver for paper Table 1 / Fig. 5 (Supplementary Tables S21-S22): ONE independent process
invocation per call. Run ten latency invocations (0..9) with timing_campaign.py.

Arms: baseline, tome_r10, tome_r12 (official ToMe, native prop_attn), s3_acc, s3_speed (models/sr_tome.py apply_patch,
PA on). B in {1, 8, 32}. FP32 eager, no autocast, no torch.compile, TF32 matmul off.
Inputs: synthetic torch.randn float32 [B, 3, 224, 224] (CPU generator seeded 20260718, drawn for B = 1, 8, 32 in that
order), moved to the GPU once per invocation (device-resident), identical for every arm.

  --mode latency  (paper measurement) for each arm in the randomized order of this invocation: build it, then for
                  B in (1, 8, 32): torch.manual_seed(1000); 30 warm-up forwards (synchronize after each); 200 timed
                  forwards [synchronize; t0 = perf_counter_ns(); model(x); synchronize; t1]. The timed boundary is the
                  whole forward incl. target draw, matching, aggregation, size tracking and the PA bias. No profiler and
                  no hooks. nvidia-smi telemetry is recorded before and after every (arm, B) cell, outside the timed
                  region. Per cell the mean of the 200 timed forwards is the invocation's latency.
  --mode profile  (separate invocations, never combined with latency) torch.profiler; each family's merge entry point
                  is wrapped in record_function('tu_merge') identically (official ToMe: tome.patch.timm
                  bipartite_soft_matching, plus merge_wavg as 'tu_merge_agg'; S3: models.sr_tome.sr_tome_merge_exact);
                  30 warm-up + 20 profiled forwards per (arm, B): kernel launches, merge calls/launches, kernel-time
                  sums, attention-related kernel names.
  --smoke         SOFTWARE TEST ONLY: 3 warm-up / 10 timed / 3 profiled forwards, CPU and --no-pretrained allowed, plus
                  a correctness check (two independent builds per arm, the second hooked: bitwise equality for the
                  baseline, allclose 1e-4 + identical argmax for ToMe / S3, whose aggregation uses scatter atomics on
                  CUDA; per-block token trajectory, attn_mask pattern and final token count). Not a paper measurement.

Output: <output-dir>/<mode>_inv<NN>.json (never overwritten).
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_common as U  # noqa: E402  (also puts the repository root on sys.path)
import numpy as np  # noqa: E402
import torch  # noqa: E402

SMOKE_COUNTS = {"warmup": 3, "iters": 10, "profiled_forwards": 3}


def build_parser():
    ap = argparse.ArgumentParser(description="Unified Table 1 timing: one process invocation (latency or profile).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["latency", "profile"], default="latency",
                    help="latency = paper measurement; profile = separate torch.profiler invocations")
    ap.add_argument("--invocation", type=int, required=True,
                    help="invocation index; selects the randomized arm order (latency 0-9, profile 0-2)")
    ap.add_argument("--output-dir", required=True, help="directory for <mode>_inv<NN>.json")
    ap.add_argument("--arms", nargs="+", choices=U.ARMS, default=list(U.ARMS), help="arms to time")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=list(U.BATCH_SIZES),
                    help="batch sizes, timed in this order within each arm")
    ap.add_argument("--warmup", type=int, default=U.WARMUP, help="synchronized warm-up forwards per (arm, B)")
    ap.add_argument("--iters", type=int, default=U.TIMED, help="timed forwards per (arm, B)")
    ap.add_argument("--profiled-forwards", type=int, default=U.PROFILED_FORWARDS, help="profiled forwards per (arm, B) "
                    "in --mode profile")
    ap.add_argument("--select-seed", type=int, default=U.SELECT_SEED, help="torch.manual_seed before every (arm, B) cell")
    ap.add_argument("--input-seed", type=int, default=U.INPUT_SEED, help="CPU generator seed of the synthetic inputs")
    ap.add_argument("--order-seed", type=int, default=U.ORDER_SEED, help="seed of the randomized arm orders")
    ap.add_argument("--order", default=None, help="explicit comma-separated arm order (overrides the randomized order)")
    U.add_model_args(ap)
    ap.add_argument("--smoke", action="store_true", help="SOFTWARE TEST ONLY (see module docstring); not a paper measurement")
    return ap


def make_sync(dev):
    return torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)


def cell(model, x, warm, iters, select_seed, sync):
    torch.manual_seed(select_seed)
    tw, t = [], []
    with torch.no_grad():
        for _ in range(warm):
            s = time.perf_counter_ns(); model(x); sync(); tw.append((time.perf_counter_ns() - s) / 1e6)  # noqa: E702
        for _ in range(iters):
            sync(); s = time.perf_counter_ns(); model(x); sync()  # noqa: E702
            t.append((time.perf_counter_ns() - s) / 1e6)
    return tw, t


def run_latency(builder, order, xs, batch_sizes, warm, iters, select_seed, sync):
    res = {"order": order, "cells": {}, "builds": {}}
    for name in order:
        m = builder.build(name)
        res["builds"][name] = m["info"]
        for B in batch_sizes:
            pre, u0 = U.gpu_snapshot(), U.utc()
            tw, t = cell(m["model"], xs[B], warm, iters, select_seed, sync)
            res["cells"][f"{name}|{B}"] = {"method": name, "B": B, "start_utc": u0, "end_utc": U.utc(), "warmup_ms": tw,
                                           "times_ms": t, "mean_ms": float(np.mean(t)), "median_ms": float(np.median(t)),
                                           "telemetry_before": pre, "telemetry_after": U.gpu_snapshot()}
        del m
        if builder.device.type == "cuda":
            torch.cuda.empty_cache()
    return res


def _kernels(e):
    out = list(getattr(e, "kernels", []) or [])
    for ch in e.cpu_children:
        out += _kernels(ch)
    return out


def run_profile(builder, tome_mod, order, xs, batch_sizes, warm, n, select_seed, sync):
    import models.sr_tome as SRT
    from torch.profiler import profile, ProfilerActivity, record_function
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if builder.device.type == "cuda" else [])
    orig = {"bsm": tome_mod.bipartite_soft_matching, "wavg": tome_mod.merge_wavg, "srt": SRT.sr_tome_merge_exact}

    def wrap(f, name):
        def g(*a, **k):
            with record_function(name):
                return f(*a, **k)
        return g
    # 'tu_merge' marks ONE merge event (ToMe: the matching call of a block; S3: sr_tome_merge_exact);
    # 'tu_merge_agg' marks ToMe's separate weighted aggregation call of the same event. Merge kernels = both ranges.
    tome_mod.bipartite_soft_matching = wrap(orig["bsm"], "tu_merge")
    tome_mod.merge_wavg = wrap(orig["wavg"], "tu_merge_agg")
    SRT.sr_tome_merge_exact = wrap(orig["srt"], "tu_merge")
    res = {"order": order, "cells": {}, "builds": {}}
    try:
        for name in order:
            m = builder.build(name)
            res["builds"][name] = m["info"]
            for B in batch_sizes:
                torch.manual_seed(select_seed)
                with torch.no_grad():
                    for _ in range(warm):
                        m["model"](xs[B]); sync()  # noqa: E702
                    with profile(activities=acts) as prof:
                        for _ in range(n):
                            with record_function("tu_forward"):
                                m["model"](xs[B])
                            sync()
                ev = prof.events()
                fw = [e for e in ev if e.name == "tu_forward"]
                ev_m = [e for e in ev if e.name == "tu_merge"]
                mg = [e for e in ev if e.name in ("tu_merge", "tu_merge_agg")]
                kf = [_kernels(e) for e in fw]
                km = [k for e in mg for k in _kernels(e)]
                names = {}
                for ks in kf:
                    for k in ks:
                        names[k.name] = names.get(k.name, 0) + 1
                res["cells"][f"{name}|{B}"] = {
                    "method": name, "B": B, "n_forwards": len(fw),
                    "kernel_launches_per_forward": float(np.mean([len(k) for k in kf])),
                    "kernel_time_us_per_forward": float(np.mean([sum(k.duration for k in ks) for ks in kf])),
                    "merge_events_per_forward": len(ev_m) / max(1, len(fw)),
                    "merge_ranges_per_forward": len(mg) / max(1, len(fw)),
                    "merge_kernel_launches_per_forward": len(km) / max(1, len(fw)),
                    "merge_kernel_time_us_per_forward": sum(k.duration for k in km) / max(1, len(fw)),
                    "merge_range_cpu_wall_us_per_forward_profiled": sum(e.cpu_time_total for e in mg) / max(1, len(fw)),
                    "attention_kernel_names": sorted({k for k in names if any(s in k.lower() for s in ("fmha", "flash", "attention", "softmax", "efficient"))}),
                    "top_kernels": sorted(names.items(), key=lambda kv: -kv[1])[:15]}
            del m
            if builder.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        tome_mod.bipartite_soft_matching, tome_mod.merge_wavg = orig["bsm"], orig["wavg"]
        SRT.sr_tome_merge_exact = orig["srt"]
    return res


def smoke_correctness(builder, arms, xs, batch_sizes, select_seed, sync):
    out = {}
    for name in arms:
        a, b = builder.build(name), builder.build(name)
        hk = U.Hooks(b["inner"], U.final_norm_of(b["model"]))
        ok, detail = True, {}
        exact = name == "baseline"
        for B in batch_sizes:
            torch.manual_seed(select_seed)
            with torch.no_grad():
                la = a["model"](xs[B])
            hk.reset()
            torch.manual_seed(select_seed)
            with torch.no_grad():
                lb = b["model"](xs[B])
            sync()
            eq = torch.equal(la, lb) if exact else (torch.allclose(la, lb, rtol=0, atol=1e-4) and torch.equal(la.argmax(-1), lb.argmax(-1)))
            struct = (hk.block_in == U.expected_block_inputs(name)
                      and [i for i, v in enumerate(hk.mask) if v] == U.expected_attn_mask_blocks(name)
                      and hk.final == U.expected_final_tokens(name))
            ok &= bool(eq) and struct and la.dtype == torch.float32
            detail[B] = {"max_abs": float((la - lb).abs().max()), "struct": struct, "block_inputs": list(hk.block_in),
                         "attn_mask_blocks": [i for i, v in enumerate(hk.mask) if v], "final_tokens": hk.final}
        out[name] = {"PASS": ok, "detail": detail, "criterion": "bitwise" if exact else "allclose 1e-4 + argmax"}
        hk.remove()
        del a, b
    out["PASS"] = all(v["PASS"] for k, v in out.items() if k != "PASS")
    return out


def main(argv=None):
    a = build_parser().parse_args(argv)
    dev = U.validate_run_args(a)
    counts = dict(SMOKE_COUNTS) if a.smoke else {"warmup": a.warmup, "iters": a.iters, "profiled_forwards": a.profiled_forwards}
    order = U.order_for(a.mode, a.invocation, a.arms, a.order_seed, a.order)
    out_dir = Path(a.output_dir)
    out = out_dir / f"{a.mode}_inv{a.invocation:02d}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    cfg = {"script": "timing_unified.py", "mode": a.mode, "invocation": a.invocation, "output": str(out),
           "arms": a.arms, "order": order, "batch_sizes": a.batch_sizes, "counts": counts,
           "select_seed": a.select_seed, "input_seed": a.input_seed, "order_seed": a.order_seed,
           "explicit_order": a.order, "device": str(dev), "precision": "FP32 eager; autocast off; no torch.compile; "
           "TF32 matmul off; cuDNN TF32 on (patch-embedding conv)", "checkpoint": a.checkpoint,
           "expected_sha256": a.expected_sha256, "expected_state_sha256": a.expected_state_sha256,
           "pretrained": not a.no_pretrained, "tome_path": a.tome_path, "verify_tome": not a.no_verify_tome,
           "methods": {k: U.METHODS[k] for k in a.arms}, "smoke": a.smoke}
    U.print_config(cfg)
    apps = U.compute_apps()    # before CUDA initialisation
    if apps and not apps.startswith("unavailable"):
        print(f"WARNING: other compute processes on the GPU at start (latency may be affected): {apps}", flush=True)
    _, tome_mod, tome_hashes = U.load_tome(a.tome_path, verify=not a.no_verify_tome)
    ckpt = U.check_checkpoint(a)
    U.set_backends()
    assert not torch.is_autocast_enabled()
    builder = U.ArmBuilder(dev, a.checkpoint, pretrained=not a.no_pretrained, expected_state_sha256=a.expected_state_sha256,
                           init_seed=0 if (a.smoke and a.no_pretrained) else None)
    sync = make_sync(dev)
    rec = {"mode": a.mode, "invocation": a.invocation, "started_utc": U.utc(), "argv": sys.argv, "smoke": a.smoke,
           "scientific": not a.smoke, "config": cfg, "checkpoint": ckpt, "tome_source_sha256": tome_hashes,
           "compute_apps_at_start_before_cuda_init": apps}
    if a.smoke:
        rec["label"] = U.SMOKE_BANNER
    xs_cpu, rec["inputs"] = U.make_inputs(a.batch_sizes, a.input_seed)
    xs = {b: x.to(dev) for b, x in xs_cpu.items()}      # device-resident for the whole invocation
    if a.smoke:
        rec["correctness"] = smoke_correctness(builder, order, xs, a.batch_sizes, a.select_seed, sync)
    if a.mode == "latency":
        rec["result"] = run_latency(builder, order, xs, a.batch_sizes, counts["warmup"], counts["iters"], a.select_seed, sync)
    else:
        rec["result"] = run_profile(builder, tome_mod, order, xs, a.batch_sizes, counts["warmup"], counts["profiled_forwards"],
                                    a.select_seed, sync)
    rec["env"] = U.env_record(dev)
    rec["finished_utc"] = U.utc()
    out_dir.mkdir(parents=True, exist_ok=True)
    U.write_atomic_json(out, rec)
    summ = {"out": str(out), "smoke": a.smoke, "order": order}
    if a.mode == "latency":
        summ["mean_ms"] = {k: round(v["mean_ms"], 4) for k, v in rec["result"]["cells"].items()}
    if a.smoke:
        summ["correctness_PASS"] = rec["correctness"]["PASS"]
    print(json.dumps(summ, indent=1))
    if a.smoke:
        print(U.SMOKE_BANNER)
        if not rec["correctness"]["PASS"]:
            raise SystemExit("SMOKE CORRECTNESS FAILED")


if __name__ == "__main__":
    main()
