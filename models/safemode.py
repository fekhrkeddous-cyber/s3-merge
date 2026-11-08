"""Safe-Mode automatic one-time calibration — conservative empirical routing policy.

Activation rule:
    activate the merged path at batch B  <=>  LCB_95(speedup_B) > 1 + epsilon,
    epsilon = 0.02;  B* = smallest qualifying batch; if none
    qualifies, baseline-only. This is a conservative empirical routing policy,
    NOT a guarantee.

Deployed calibration protocol (short): B in {1,2,4,8,16,32}; per B, interleaved
baseline/merged blocks of 10 warmup + 50 timed iterations (per-iter sync,
perf_counter); speedup LCB via percentile bootstrap (2000 resamples of the two
50-iteration arrays, ratio of means). Result cached as JSON keyed by
SHA1(model | merge-config | device | driver-major | precision | resolution | backend).
Invalidation: any key component change, protocol version bump, or --recalibrate.

Scientific validation mode (--validate): runs ONE cold calibration in this process
and reports duration + per-B decisions + B*; five independent cold runs are obtained
by invoking this script five times in fresh processes.

Model under calibration: DeiT-S/16 @224, S3-Merge late_start=8 (paper L_s=9; target 84 tokens, w=7,
Eq. 1 uniform-average path). Defaults: FP32 (SM_FP32=1) and no PA (SM_PA=0).
Run from the repository root:  python -m models.safemode --validate
This module calibrates B* and caches the per-batch decision; it does not dispatch batches.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

import os as _os
PROP_ATTN = _os.environ.get("SM_PA", "0") == "1"
FP32 = _os.environ.get("SM_FP32", "1") == "1"
PROTO_VERSION = "smc-1"
EPSILON = 0.02          # pre-registered
BATCHES = (1, 2, 4, 8, 16, 32)
WARM, ITERS = 10, 50
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "safemode_cache.json")


def build_models(device="cuda"):
    import timm
    from models.sr_tome import apply_patch
    base = timm.create_model("deit_small_patch16_224", pretrained=True).to(device).eval()
    merged = timm.create_model("deit_small_patch16_224", pretrained=True).to(device).eval()
    apply_patch(merged, target_tokens=84, late_start=8, window_size=7,
                weighted_merge=False, tau=0.3, prop_attn=PROP_ATTN)
    return base, merged


def cache_key(device):
    parts = ["deit_small_patch16_224", "s3merge_l8_t84_w7_nopa_eq2",
             torch.cuda.get_device_name(0), torch.version.cuda.split(".")[0],
             ("fp32" if FP32 else "fp16_autocast") + ("_pa" if PROP_ATTN else "_nopa"), "224", "sdpa_default", PROTO_VERSION]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def timed_block(model, B, device="cuda"):
    x = torch.randn(B, 3, 224, 224, device=device)
    out = []
    import contextlib
    amp = contextlib.nullcontext() if FP32 else torch.autocast("cuda", dtype=torch.float16)
    with torch.no_grad(), amp:
        for _ in range(WARM):
            model(x); torch.cuda.synchronize()
        for _ in range(ITERS):
            t0 = time.perf_counter(); model(x); torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3)
    return out


def lcb95_speedup(base_ms, merged_ms, n_boot=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = torch.tensor(base_ms); m = torch.tensor(merged_ms)
    ratios = []
    for _ in range(n_boot):
        bi = b[torch.randint(len(b), (len(b),), generator=g)]
        mi = m[torch.randint(len(m), (len(m),), generator=g)]
        ratios.append(float(bi.mean() / mi.mean()))
    r = torch.tensor(sorted(ratios))
    return float(r[int(0.025 * n_boot)]), float(r[int(0.975 * n_boot)]), \
        float(b.mean() / m.mean())


def calibrate(device="cuda"):
    t_start = time.perf_counter()
    base, merged = build_models(device)
    per_b = {}
    for B in BATCHES:
        # interleaved arm order to cancel drift: base, merged, base, merged (2 half-blocks)
        b1 = timed_block(base, B); m1 = timed_block(merged, B)
        b2 = timed_block(base, B); m2 = timed_block(merged, B)
        base_ms, merged_ms = b1 + b2, m1 + m2
        lcb, ucb, point = lcb95_speedup(base_ms, merged_ms)
        per_b[str(B)] = {"speedup_point": point, "lcb95": lcb, "ucb95": ucb,
                         "activate": lcb > 1.0 + EPSILON,
                         "base_mean_ms": float(torch.tensor(base_ms).mean()),
                         "merged_mean_ms": float(torch.tensor(merged_ms).mean())}
    qualifying = [int(B) for B, v in per_b.items() if v["activate"]]
    bstar = min(qualifying) if qualifying else None   # None => baseline-only
    duration_s = time.perf_counter() - t_start
    return {"protocol": PROTO_VERSION, "epsilon": EPSILON, "batches": list(BATCHES),
            "per_batch": per_b, "b_star": bstar, "calibration_duration_s": duration_s,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    key = cache_key("cuda")
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))
    if args.recalibrate or key not in cache or args.validate:
        t0 = time.perf_counter()
        result = calibrate()
        cache[key] = result
        with open(CACHE, "w") as f:
            json.dump(cache, f, indent=2)
        # cache lookup overhead (measured):
        t1 = time.perf_counter()
        _ = json.load(open(CACHE))[key]
        lookup_ms = (time.perf_counter() - t1) * 1e3
        result["cache_lookup_ms"] = lookup_ms
        result["amortized_us_per_input"] = {
            str(n): result["calibration_duration_s"] * 1e6 / n for n in (1000, 10000, 100000)}
        out = os.path.join(os.path.dirname(CACHE), f"safemode_cal_{args.tag}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps({"b_star": result["b_star"],
                          "duration_s": round(result["calibration_duration_s"], 2),
                          "decisions": {b: v["activate"] for b, v in result["per_batch"].items()},
                          "lcb95": {b: round(v["lcb95"], 4) for b, v in result["per_batch"].items()}}))
    else:
        print(json.dumps({"cached_b_star": cache[key]["b_star"], "cache_hit": True}))


if __name__ == "__main__":
    main()
