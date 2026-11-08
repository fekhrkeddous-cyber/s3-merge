"""Factorial-study timing analysis (CPU only): paper Table 4 latency columns and contrast rows, Supplementary Table S19.

Unit = one process invocation (n = 10 latency invocations). Per invocation and (arm, B) the value is the MEAN of the
200 timed iterations (iterations are pseudoreplicates; medians are descriptive only).
  * Factorial contrasts on ln(latency) WITHIN each invocation, then summarized over invocations: mean, SD,
    95% Student-t (df = n - 1 = 9), reported as ratios exp(.). Sign: NEGATIVE ln-contrast = first-named level is
    FASTER. Robustness: percentile bootstrap over invocations (10,000 resamples, numpy default_rng(0)).
  * Reading rules (margin +/-1%, fixed before the runs): EQUIVALENT-WITHIN-MARGIN if the ratio CI lies inside
    [0.99, 1.01]; EFFECT if the CI excludes 1 and |ln ratio| >= ln(1.01); INCONCLUSIVE otherwise.
  * Speedup vs baseline: S_i = L_base,i / L_arm,i; geometric mean exp(mean ln S_i) with t-CI. Also shown, labelled:
    arithmetic mean of paired ratios, and the ratio of mean latencies (different estimands; never mixed).
  * Throughput B / L (images/s): geometric mean over invocations. Net latency change L_arm - L_base (ms): mean, t-CI.
  * Isolated merge cost (merge_micro invocations) and profiler counts (profile invocations) are optional, reported
    in their own tables and never combined with wall-clock latency.
Inputs: <raw-dir>/latency_inv00.json ... latency_inv<n-1>.json (+ optional merge_micro_inv*.json, profile_inv*.json)
written by timing.py. Outputs in --output-dir (refuse to overwrite): timing_latency_per_invocation.csv,
timing_factorial_contrasts.csv, timing_speedup_throughput.csv, timing_analysis.json, timing_report.md
"""
import sys
sys.dont_write_bytecode = True
import argparse, csv, io, json, math  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

GMAC = {"baseline": 4.5989, "single_shot": 3.9316, "progressive": 3.5919}   # analytical (paper Table 4)
TRAJ = {"baseline": [197] * 12, "SW": [197] * 9 + [85] * 3, "SG": [197] * 9 + [85] * 3,
        "PW": [197] * 6 + [169, 141, 113] + [85] * 3, "PG": [197] * 6 + [169, 141, 113] + [85] * 3}
MERGE_CALLS = {"baseline": 0, "SW": 1, "SG": 1, "PW": 4, "PG": 4}
SCOPE_LABEL = "PA-off S3-Merge Core factorial arms; NOT the PA-on S3-Merge accuracy-oriented operating points"
OUT_NAMES = ("timing_latency_per_invocation.csv", "timing_factorial_contrasts.csv", "timing_speedup_throughput.csv",
             "timing_analysis.json", "timing_report.md")


def tci(v):
    v = np.asarray(v, dtype=np.float64)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    t = float(stats.t.ppf(0.975, len(v) - 1))
    return m, sd, m - t * sd / math.sqrt(len(v)), m + t * sd / math.sqrt(len(v))


def boot(v, n=10000):
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    bm = v[idx].mean(1)
    return float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Factorial-study timing analysis (CPU only).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw-dir", default="runs/r3_factorial/timing", help="directory with the timing.py outputs")
    ap.add_argument("--output-dir", default="runs/r3_factorial/analysis", help="where the analysis files are written")
    ap.add_argument("--n-invocations", type=int, default=10, help="latency invocations to read (latency_inv00..)")
    ap.add_argument("--margin-pct", type=float, default=1.0, help="latency equivalence margin in percent")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples (numpy default_rng(0))")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    RAW, OUT, NL = Path(a.raw_dir), Path(a.output_dir), a.n_invocations
    LM = math.log(1 + a.margin_pct / 100.0)
    T9 = float(stats.t.ppf(0.975, NL - 1))
    cfg = {"script": "r3/factorial/analyze_timing.py", "raw_dir": str(RAW.resolve()), "output_dir": str(OUT.resolve()),
           "n_invocations": NL, "margin_pct": a.margin_pct, "ln_margin": LM, "t975": T9, "df": NL - 1,
           "bootstrap_resamples": a.bootstrap, "bootstrap_rng": "numpy default_rng(0)"}
    print("RESOLVED CONFIGURATION:\n" + json.dumps(cfg, indent=1), flush=True)

    def reading(lo, hi, m):
        if lo >= -LM and hi <= LM:
            return "EQUIVALENT-WITHIN-MARGIN" + (" (CI excludes 1)" if lo > 0 or hi < 0 else "")
        if (lo > 0 or hi < 0) and abs(m) >= LM:
            return "EFFECT"
        return "INCONCLUSIVE"

    for n in OUT_NAMES:
        if (OUT / n).exists():
            raise SystemExit(f"refusing to overwrite {OUT / n}")
    inv = [json.loads((RAW / f"latency_inv{i:02d}.json").read_text()) for i in range(NL)]
    assert all(r["mode"] == "latency" for r in inv)
    smoke = {bool(r.get("smoke", False)) for r in inv}
    if len(smoke) != 1:
        raise SystemExit("smoke and non-smoke invocations cannot be analysed together")
    SMOKE = smoke.pop()
    arms_seen = {tuple(sorted(r["result"]["arm_order"])) for r in inv}
    bs_seen = {tuple(r["config"]["batch_sizes"]) for r in inv}
    if len(arms_seen) != 1 or len(bs_seen) != 1:
        raise SystemExit("all latency invocations must time the same arms and batch sizes")
    arms_found = set(arms_seen.pop())
    ARMS = [x for x in ["baseline", "SW", "SG", "PW", "PG"] if x in arms_found]
    BS = list(bs_seen.pop())
    if ARMS != ["baseline", "SW", "SG", "PW", "PG"]:
        raise SystemExit(f"the analysis needs all five arms (found {ARMS})")
    L = {(x, b): np.array([r["result"]["cells"][f"{x}|{b}"]["mean_ms"] for r in inv]) for x in ARMS for b in BS}
    Lmed = {(x, b): np.array([r["result"]["cells"][f"{x}|{b}"]["median_ms"] for r in inv]) for x in ARMS for b in BS}
    rows = [{"invocation": i, "arm": x, "B": b, "mean_ms": repr(float(L[x, b][i])), "median_ms": repr(float(Lmed[x, b][i])),
             "arm_order": "-".join(inv[i]["result"]["arm_order"]), "smoke": SMOKE} for i in range(NL) for x in ARMS for b in BS]
    lat = {}
    for x in ARMS:
        for b in BS:
            m, sd, lo, hi = tci(L[x, b])
            lat[f"{x}|{b}"] = {"mean_ms": m, "sd_ms": sd, "ci95_ms": [lo, hi], "per_invocation_ms": L[x, b].tolist(),
                               "throughput_img_s_geomean": float(np.exp(np.mean(np.log(b / L[x, b] * 1e3))))}
    defs = {"E_schedule (single-shot vs progressive)": lambda d: (d["SG"] + d["SW"] - d["PG"] - d["PW"]) / 2,
            "E_scope (windowed vs global)": lambda d: (d["PW"] + d["SW"] - d["PG"] - d["SG"]) / 2,
            "I (SW - SG - PW + PG)": lambda d: d["SW"] - d["SG"] - d["PW"] + d["PG"],
            "SW vs SG": lambda d: d["SW"] - d["SG"], "PW vs PG": lambda d: d["PW"] - d["PG"],
            "SW vs PW": lambda d: d["SW"] - d["PW"], "SG vs PG": lambda d: d["SG"] - d["PG"]}
    con, crows = {}, []
    for b in BS:
        lnd = {x: np.log(L[x, b]) for x in ARMS}
        for name, fn in defs.items():
            v = fn(lnd)
            m, sd, lo, hi = tci(v)
            blo, bhi = boot(v, a.bootstrap)
            con[f"{name}|B={b}"] = {"B": b, "mean_ln": m, "sd_ln": sd, "ratio": math.exp(m), "ratio_ci95_t": [math.exp(lo), math.exp(hi)],
                                    "ratio_ci95_bootstrap": [math.exp(blo), math.exp(bhi)], "reading": reading(lo, hi, m),
                                    "per_invocation_ln": v.tolist(), "primary": name[0] in "EI" and b == 1}
            crows.append({"contrast": name, "B": b, "primary": con[f"{name}|B={b}"]["primary"], "ratio": repr(math.exp(m)),
                          "ratio_ci_lo": repr(math.exp(lo)), "ratio_ci_hi": repr(math.exp(hi)), "boot_lo": repr(math.exp(blo)),
                          "boot_hi": repr(math.exp(bhi)), "reading": con[f"{name}|B={b}"]["reading"]})
    sp, srows = {}, []
    for x in ARMS[1:]:
        for b in BS:
            lnS = np.log(L["baseline", b] / L[x, b])
            m, sd, lo, hi = tci(lnS)
            dm, dsd, dlo, dhi = tci(L[x, b] - L["baseline", b])
            sp[f"{x}|{b}"] = {"speedup_geomean": math.exp(m), "speedup_ci95": [math.exp(lo), math.exp(hi)],
                              "speedup_arith_mean_of_paired_ratios": float(np.mean(L["baseline", b] / L[x, b])),
                              "speedup_ratio_of_means": float(L["baseline", b].mean() / L[x, b].mean()),
                              "net_latency_change_ms": dm, "net_latency_change_ci95_ms": [dlo, dhi]}
            srows.append({"arm": x, "B": b, **{k: (repr(v) if isinstance(v, float) else json.dumps(v)) for k, v in sp[f"{x}|{b}"].items()}})
    drift = {str(b): {"cv": float(L["baseline", b].std(ddof=1) / L["baseline", b].mean()), "min_ms": float(L["baseline", b].min()),
                      "max_ms": float(L["baseline", b].max())} for b in BS}
    micro = sorted(RAW.glob("merge_micro_inv*.json"))
    mm = {}
    if micro:
        mr = [json.loads(p.read_text()) for p in micro]
        if {bool(r.get("smoke", False)) for r in mr} != {SMOKE}:
            raise SystemExit("merge_micro invocations differ in smoke status from the latency invocations")
        for x in ARMS[1:]:
            for b in BS:
                v = [r["result"]["cells"][f"{x}|{b}"]["sum_of_stage_means_ms"] for r in mr if f"{x}|{b}" in r["result"]["cells"]]
                if not v:
                    continue
                m, sd, lo, hi = tci(v) if len(v) > 1 else (v[0], float("nan"), float("nan"), float("nan"))
                mm[f"{x}|{b}"] = {"isolated_merge_ms_mean": m, "ci95": [lo, hi], "n_invocations": len(v), "per_invocation": v}
    prof = sorted(RAW.glob("profile_inv*.json"))
    pf = {}
    if prof:
        pr = [json.loads(p.read_text()) for p in prof]
        if {bool(r.get("smoke", False)) for r in pr} != {SMOKE}:
            raise SystemExit("profile invocations differ in smoke status from the latency invocations")
        for x in ARMS:
            for b in BS:
                cells = [r["result"]["cells"][f"{x}|{b}"] for r in pr if f"{x}|{b}" in r["result"]["cells"]]
                if not cells:
                    continue
                pf[f"{x}|{b}"] = {k: float(np.mean([c[k] for c in cells])) for k in
                                  ("kernel_launches_per_forward", "merge_kernel_launches_per_forward", "merge_stage_calls_per_forward",
                                   "kernel_time_us_per_forward", "merge_kernel_time_us_per_forward")}
                pf[f"{x}|{b}"]["attention_kernels"] = sorted({n for c in cells for n in c["attention_related_kernel_names"]})
    tele = [{"invocation": i, "temps": [c["telemetry_before"].get("gpu", "") for c in r["result"]["cells"].values()][:1],
             "compute_apps_at_start": r.get("compute_apps_at_start_before_cuda_init")} for i, r in enumerate(inv)]
    label = ("SMOKE (software test; NOT paper measurements)" if SMOKE else "factorial timing analysis")
    res = {"label": label, "smoke": SMOKE, "config": cfg, "arms": ARMS, "batch_sizes": BS, "n_invocations": NL, "t975": T9,
           "latency": lat, "contrasts_ln_latency": con, "speedup_vs_baseline": sp, "baseline_drift": drift,
           "isolated_merge_cost": mm, "profiling": pf, "telemetry_digest": tele, "gmac_analytical": GMAC,
           "token_trajectories": TRAJ, "merge_calls_per_forward": MERGE_CALLS, "scope": SCOPE_LABEL,
           "env_first_invocation": inv[0].get("env")}
    OUT.mkdir(parents=True, exist_ok=True)

    def wcsv(name, rr):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rr[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rr)
        (OUT / name).write_text(buf.getvalue())
    wcsv("timing_latency_per_invocation.csv", rows)
    wcsv("timing_factorial_contrasts.csv", crows)
    wcsv("timing_speedup_throughput.csv", srows)
    (OUT / "timing_analysis.json").write_text(json.dumps(res, indent=1))
    w0 = inv[0]["config"]
    R = ["# Factorial study: timing report (generated by r3/factorial/analyze_timing.py)\n"]
    if SMOKE:
        R.append("**SMOKE INVOCATIONS: software test with reduced counts. These timings are NOT paper measurements.**\n")
    R += [f"Scope: {SCOPE_LABEL}. FP32, PA off, batch sizes {BS}, {NL} independent process invocations "
          f"(unit), {w0['warmup']} warm-up + {w0['timed']} synchronized timed forwards per (arm, B). "
          f"Device: {(inv[0].get('env') or {}).get('gpu') or (inv[0].get('env') or {}).get('device')}.\n",
          "## Main five-arm table (mean of per-invocation means, ms; 95% t-CI over invocations; throughput = geometric mean)\n",
          "| Arm | " + " | ".join(f"B={b} latency ms [CI] / img/s" for b in BS) + " |", "|---|" + "---|" * len(BS)]
    for x in ARMS:
        R.append(f"| {x} | " + " | ".join(f"{lat[f'{x}|{b}']['mean_ms']:.4f} [{lat[f'{x}|{b}']['ci95_ms'][0]:.4f}, "
                                         f"{lat[f'{x}|{b}']['ci95_ms'][1]:.4f}] / {lat[f'{x}|{b}']['throughput_img_s_geomean']:.1f}" for b in BS) + " |")
    R += [f"\n## Factorial effects (ratio of latencies; < 1 means the first-named level is faster)\n",
          f"| Contrast | B | ratio | 95% t-CI | bootstrap CI | reading (±{a.margin_pct:g}%) |", "|---|---|---|---|---|---|"]
    for k, v in con.items():
        R.append(f"| {k.split('|')[0]}{' **(primary)**' if v['primary'] else ''} | {v['B']} | {v['ratio']:.4f} | "
                 f"[{v['ratio_ci95_t'][0]:.4f}, {v['ratio_ci95_t'][1]:.4f}] | [{v['ratio_ci95_bootstrap'][0]:.4f}, {v['ratio_ci95_bootstrap'][1]:.4f}] | {v['reading']} |")
    R += ["\n## Speedup vs unmerged baseline (paired per invocation)\n",
          "| Arm | B | geometric-mean speedup [95% CI] | arithmetic mean of paired ratios | ratio of means | net latency change ms [CI] |", "|---|---|---|---|---|---|"]
    for k, v in sp.items():
        x, b = k.split("|")
        R.append(f"| {x} | {b} | {v['speedup_geomean']:.4f} [{v['speedup_ci95'][0]:.4f}, {v['speedup_ci95'][1]:.4f}] | "
                 f"{v['speedup_arith_mean_of_paired_ratios']:.4f} | {v['speedup_ratio_of_means']:.4f} | "
                 f"{v['net_latency_change_ms']:+.4f} [{v['net_latency_change_ci95_ms'][0]:+.4f}, {v['net_latency_change_ci95_ms'][1]:+.4f}] |")
    R += ["\n## Mechanism / compute (separate measurements; never combined with wall-clock latency)\n",
          "| Arm | trajectory | merge calls | analytical GMAC | B | isolated merge ms (micro) | kernel launches/fwd | merge launches/fwd | merge kernel µs/fwd |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in ARMS:
        g = {"baseline": GMAC["baseline"], "SW": GMAC["single_shot"], "SG": GMAC["single_shot"],
             "PW": GMAC["progressive"], "PG": GMAC["progressive"]}[x]
        for b in BS:
            p = pf.get(f"{x}|{b}", {})
            mic = mm.get(f"{x}|{b}", {}).get("isolated_merge_ms_mean", 0.0 if x == "baseline" else float("nan"))
            R.append(f"| {x} | {TRAJ[x]} | {MERGE_CALLS[x]} | {g} | {b} | "
                     f"{mic:.4f} | {p.get('kernel_launches_per_forward', float('nan')):.1f} | {p.get('merge_kernel_launches_per_forward', float('nan')):.1f} | "
                     f"{p.get('merge_kernel_time_us_per_forward', float('nan')):.1f} |")
    R += [f"\nBaseline process drift (CV of per-invocation means): " + ", ".join(f"B={b}: {drift[str(b)]['cv']:.4%}" for b in BS) + ".",
          f"\nCaveats: {NL} invocations (df={NL - 1}); clocks not locked (recorded); latency is hardware-dependent and not "
          "bit-exactly reproducible; profiler counts are descriptive and come from separate processes; the isolated "
          "merge cost is not a decomposition of end-to-end latency; PA-off Core scope only."]
    (OUT / "timing_report.md").write_text("\n".join(R) + "\n")
    print(json.dumps({k: {"ratio": round(v["ratio"], 5), "ci": [round(x, 5) for x in v["ratio_ci95_t"]], "reading": v["reading"]}
                      for k, v in con.items() if v["primary"]}, indent=1))
    if SMOKE:
        print("SMOKE INVOCATIONS: software test only; these timings are NOT paper measurements.", flush=True)


if __name__ == "__main__":
    main()
