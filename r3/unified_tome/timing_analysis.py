#!/usr/bin/env python3
"""Timing analysis for paper Table 1 / Fig. 5 (Supplementary Tables S21-S22). CPU only (numpy/scipy).

Unit = one process invocation (paper: n = 10). Per invocation and (arm, B): mean of the timed forwards.
  * Absolute latency: mean of the invocation means with 95% t-CI (df = n - 1); throughput B/L as the geometric mean
    over invocations (Table 1 B=32 img/s).
  * Speed-up vs the CONTEMPORANEOUS unmerged baseline, paired per invocation: geometric mean exp(mean ln(L_base/L_m))
    with 95% t-CI and a percentile bootstrap CI over invocations (10,000 resamples, numpy default_rng(0)).
    The arithmetic mean of paired ratios and the ratio of means are also reported (different estimands).
  * Paired latency ratios between operating points (ratio < 1 = first arm faster), same CIs; reading rule
    (margin +/-1%): EQUIVALENT-WITHIN-MARGIN if the CI is inside [0.99, 1.01]; EFFECT if the CI excludes 1 and
    |ln ratio| >= ln 1.01; else INCONCLUSIVE.
  * Mechanics from the separate profiling invocations (never combined with latency), if present.
  * Optional join with accuracy_analysis.py output by identical arm key (--accuracy-analysis).
Inputs: <raw-dir>/latency_inv00.json .. latency_inv<n-1>.json (+ optional profile_inv*.json).
Outputs (never overwritten): <output-dir>/timing_analysis.json, timing_per_invocation.csv, unified_table.md
"""
import sys
sys.dont_write_bytecode = True
import argparse, csv, io, json, math  # noqa: E401,E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import gmac  # noqa: E402
import unified_common as U  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description="Aggregate the unified timing invocations (Table 1 latency/throughput).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="directory with latency_inv<NN>.json (and profile_inv<NN>.json)")
    ap.add_argument("--output-dir", required=True, help="directory for the analysis outputs")
    ap.add_argument("--invocations", type=int, default=U.N_LATENCY_INVOCATIONS, help="latency invocations 0..n-1 to read")
    ap.add_argument("--arms", nargs="+", choices=U.ARMS, default=list(U.ARMS), help="arms (must include baseline)")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=list(U.BATCH_SIZES))
    ap.add_argument("--bootstrap-resamples", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.01, help="equivalence margin of the paired-ratio reading rule")
    ap.add_argument("--accuracy-analysis", default=None, help="accuracy_analysis.json to join by arm key (optional)")
    ap.add_argument("--smoke", action="store_true", help="accept smoke-test invocation files (SOFTWARE TEST ONLY)")
    return ap


def tci(v):
    v = np.asarray(v, float); m, sd = v.mean(), v.std(ddof=1); t = stats.t.ppf(0.975, len(v) - 1); h = t * sd / math.sqrt(len(v))  # noqa: E702
    return float(m), float(sd), float(m - h), float(m + h)


def make_boot(n_resamples, seed):
    def boot(v):
        v = np.asarray(v, float); idx = np.random.default_rng(seed).integers(0, len(v), (n_resamples, len(v))); b = v[idx].mean(1)  # noqa: E702
        return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
    return boot


def make_reading(margin):
    LM = math.log(1.0 + margin)

    def reading(lo, hi, m):
        if lo >= -LM and hi <= LM:
            return "EQUIVALENT-WITHIN-MARGIN" + (" (CI excludes 1)" if lo > 0 or hi < 0 else "")
        if (lo > 0 or hi < 0) and abs(m) >= LM:
            return "EFFECT"
        return "INCONCLUSIVE"
    return reading


def main(argv=None):
    a = build_parser().parse_args(argv)
    if "baseline" not in a.arms:
        raise SystemExit("--arms must include baseline (speed-ups are paired against it)")
    RAW, OUT = Path(a.raw_dir), Path(a.output_dir)
    cfg = {"script": "timing_analysis.py", "raw_dir": str(RAW), "output_dir": str(OUT), "invocations": a.invocations,
           "arms": a.arms, "batch_sizes": a.batch_sizes, "bootstrap_resamples": a.bootstrap_resamples,
           "bootstrap_seed": a.bootstrap_seed, "margin": a.margin, "paired_ratio_pairs": [list(p) for p in U.PAIRED_RATIO_PAIRS
                                                                                          if p[0] in a.arms and p[1] in a.arms],
           "accuracy_analysis": a.accuracy_analysis, "smoke": a.smoke}
    U.print_config(cfg)
    names = ("timing_analysis.json", "timing_per_invocation.csv", "unified_table.md")
    for n in names:
        if (OUT / n).exists():
            raise SystemExit(f"refusing to overwrite {OUT / n}")
    M, BS, N = a.arms, a.batch_sizes, a.invocations
    boot, reading = make_boot(a.bootstrap_resamples, a.bootstrap_seed), make_reading(a.margin)
    inv = [json.loads((RAW / f"latency_inv{i:02d}.json").read_text()) for i in range(N)]
    for i, r in enumerate(inv):
        if r["mode"] != "latency" or r["invocation"] != i:
            raise SystemExit(f"latency_inv{i:02d}.json is not latency invocation {i}")
        if bool(r.get("smoke")) != a.smoke:
            raise SystemExit(f"latency_inv{i:02d}.json has smoke={r.get('smoke')}; pass --smoke only for smoke-test files")
    L = {(m, b): np.array([r["result"]["cells"][f"{m}|{b}"]["mean_ms"] for r in inv]) for m in M for b in BS}
    lat, sp, pr = {}, {}, {}
    for m in M:
        for b in BS:
            mm, sd, lo, hi = tci(L[m, b])
            lat[f"{m}|{b}"] = {"mean_ms": mm, "sd_ms": sd, "ci95_ms": [lo, hi], "per_invocation_ms": L[m, b].tolist(),
                               "throughput_img_s_geomean": float(np.exp(np.mean(np.log(b / L[m, b] * 1e3))))}
            if m != "baseline":
                ln = np.log(L["baseline", b] / L[m, b]); g, _, glo, ghi = tci(ln); blo, bhi = boot(ln)  # noqa: E702
                sp[f"{m}|{b}"] = {"speedup_geomean": math.exp(g), "ci95_t": [math.exp(glo), math.exp(ghi)], "ci95_boot": [math.exp(blo), math.exp(bhi)],
                                  "arith_mean_paired_ratio": float(np.mean(L["baseline", b] / L[m, b])), "ratio_of_means": float(L["baseline", b].mean() / L[m, b].mean()),
                                  "net_latency_change_ms": tci(L[m, b] - L["baseline", b])[0]}
    for x, c in cfg["paired_ratio_pairs"]:
        for b in BS:
            ln = np.log(L[x, b] / L[c, b]); g, _, glo, ghi = tci(ln); blo, bhi = boot(ln)  # noqa: E702
            pr[f"{x}/{c}|{b}"] = {"ratio": math.exp(g), "ci95_t": [math.exp(glo), math.exp(ghi)], "ci95_boot": [math.exp(blo), math.exp(bhi)],
                                  "reading": reading(glo, ghi, g)}
    prof = {}
    pf = sorted(RAW.glob("profile_inv*.json"))
    if pf:
        P = [json.loads(p.read_text()) for p in pf]
        for m in M:
            for b in BS:
                cs = [p["result"]["cells"][f"{m}|{b}"] for p in P if f"{m}|{b}" in p["result"]["cells"]]
                if not cs:
                    continue
                prof[f"{m}|{b}"] = {k: float(np.mean([c[k] for c in cs])) for k in ("kernel_launches_per_forward", "merge_events_per_forward", "merge_ranges_per_forward",
                                    "merge_kernel_launches_per_forward", "merge_kernel_time_us_per_forward", "kernel_time_us_per_forward",
                                    "merge_range_cpu_wall_us_per_forward_profiled")}
                prof[f"{m}|{b}"]["attention_kernel_names"] = sorted({n for c in cs for n in c["attention_kernel_names"]})
    acc = json.loads(Path(a.accuracy_analysis).read_text())["methods"] if a.accuracy_analysis else {}
    drift = {str(b): float(L["baseline", b].std(ddof=1) / L["baseline", b].mean()) for b in BS}
    import scipy
    res = {"latency": lat, "speedup_vs_baseline": sp, "paired_ratios": pr, "profile": prof, "baseline_drift_cv": drift,
           "invocation_orders": [r["result"]["order"] for r in inv], "config": cfg, "smoke": a.smoke,
           "label": U.SMOKE_BANNER if a.smoke else "unified comparison (contemporaneous invocations)",
           "env": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
           "invocation_env": [r.get("env", {}) for r in inv]}
    OUT.mkdir(parents=True, exist_ok=True)
    U.write_atomic_json(OUT / "timing_analysis.json", res)
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n"); w.writerow(["invocation", "method", "B", "mean_ms", "order"])  # noqa: E702
    for i, r in enumerate(inv):
        for m in M:
            for b in BS:
                w.writerow([i, m, b, repr(r["result"]["cells"][f"{m}|{b}"]["mean_ms"]), "-".join(r["result"]["order"])])
    U.write_atomic_bytes(OUT / "timing_per_invocation.csv", buf.getvalue().encode())
    gm = {k: round(gmac.gmac(x, y), 4) for k, (x, y) in gmac.trajectories().items()}
    head = "SMOKE TEST OUTPUT -- not paper measurements\n\n" if a.smoke else ""
    T = [head + "# Unified comparison: accuracy and throughput at identical operating points (generated)\n",
         "| Method | PA | final patch/total | GMAC | Top-1 % (mean) | " + " | ".join(f"B={b} ms [CI]" for b in BS)
         + " | speed-up vs baseline " + " / ".join(f"B={b}" for b in BS) + f" [CI] | img/s B={BS[-1]} |",
         "|---|---|---|---|---|" + "---|" * len(BS) + "---|---|"]
    for m in M:
        dsc = U.DISCLOSURE[m]
        top1 = f"{acc[m]['top1_mean_pct']:.3f}" if m in acc else "n/a"
        cells = " | ".join(f"{lat[f'{m}|{b}']['mean_ms']:.3f} [{lat[f'{m}|{b}']['ci95_ms'][0]:.3f}, {lat[f'{m}|{b}']['ci95_ms'][1]:.3f}]" for b in BS)
        spd = "—" if m == "baseline" else " / ".join(f"{sp[f'{m}|{b}']['speedup_geomean']:.3f} [{sp[f'{m}|{b}']['ci95_t'][0]:.3f}, {sp[f'{m}|{b}']['ci95_t'][1]:.3f}]" for b in BS)
        T.append(f"| {dsc['label']} | {dsc['PA']} | {dsc['final_patch']}/{dsc['final_total']} | {gm[dsc['gmac_key']]} | {top1} | {cells} | {spd} | "
                 f"{lat[f'{m}|{BS[-1]}']['throughput_img_s_geomean']:.0f} |")
    T += [f"\n## Paired latency ratios between operating points (< 1 = first faster; margin ±{100 * a.margin:g}%)\n",
          "| Pair | B | ratio | 95% t-CI | bootstrap CI | reading |", "|---|---|---|---|---|---|"]
    for k, v in pr.items():
        p, b = k.split("|")
        T.append(f"| {p} | {b} | {v['ratio']:.4f} | [{v['ci95_t'][0]:.4f}, {v['ci95_t'][1]:.4f}] | [{v['ci95_boot'][0]:.4f}, {v['ci95_boot'][1]:.4f}] | {v['reading']} |")
    T.append(f"\nBaseline drift CV: {drift}. Mechanics (profiling, separate processes): see timing_analysis.json['profile'].")
    U.write_atomic_bytes(OUT / "unified_table.md", ("\n".join(T) + "\n").encode())
    print(json.dumps({"written": [str(OUT / n) for n in names], "smoke": a.smoke}, indent=1))
    if a.smoke:
        print(U.SMOKE_BANNER)


if __name__ == "__main__":
    main()
