"""Factorial-study accuracy analysis (CPU only): paper Table 4 Top-1 column and contrast rows, Supplementary
Tables S17-S18.

Statistical unit: one complete 50,000-image validation run per (arm, seed). Contrasts are computed WITHIN each seed
from integer correct counts with exact rational arithmetic (fractions), then summarized across the seeds:
mean, sample SD (n-1), SE, 95% Student-t interval (df = n_seeds - 1 = 4). Reading rules (margin +/-0.10 points,
fixed before the runs):
  EQUIVALENT-WITHIN-MARGIN : 95% CI inside [-0.10, +0.10]  (two one-sided tests at alpha = 0.025 each)
  EFFECT                   : 95% CI excludes 0 AND |mean| >= 0.10
  INCONCLUSIVE             : otherwise
The deterministic baseline is reported once; baseline - seed values are listed per arm without any CI.
Image-level discordance counts are DESCRIPTIVE only (images are not replicates for seed variability).

Input: the run directories written by accuracy.py (any names) under --runs-dir.
Outputs in --output-dir (refuse to overwrite): accuracy_seed_level_results.csv, accuracy_paired_contrasts.csv,
accuracy_discordance.csv, accuracy_analysis.json, accuracy_report.md
"""
import sys
sys.dont_write_bytecode = True
import argparse, csv, hashlib, io, json, math, statistics  # noqa: E401,E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

FACT = ["SW", "SG", "PW", "PG"]
GMAC = {"baseline": 4.5989, "single_shot": 3.9316, "progressive": 3.5919}   # analytical (paper Table 4)
OUT_NAMES = ("accuracy_seed_level_results.csv", "accuracy_paired_contrasts.csv", "accuracy_discordance.csv",
             "accuracy_analysis.json", "accuracy_report.md")
# Published correct counts out of 50,000 (paper Table 4 / Supplementary Table S17), for comparison only.
PAPER_CORRECT = {"baseline": {0: 39854}, "core": {0: 39495},
                 "SW": {0: 39495, 1: 39462, 2: 39497, 3: 39509, 4: 39504},
                 "SG": {0: 39462, 1: 39449, 2: 39534, 3: 39480, 4: 39514},
                 "PW": {0: 39390, 1: 39370, 2: 39405, 3: 39412, 4: 39460},
                 "PG": {0: 39385, 1: 39382, 2: 39415, 3: 39405, 4: 39450}}
# sha256(json.dumps(per-batch correct-count list)) of the paper's baseline run (batch 64, order 0..49999): a
# fingerprint of data / order / preprocessing / batching identity for the deterministic arm.
PAPER_BASELINE_PER_BATCH_SHA256 = "23ed000d5a8ae4866a7fed93581c312c268300d0a0eebe13573e88fdfa438d28"
CANONICAL_TOLERANCE = {"max_discordant_top1": 10, "max_abs_delta_correct": 10}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(d):
    comp = json.loads((d / "RUN_COMPLETE.json").read_text())
    for fn, h in comp["files_sha256"].items():
        assert sha(d / fn) == h, f"hash mismatch {d}/{fn}"
    s = json.loads((d / "run_summary.json").read_text())
    p = dict(np.load(d / "predictions.npz"))
    t = np.load(d / "targets_u8.npy") if (d / "targets_u8.npy").exists() else None
    assert s["invariants_all_pass"], f"{d}: invariants failed"
    return s, p, comp, t


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Factorial-study accuracy analysis (CPU only).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs-dir", default="runs/r3_factorial/accuracy", help="directory containing the accuracy run directories")
    ap.add_argument("--output-dir", default="runs/r3_factorial/analysis", help="where the analysis files are written")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="seeds of the factorial arms")
    ap.add_argument("--margin-pp", type=float, default=0.10, help="equivalence margin in percentage points")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    RUNS, OUT, SEEDS = Path(a.runs_dir), Path(a.output_dir), list(a.seeds)
    MARGIN = Fraction(a.margin_pp).limit_denominator(1000)
    T975 = float(stats.t.ppf(0.975, len(SEEDS) - 1))
    if len(SEEDS) == 5:
        assert abs(T975 - 2.7764451051977987) < 1e-12
    cfg = {"script": "r3/factorial/analyze_accuracy.py", "runs_dir": str(RUNS.resolve()), "output_dir": str(OUT.resolve()),
           "seeds": SEEDS, "margin_pp": float(MARGIN), "t975": T975, "df": len(SEEDS) - 1}
    print("RESOLVED CONFIGURATION:\n" + json.dumps(cfg, indent=1), flush=True)
    for name in OUT_NAMES:
        if (OUT / name).exists():
            raise SystemExit(f"refusing to overwrite {OUT / name}")

    # ---- discover runs by (arm, seed)
    runs = {}
    for d in sorted(p for p in RUNS.iterdir() if (p / "RUN_COMPLETE.json").exists()):
        s, p, comp, t = load(d)
        key = (s["arm"], s["seed"])
        if key in runs:
            raise SystemExit(f"duplicate run for arm {key[0]} seed {key[1]}: {runs[key][4].name} and {d.name}")
        runs[key] = (s, p, comp, t, d)
    if not any(k[0] == "baseline" for k in runs):
        raise SystemExit("no baseline run found")
    base_key = sorted(k for k in runs if k[0] == "baseline")[0]
    missing = [(arm, s) for arm in FACT for s in SEEDS if (arm, s) not in runs]
    if missing:
        raise SystemExit(f"missing factorial runs: {missing}")
    used = [base_key] + ([("core", SEEDS[0])] if ("core", SEEDS[0]) in runs else []) + [(arm, s) for s in SEEDS for arm in FACT]
    smoke_flags = {bool(runs[k][0].get("smoke", False)) for k in used}
    if len(smoke_flags) != 1:
        raise SystemExit("smoke and non-smoke runs cannot be analysed together")
    SMOKE = smoke_flags.pop()
    N = runs[base_key][0]["n_images"]
    for k in used:
        s, p = runs[k][0], runs[k][1]
        assert s["n_images"] == N and (p["hf_index"] == np.arange(N)).all(), f"{k}: image set/order differs"
    if not SMOKE and N != 50000:
        raise SystemExit(f"full analysis requires 50,000 images per run (found {N})")

    cnt = {k: runs[k][0]["correct_top1"] for k in used}
    cnt5 = {k: runs[k][0]["correct_top5"] for k in used}
    pred = {k: runs[k][1] for k in used}

    def pp(count):                      # exact percentage points from a count of images
        return Fraction(100 * count, N)

    def summarize(vals):
        """vals: list of Fractions (pp). Returns float summary with t-interval and reading."""
        f = [float(v) for v in vals]
        mean = float(sum(vals) / len(vals))
        sd = statistics.stdev(f)
        se = sd / math.sqrt(len(f))
        lo, hi = mean - T975 * se, mean + T975 * se
        m = float(MARGIN)
        if lo >= -m and hi <= m:
            reading = "EQUIVALENT-WITHIN-MARGIN" + (" (CI excludes 0)" if (lo > 0 or hi < 0) else "")
        elif (lo > 0 or hi < 0) and abs(mean) >= m:
            reading = "EFFECT"
        else:
            reading = "INCONCLUSIVE"
        return {"mean_pp": mean, "sd_pp": sd, "se_pp": se, "t975": T975, "ci95_lo_pp": lo, "ci95_hi_pp": hi,
                "ci_excludes_0": bool(lo > 0 or hi < 0), "reading": reading, "n_seeds": len(f), "seed_values_pp": f}

    def role(arm):
        return {"baseline": "deterministic unmerged baseline (reported once)",
                "core": "canonical S3-Merge Core control (PA off; not a factorial arm)"}.get(
            arm, "factorial stochastic arm (statistical unit = this full run)")

    # ---- seed-level table
    rows = []
    for k in used:
        s, p, comp, t, d = runs[k]
        rows.append({"run_id": s["run_id"], "arm": s["arm"], "seed": s["seed"] if s["arm"] != "baseline" else "deterministic",
                     "role": role(s["arm"]), "n_images": s["n_images"], "correct_top1": s["correct_top1"],
                     "top1_pct": repr(float(pp(s["correct_top1"]))), "correct_top5": s["correct_top5"],
                     "top5_pct": repr(float(pp(s["correct_top5"]))), "logits_sha256": s["logits_sha256"],
                     "targets_sha256_raw": s.get("targets_sha256_raw", ""), "predictions_npz_sha256": comp["files_sha256"]["predictions.npz"],
                     "run_summary_sha256": comp["files_sha256"]["run_summary.json"], "started_utc": s["started_utc"],
                     "finished_utc": s["finished_utc"], "hostname": s["env"]["hostname"], "smoke": SMOKE})

    # ---- contrasts within seed
    defs = {
        "E_schedule (single-shot minus progressive)": lambda c, s: (c["SG", s] + c["SW", s] - c["PG", s] - c["PW", s]) / Fraction(2),
        "E_scope (windowed minus global)": lambda c, s: (c["PW", s] + c["SW", s] - c["PG", s] - c["SG", s]) / Fraction(2),
        "I (SW - SG - PW + PG)": lambda c, s: Fraction(c["SW", s] - c["SG", s] - c["PW", s] + c["PG", s]),
        "SW - SG (scope within single-shot)": lambda c, s: Fraction(c["SW", s] - c["SG", s]),
        "PW - PG (scope within progressive)": lambda c, s: Fraction(c["PW", s] - c["PG", s]),
        "SW - PW (schedule within windowed)": lambda c, s: Fraction(c["SW", s] - c["PW", s]),
        "SG - PG (schedule within global)": lambda c, s: Fraction(c["SG", s] - c["PG", s]),
    }
    primary = list(defs)[:3]
    contrasts = {}
    for name, fn in defs.items():
        per_seed_images = [fn(cnt, s) for s in SEEDS]                   # in images (may be half-integers)
        vals = [Fraction(100, N) * x for x in per_seed_images]
        contrasts[name] = summarize(vals) | {"per_seed_images": [float(x) for x in per_seed_images],
                                            "primary": name in primary}
    top5_contrasts = {}
    for name, fn in list(defs.items())[:3]:
        top5_contrasts[name] = summarize([Fraction(100, N) * fn(cnt5, s) for s in SEEDS])

    nS = len(SEEDS)
    arms = {}
    for arm in FACT:
        v = [pp(cnt[arm, s]) for s in SEEDS]
        f = [float(x) for x in v]
        arms[arm] = {"top1_mean_pct": float(sum(v) / nS), "top1_sd_pp": statistics.stdev(f), "top1_seed_values_pct": f,
                     "correct_seed_values": [cnt[arm, s] for s in SEEDS], "min_pct": min(f), "max_pct": max(f),
                     "top5_mean_pct": float(sum(pp(cnt5[arm, s]) for s in SEEDS) / nS),
                     "top5_sd_pp": statistics.stdev([float(pp(cnt5[arm, s])) for s in SEEDS])}
    base_c = cnt[base_key]
    baseline = {"correct_top1": base_c, "top1_pct": float(pp(base_c)), "correct_top5": cnt5[base_key],
                "top5_pct": float(pp(cnt5[base_key])),
                "baseline_minus_seed_pp": {arm: [float(pp(base_c) - pp(cnt[arm, s])) for s in SEEDS] for arm in FACT},
                "baseline_minus_mean_pp": {arm: float(pp(base_c) - sum(pp(cnt[arm, s]) for s in SEEDS) / nS) for arm in FACT},
                "note": "deterministic arm evaluated once; no CI is formed for baseline - stochastic arm differences"}

    # ---- descriptive discordance (within seed, arm pairs)
    disc_rows = []
    pairs = [("SW", "SG"), ("PW", "PG"), ("SW", "PW"), ("SG", "PG"), ("SW", "PG"), ("SG", "PW")]
    lab = pred["SW", SEEDS[0]]["label"]
    for s in SEEDS:
        for x, y in pairs:
            pa, pb = pred[x, s], pred[y, s]
            ca, cb = pa["correct_top1"], pb["correct_top1"]
            disc_rows.append({"seed": s, "arm_a": x, "arm_b": y, "n_pred_differ": int((pa["argmax"] != pb["argmax"]).sum()),
                              "both_correct": int((ca & cb).sum()), "a_only_correct": int((ca & ~cb).sum()),
                              "b_only_correct": int((~ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum())})
    seed_span = f"{min(SEEDS)}-{max(SEEDS)}"
    for arm in FACT:
        st = np.stack([pred[arm, s]["argmax"] for s in SEEDS])
        disc_rows.append({"seed": seed_span, "arm_a": arm, "arm_b": arm, "n_pred_differ": int((st != st[0]).any(0).sum()),
                          "both_correct": "", "a_only_correct": "", "b_only_correct": "",
                          "both_wrong": f"images whose Top-1 prediction changes across the {nS} seeds"})
    assert all((pred[k]["label"] == lab).all() for k in pred)

    # ---- canonical control (core vs SW, first seed) and comparison with the published counts
    canonical = None
    if ("core", SEEDS[0]) in runs:
        cc, sw = cnt["core", SEEDS[0]], cnt["SW", SEEDS[0]]
        disc = int((pred["core", SEEDS[0]]["argmax"] != pred["SW", SEEDS[0]]["argmax"]).sum())
        canonical = {"seed": SEEDS[0], "canonical_core_correct": cc, "SW_correct": sw, "discordant_top1": disc,
                     "tolerance": CANONICAL_TOLERANCE,
                     "within_tolerance": disc <= CANONICAL_TOLERANCE["max_discordant_top1"]
                     and abs(cc - sw) <= CANONICAL_TOLERANCE["max_abs_delta_correct"],
                     "purpose": "full-scale check that factorial SW reproduces the canonical S3-Merge Core (same seed, "
                                "same draws); the canonical CUDA aggregation is not bitwise reproducible, so the "
                                "tolerance is an image count"}
    paper = {"note": "published correct counts (paper Table 4 / Supplementary Table S17); descriptive comparison",
             "per_run": {f"{k[0]}|{k[1]}": {"observed": cnt[k], "paper": PAPER_CORRECT.get(k[0], {}).get(k[1]),
                                             "delta": (cnt[k] - PAPER_CORRECT[k[0]][k[1]])
                                             if k[1] in PAPER_CORRECT.get(k[0], {}) else None} for k in used},
             "baseline_per_batch_correct_sha256": {"observed": runs[base_key][0].get("per_batch_correct_sha256"),
                                                   "paper": PAPER_BASELINE_PER_BATCH_SHA256,
                                                   "identical": runs[base_key][0].get("per_batch_correct_sha256") == PAPER_BASELINE_PER_BATCH_SHA256}}

    mech = {}
    for arm in FACT:
        mech[arm] = {"cross_window_member_rate_per_seed": [runs[arm, s][0]["cross_window_member_rate"] for s in SEEDS],
                     "cluster_size_hist_first_seed": runs[arm, SEEDS[0]][0]["cluster_size_hist"]}
    tgt_same = {str(s): all(np.array_equal(runs[arm, s][3], runs["SW", s][3]) for arm in FACT) for s in SEEDS}
    tgt_diff = {f"{s}vs{t}": not np.array_equal(runs["SW", s][3], runs["SW", t][3]) for s, t in zip(SEEDS, SEEDS[1:])}
    mech["final_targets_identical_across_arms_by_seed"] = tgt_same
    mech["SW_final_targets_differ_between_consecutive_seeds"] = tgt_diff

    label = ("SMOKE (software test; NOT paper measurements)" if SMOKE else
             "factorial accuracy analysis (full 50,000-image validation runs)")
    res = {"label": label, "smoke": SMOKE, "config": cfg, "n_images": N, "seeds": SEEDS,
           "arms": arms, "baseline": baseline, "contrasts_top1": contrasts, "contrasts_top5_secondary": top5_contrasts,
           "canonical_control": canonical, "paper_reference": paper,
           "mechanism_descriptive": mech, "gmac_analytical": GMAC,
           "statistics": {"t975": T975, "df": nS - 1, "margin_pp": float(MARGIN), "reading_rules": {
               "EQUIVALENT-WITHIN-MARGIN": f"95% CI inside [-{float(MARGIN):.2f},+{float(MARGIN):.2f}] pp (TOST alpha=0.025 each side)",
               "EFFECT": f"95% CI excludes 0 and |mean| >= {float(MARGIN):.2f} pp", "INCONCLUSIVE": "otherwise"}},
           "run_dirs": {f"{k[0]}|{k[1]}": str(runs[k][4]) for k in used}}

    OUT.mkdir(parents=True, exist_ok=True)

    def write_csv(name, rows_):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows_[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows_)
        (OUT / name).write_text(buf.getvalue())

    write_csv("accuracy_seed_level_results.csv", rows)
    crow = []
    for name, c in contrasts.items():
        for i, s in enumerate(SEEDS):
            crow.append({"contrast": name, "primary": c["primary"], "seed": s, "value_pp": repr(c["seed_values_pp"][i]),
                         "value_images": c["per_seed_images"][i], "mean_pp": "", "sd_pp": "", "ci95_lo_pp": "", "ci95_hi_pp": "", "reading": ""})
        crow.append({"contrast": name, "primary": c["primary"], "seed": f"summary({nS} seeds, df={nS - 1})", "value_pp": "", "value_images": "",
                     "mean_pp": repr(c["mean_pp"]), "sd_pp": repr(c["sd_pp"]), "ci95_lo_pp": repr(c["ci95_lo_pp"]),
                     "ci95_hi_pp": repr(c["ci95_hi_pp"]), "reading": c["reading"]})
    write_csv("accuracy_paired_contrasts.csv", crow)
    write_csv("accuracy_discordance.csv", disc_rows)
    (OUT / "accuracy_analysis.json").write_text(json.dumps(res, indent=1))

    # ---- report (generated; numbers are never hand-typed)
    L = []
    L.append("# Factorial study: accuracy report (generated by r3/factorial/analyze_accuracy.py)\n")
    if SMOKE:
        L.append("**SMOKE RUNS: software test on synthetic inputs. These numbers are NOT paper measurements.**\n")
    L.append("**Setting:**")
    L.append("* DeiT-S/16 224², FP32, PA off, final T=84 (85 with CLS), matched final tokens.")
    L.append(f"* {N:,} images per run, fixed order, batch {runs[base_key][0]['batch_size']}.")
    L.append(f"* Seeds {', '.join(map(str, SEEDS))}. One `torch.manual_seed(s); np.random.seed(s)` per run.")
    L.append(f"* Analytical backbone GMAC: single-shot {GMAC['single_shot']}, progressive {GMAC['progressive']}, "
             f"unmerged {GMAC['baseline']}.")
    L.append("* This is a matched-final-token comparison, NOT iso-compute. The schedule contrast bundles the "
             "token-compute trajectory, intermediate representation changes and the number of merges.\n")
    L.append(f"## 1. Arm results (Top-1 %, {nS} seeds)\n")
    L.append("| Arm | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean | sample SD (pp) |")
    L.append("|---|" + "---|" * (nS + 2))
    for arm in FACT:
        v = arms[arm]["top1_seed_values_pct"]
        L.append(f"| {arm} | " + " | ".join(f"{x:.3f} ({cnt[arm, s]})" for x, s in zip(v, SEEDS)) +
                 f" | {arms[arm]['top1_mean_pct']:.4f} | {arms[arm]['top1_sd_pp']:.4f} |")
    L.append(f"\nUnmerged baseline (deterministic, evaluated once): **{baseline['top1_pct']:.3f}%** ({base_c}/{N:,}); "
             f"Top-5 {baseline['top5_pct']:.3f}%. Baseline − arm mean (pp): " +
             ", ".join(f"{arm} {baseline['baseline_minus_mean_pp'][arm]:.4f}" for arm in FACT) + " (no CI; see JSON for per-seed values).\n")
    L.append(f"## 2. Paired factorial contrasts (Top-1, percentage points; within seed, then across {nS} seeds, t df={nS - 1})\n")
    L.append(f"| Contrast | per-seed values (pp) | mean | SD | 95% CI | reading (margin ±{float(MARGIN):.2f} pp) |")
    L.append("|---|---|---|---|---|---|")
    for name, c in contrasts.items():
        tag = " **(primary)**" if c["primary"] else ""
        L.append(f"| {name}{tag} | " + ", ".join(f"{x:+.3f}" for x in c["seed_values_pp"]) +
                 f" | {c['mean_pp']:+.4f} | {c['sd_pp']:.4f} | [{c['ci95_lo_pp']:+.4f}, {c['ci95_hi_pp']:+.4f}] | {c['reading']} |")
    L.append("\nThree primary contrasts were pre-specified, with unadjusted 95% intervals. Top-5 contrasts are secondary "
             "and in `accuracy_analysis.json`.\n")
    L.append("## 3. Canonical control and published values\n")
    if canonical is not None:
        L.append(f"* **Canonical S³-Merge Core** (`models/sr_tome.py` `apply_patch`, PA off, T=84), seed {SEEDS[0]}: "
                 f"{canonical['canonical_core_correct']} correct vs SW {canonical['SW_correct']}; discordant Top-1 predictions "
                 f"{canonical['discordant_top1']} (tolerance ≤ {CANONICAL_TOLERANCE['max_discordant_top1']}; |Δ correct| ≤ "
                 f"{CANONICAL_TOLERANCE['max_abs_delta_correct']}): {'within' if canonical['within_tolerance'] else 'OUTSIDE'} tolerance.")
    else:
        L.append("* Canonical S³-Merge Core control: not run.")
    L.append(f"* Baseline per-batch correct-count sha256 vs the paper's baseline run: "
             f"{'identical' if paper['baseline_per_batch_correct_sha256']['identical'] else 'DIFFERENT'}.")
    L.append("* Correct counts minus published counts (Table S17): " +
             "; ".join(f"{k}: {v['delta']:+d}" for k, v in paper["per_run"].items() if v["delta"] is not None) + "\n")
    L.append("## 4. Descriptive image-level discordance (not seed replicates)\n")
    L.append("| seed | pair | predictions differ | both correct | A only | B only | both wrong |")
    L.append("|---|---|---|---|---|---|---|")
    for r in disc_rows:
        if r["arm_a"] != r["arm_b"] and r["arm_a"] + r["arm_b"] in ("SWSG", "PWPG", "SWPW", "SGPG"):
            L.append(f"| {r['seed']} | {r['arm_a']} vs {r['arm_b']} | {r['n_pred_differ']} | {r['both_correct']} | "
                     f"{r['a_only_correct']} | {r['b_only_correct']} | {r['both_wrong']} |")
    L.append(f"\nImages whose Top-1 prediction changes across the {nS} seeds: " +
             ", ".join(f"{r['arm_a']} {r['n_pred_differ']}" for r in disc_rows if r["seed"] == seed_span) + ".\n")
    L.append("## 5. Mechanism (descriptive)\n")
    L.append("Cross-window membership rate (fraction of the 196 patches whose final representative lies in another "
             "original 7×7 window; windowed arms must be 0): " +
             "; ".join(f"{arm}: " + ", ".join(f"{x:.4f}" for x in mech[arm]["cross_window_member_rate_per_seed"]) for arm in FACT) + ".")
    L.append(f"Final targets identical across SW/SG/PW/PG within each seed: {tgt_same}; SW targets differ between "
             f"consecutive seeds: {tgt_diff}.\n")
    L.append("## 6. Caveats\n")
    L.append(f"* {nS} seeds give {nS - 1} degrees of freedom. The intervals describe **between-seed (method-stochasticity) "
             "variability** on this fixed validation set and checkpoint. They do not describe generalization to other "
             "models or datasets, and they may be unstable.")
    L.append(f"* Failing to reject a difference is not evidence of equivalence. EQUIVALENT-WITHIN-MARGIN requires the whole "
             f"95% CI inside ±{float(MARGIN):.2f} pp, which equals two one-sided tests at α = 0.025.")
    L.append("* The margin was fixed before any result.")
    L.append("* The matched-final-token design is not iso-compute; the analytical GMAC difference is reported, not measured.")
    L.append("* PA-off Core scope: these numbers must not be conflated with the PA-on S³-Merge operating points.")
    (OUT / "accuracy_report.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: {"mean": round(v["mean_pp"], 5), "ci": [round(v["ci95_lo_pp"], 5), round(v["ci95_hi_pp"], 5)],
                          "reading": v["reading"]} for k, v in contrasts.items()}, indent=1))
    if SMOKE:
        print("SMOKE RUNS: software test only; these numbers are NOT paper measurements.", flush=True)


if __name__ == "__main__":
    main()
