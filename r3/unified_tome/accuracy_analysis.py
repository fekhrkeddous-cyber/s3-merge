#!/usr/bin/env python3
"""Accuracy analysis for paper Table 1 (Top-1 column). CPU only (numpy/scipy).
Every row's accuracy comes from full-validation runs of exactly that row's configuration (accuracy_eval.py):
  * S3-Merge-Acc, S3-Merge-Speed (PA on, random targets: stochastic): seeds 0..4 -> all seed values, mean, sample SD,
    95% Student-t interval of the mean (df = 4; between-seed variability, not generalization). Table 1: mean +/- SD.
  * official ToMe r=10, r=12 (deterministic algorithm): primary run + determinism replicate (both reported; the
    primary is the reported value; nothing is selected).
  * baseline (deterministic): one run (seed 0).
Outputs (never overwritten): <output-dir>/accuracy_table.csv, accuracy_analysis.json, accuracy_report.md
"""
import sys
sys.dont_write_bytecode = True
import argparse, csv, hashlib, io, json, math, statistics  # noqa: E401,E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import gmac  # noqa: E402
import unified_common as U  # noqa: E402

T4 = float(stats.t.ppf(0.975, 4))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(d, smoke):
    d = Path(d)
    comp = json.loads((d / "RUN_COMPLETE.json").read_text())
    for fn, h in comp["files_sha256"].items():
        assert sha(d / fn) == h, f"hash mismatch {d}/{fn}"
    s = json.loads((d / "run_summary.json").read_text())
    assert s["invariants_all_pass"] and s["labels_read"] and bool(s.get("smoke")) == smoke, d
    if not smoke:
        assert s["n_images"] == U.N_VAL, d
    return s, np.load(d / "predictions.npz"), comp


def build_parser():
    ap = argparse.ArgumentParser(description="Aggregate the Table 1 accuracy runs.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs-dir", required=True, help="parent directory of the run directories (campaign runs/)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--arms", nargs="+", choices=U.ARMS, default=list(U.ARMS))
    ap.add_argument("--smoke", action="store_true", help="aggregate smoke-test runs (missing runs are skipped; "
                    "SOFTWARE TEST ONLY)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    RUNS, OUT = Path(a.runs_dir), Path(a.output_dir)
    cfg = {"script": "accuracy_analysis.py", "runs_dir": str(RUNS), "output_dir": str(OUT), "arms": a.arms,
           "run_list": [r for r in U.ACCURACY_RUNS if r["arm"] in a.arms], "t975_df4": T4, "smoke": a.smoke}
    U.print_config(cfg)
    names = ("accuracy_table.csv", "accuracy_analysis.json", "accuracy_report.md")
    for n in names:
        if (OUT / n).exists():
            raise SystemExit(f"refusing to overwrite {OUT / n}")
    traj = gmac.trajectories()
    gm = {k: round(gmac.gmac(x, y), 4) for k, (x, y) in traj.items()}
    rows, res = [], {}
    for name in a.arms:
        ids = [r["run_id"] for r in U.ACCURACY_RUNS if r["arm"] == name]
        if a.smoke:
            ids = [r for r in ids if (RUNS / r / "RUN_COMPLETE.json").exists()]
            if not ids:
                continue
        recs = {r: load(RUNS / r, a.smoke) for r in ids}
        role = {r: U.ACCURACY_RUN_INDEX[r]["role"] for r in ids}
        res[name] = {"provenance": "full validation run(s) of this configuration" if not a.smoke else "SMOKE (synthetic images)",
                     "runs": {r: {"seed": U.ACCURACY_RUN_INDEX[r]["seed"], "role": role[r], "n_images": recs[r][0]["n_images"],
                                  "correct": recs[r][0]["correct_top1"], "correct_top5": recs[r][0]["correct_top5"],
                                  "logits_sha256": recs[r][0]["logits_sha256"],
                                  "per_batch_correct_sha256": recs[r][0]["per_batch_correct_sha256"],
                                  "run_summary_sha256": recs[r][2]["files_sha256"]["run_summary.json"]} for r in ids}}
        if name.startswith("tome"):
            prim = [r for r in ids if role[r] == "primary"]
            rep = [r for r in ids if role[r] == "determinism_replicate"]
            use = prim[0] if prim else ids[0]
            res[name]["primary"] = use
            res[name]["correct"] = [recs[use][0]["correct_top1"]]
            res[name]["correct_top5"] = [recs[use][0]["correct_top5"]]
            res[name]["n_images"] = [recs[use][0]["n_images"]]
            if prim and rep:
                p, q = prim[0], rep[0]
                res[name]["determinism"] = {"replicate": q, "correct_replicate": recs[q][0]["correct_top1"],
                                            "discordant_top1": int((recs[p][1]["argmax"] != recs[q][1]["argmax"]).sum()),
                                            "logits_bitwise_identical": recs[p][0]["logits_sha256"] == recs[q][0]["logits_sha256"]}
        else:
            order = sorted(ids, key=lambda r: U.ACCURACY_RUN_INDEX[r]["seed"])
            res[name]["correct"] = [recs[r][0]["correct_top1"] for r in order]
            res[name]["correct_top5"] = [recs[r][0]["correct_top5"] for r in order]
            res[name]["n_images"] = [recs[r][0]["n_images"] for r in order]
            res[name]["seeds"] = [U.ACCURACY_RUN_INDEX[r]["seed"] for r in order]
            res[name]["per_batch_sha256"] = [recs[r][0]["per_batch_correct_sha256"] for r in order]
    # ---- summaries + disclosure
    for name, r in res.items():
        v = [100.0 * c / n for c, n in zip(r["correct"], r["n_images"])]
        r["top1_values_pct"] = v
        r["top1_mean_pct"] = statistics.mean(v)
        r["top5_mean_pct"] = statistics.mean(100.0 * c / n for c, n in zip(r["correct_top5"], r["n_images"]))
        if len(v) >= 2 and name not in ("tome_r10", "tome_r12"):
            sd = statistics.stdev(v); h = T4 * sd / math.sqrt(len(v)) if len(v) == 5 else float("nan")  # noqa: E702
            r.update(top1_sd_pp=sd, top1_ci95=[r["top1_mean_pct"] - h, r["top1_mean_pct"] + h])
        d = U.DISCLOSURE[name]
        r["disclosure"] = d
        r["analytical_gmac"] = gm[d["gmac_key"]]
        r["trajectory_attn_tokens"] = traj[d["gmac_key"]][0]
        rows.append({"method": name, "label": d["label"], "family": d["family"], "PA": d["PA"], "schedule": d["schedule"], "scope": d["scope"],
                     "merge_location": d["merge_location"], "w": d["w"], "K_or_r": d["K_or_r"], "final_patch_tokens": d["final_patch"],
                     "final_total_tokens": d["final_total"], "merge_calls_per_forward": d["merge_calls"], "analytical_gmac": r["analytical_gmac"],
                     "n_runs": len(r["correct"]), "top1_values_pct": " ".join(f"{x:.3f}" for x in v), "top1_mean_pct": f"{r['top1_mean_pct']:.4f}",
                     "top1_sd_pp": f"{r.get('top1_sd_pp', float('nan')):.4f}", "top1_ci95": json.dumps([round(x, 4) for x in r.get("top1_ci95", [])]),
                     "top5_mean_pct": f"{r['top5_mean_pct']:.4f}", "provenance": r["provenance"]})
    if not rows:
        raise SystemExit(f"no completed runs found under {RUNS}")
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)  # noqa: E702
    U.write_atomic_bytes(OUT / "accuracy_table.csv", buf.getvalue().encode())
    import scipy
    U.write_atomic_json(OUT / "accuracy_analysis.json", {"t975_df4": T4, "methods": res, "config": cfg, "smoke": a.smoke,
                                                          "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                                                                  "scipy": scipy.__version__}})
    L = (["SMOKE TEST OUTPUT -- synthetic images, not paper measurements\n"] if a.smoke else []) + [
        "# Full-validation accuracy of the Table 1 arms (generated)\n",
        "Every value comes from a full 50,000-image ImageNet-1K validation of exactly that configuration (DeiT-S/16, same "
        "checkpoint, identical preprocessing, FP32).\n",
        "| Method | PA | schedule / scope | merge location, w, K/r | final patch / total tokens | GMAC (analytical) | Top-1 % (runs) | mean | SD | 95% t-CI | Top-5 mean | provenance |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['PA']} | {r['schedule']} / {r['scope']} | {r['merge_location']}, w={r['w']}, {r['K_or_r']} | {r['final_patch_tokens']} / {r['final_total_tokens']} | "
                 f"{r['analytical_gmac']} | {r['top1_values_pct']} | {r['top1_mean_pct']} | {r['top1_sd_pp']} | {r['top1_ci95']} | {r['top5_mean_pct']} | {r['provenance']} |")
    for name in ("tome_r10", "tome_r12"):
        if name in res and "determinism" in res[name]:
            dd = res[name]["determinism"]
            L.append(f"\n{name}: primary {res[name]['correct'][0]} correct; determinism replicate {dd['correct_replicate']} "
                     f"(discordant predictions {dd['discordant_top1']}; logits bitwise identical: {dd['logits_bitwise_identical']}).")
    L.append("\nCaveats: five seeds give 4 df (between-seed variability only). The comparison is not a causal isolation of "
             "spatial scope or PA; methods differ in PA, schedule, scope, final token budget and compute (see columns).")
    U.write_atomic_bytes(OUT / "accuracy_report.md", ("\n".join(L) + "\n").encode())
    print(json.dumps({k: [v["top1_mean_pct"], v.get("top1_sd_pp")] for k, v in res.items()}, indent=1))
    if a.smoke:
        print(U.SMOKE_BANNER)


if __name__ == "__main__":
    main()
