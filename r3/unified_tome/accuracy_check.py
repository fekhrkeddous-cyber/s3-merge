#!/usr/bin/env python3
"""Post-run check of one accuracy run (CPU; numpy only). Exit 0 = PASS, 4 = FAIL.

Checks: RUN_COMPLETE.json present and its file hashes verify; n = 50,000 images in hf_index order 0..49999 (smoke: the
synthetic count, in order); labels read; arm/seed as in the paper run list; all per-batch invariants true. For an
official-ToMe determinism replicate, the predictions are compared with its primary run and the comparison is RECORDED
(it never selects or replaces a run). Writes <output-dir>/<run-id>.json (never overwritten).
"""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

N_VAL = 50000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(d):
    comp = json.loads((d / "RUN_COMPLETE.json").read_text())
    for fn, h in comp["files_sha256"].items():
        assert sha(d / fn) == h, f"hash mismatch {d}/{fn}"
    return json.loads((d / "run_summary.json").read_text()), np.load(d / "predictions.npz")


def build_parser():
    ap = argparse.ArgumentParser(description="Check one completed accuracy run.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs-dir", required=True, help="parent directory of the run directories")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-dir", required=True, help="directory for <run-id>.json")
    ap.add_argument("--smoke", action="store_true", help="check a smoke-test run (any image count; SOFTWARE TEST ONLY)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from unified_common import ACCURACY_RUN_INDEX, print_config, SMOKE_BANNER
    print_config({"script": "accuracy_check.py", "runs_dir": a.runs_dir, "run_id": a.run_id, "output_dir": a.output_dir,
                  "smoke": a.smoke})
    runs = Path(a.runs_dir)
    run = ACCURACY_RUN_INDEX.get(a.run_id)
    s, p = load(runs / a.run_id)
    n = s["n_images"]
    c = {"complete_and_hashes_verify": True,
         "n_images": (n == N_VAL) if not a.smoke else n > 0,
         "index_order": bool((p["hf_index"] == np.arange(n)).all()) and len(p["hf_index"]) == n,
         "labels_read": s["labels_read"] is True,
         "mode_matches": s["mode"] == ("smoke" if a.smoke else "full") and bool(s.get("smoke")) == a.smoke,
         "run_in_paper_list": run is not None,
         "method_seed_match": run is not None and s["method"] == run["arm"] and s["seed"] == run["seed"],
         "invariants_all_pass": bool(s["invariants_all_pass"])}
    out = {"run_id": a.run_id, "checks": c, "correct_top1": s["correct_top1"], "n_images": n, "smoke": a.smoke}
    if run is not None and run.get("role") == "determinism_replicate" and (runs / run["primary"] / "RUN_COMPLETE.json").exists():
        s0, p0 = load(runs / run["primary"])
        out["determinism_vs_primary"] = {"primary": run["primary"], "correct_primary": s0["correct_top1"],
                                         "correct_replicate": s["correct_top1"],
                                         "discordant_top1": int((p0["argmax"] != p["argmax"]).sum()),
                                         "logits_sha256_identical": s0["logits_sha256"] == s["logits_sha256"]}
    out["PASS"] = all(c.values())
    d = Path(a.output_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{a.run_id}.json", "x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if a.smoke:
        print(SMOKE_BANNER)
    sys.exit(0 if out["PASS"] else 4)


if __name__ == "__main__":
    main()
