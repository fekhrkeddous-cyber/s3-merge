#!/usr/bin/env python3
"""Accuracy campaign for paper Table 1 (Top-1 column): runs the paper run list ONE AT A TIME, each as an independent
Python process (accuracy_eval.py), followed by accuracy_check.py. Any failing run or check stops the campaign.

Run list (unified_common.ACCURACY_RUNS):
  baseline_s0                    unmerged DeiT-S/16, seed 0 (deterministic; reported once)
  tome_r10_s0, tome_r10_s0_rep   official ToMe r=10, seed 0: primary run + determinism replicate (both reported)
  tome_r12_s0, tome_r12_s0_rep   official ToMe r=12, likewise
  s3_acc_s0 .. s3_acc_s4         S3-Merge-Acc, seeds 0..4 (random targets: stochastic)
  s3_speed_s0 .. s3_speed_s4     S3-Merge-Speed, seeds 0..4

Layout of <output-dir>: runs/<run-id>/, checks/<run-id>.json, logs/<run-id>.log, accuracy_run_manifest.tsv,
accuracy_campaign_status.json. Then run accuracy_analysis.py --runs-dir <output-dir>/runs.
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, subprocess  # noqa: E401,E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unified_common as U  # noqa: E402

RUN_IDS = [r["run_id"] for r in U.ACCURACY_RUNS]


def build_parser():
    ap = argparse.ArgumentParser(description="Run the Table 1 accuracy run list sequentially.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output-dir", required=True, help="campaign directory (runs/, checks/, logs/)")
    ap.add_argument("--data-dir", default=None, help="ImageNet-1K validation arrow shards (see accuracy_eval.py); "
                    "required unless --smoke")
    ap.add_argument("--runs", nargs="+", choices=RUN_IDS, default=RUN_IDS, help="subset of the run list, in this order")
    ap.add_argument("--resume", action="store_true", help="skip runs that are complete and passed their check")
    ap.add_argument("--num-shards", type=int, default=U.N_SHARDS)
    ap.add_argument("--batch-size", type=int, default=U.EVAL_BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=U.EVAL_NUM_WORKERS)
    U.add_model_args(ap)
    ap.add_argument("--smoke", action="store_true", help="SOFTWARE TEST ONLY: synthetic images (passed to every run)")
    ap.add_argument("--smoke-images", type=int, default=8)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if not a.smoke and not a.data_dir:
        raise SystemExit("--data-dir is required unless --smoke")
    out = Path(a.output_dir)
    runs_dir, checks, logs = out / "runs", out / "checks", out / "logs"
    man, status_p = out / "accuracy_run_manifest.tsv", out / "accuracy_campaign_status.json"
    cfg = {"script": "accuracy_campaign.py", "output_dir": str(out), "data_dir": a.data_dir,
           "runs": [U.ACCURACY_RUN_INDEX[r] for r in a.runs], "resume": a.resume, "num_shards": a.num_shards,
           "batch_size": a.batch_size, "num_workers": a.num_workers, "device": a.device, "checkpoint": a.checkpoint,
           "expected_sha256": a.expected_sha256, "expected_state_sha256": a.expected_state_sha256,
           "pretrained": not a.no_pretrained, "tome_path": a.tome_path, "verify_tome": not a.no_verify_tome,
           "smoke": a.smoke, "smoke_images": a.smoke_images if a.smoke else None}
    U.print_config(cfg)
    for d in (runs_dir, checks, logs):
        d.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{U.utc()}\t{msg}"
        print(line, flush=True)
        with open(logs / "accuracy_campaign.log", "a") as f:
            f.write(line + "\n")

    st = {"started_utc": U.utc(), "completed": [], "resume": a.resume, "smoke": a.smoke, "config": cfg}

    def stop(reason):
        st.update(status="STOPPED", reason=reason, stopped_utc=U.utc())
        status_p.write_text(json.dumps(st, indent=1))
        log("STOP: " + reason)
        sys.exit(5)

    if not man.exists():
        man.write_text("order\trun_id\tmethod\tseed\trole\tstart_utc\tend_utc\trc\tcheck_pass\trun_summary_sha256\tpredictions_sha256\n")
    common = ["--output-dir", str(runs_dir), "--num-shards", str(a.num_shards), "--batch-size", str(a.batch_size),
              "--num-workers", str(a.num_workers), "--device", a.device, "--expected-sha256", a.expected_sha256,
              "--expected-state-sha256", a.expected_state_sha256]
    for flag, val in (("--data-dir", a.data_dir), ("--checkpoint", a.checkpoint), ("--tome-path", a.tome_path)):
        if val:
            common += [flag, val]
    for flag, on in (("--no-pretrained", a.no_pretrained), ("--no-verify-tome", a.no_verify_tome), ("--smoke", a.smoke)):
        if on:
            common.append(flag)
    if a.smoke:
        common += ["--smoke-images", str(a.smoke_images)]
    for k, rid in enumerate(a.runs):
        run = U.ACCURACY_RUN_INDEX[rid]
        d = runs_dir / rid
        if (d / "RUN_COMPLETE.json").exists():
            chk = checks / f"{rid}.json"
            if a.resume and chk.exists() and json.loads(chk.read_text())["PASS"]:
                st["completed"].append(rid); log(f"{rid}: complete and checked (resume) -> skip"); continue  # noqa: E702
            stop(f"{rid}: output exists (collision)")
        if d.exists():
            stop(f"{rid}: partial run directory present: {d}")
        cmd = [sys.executable, str(HERE / "accuracy_eval.py"), "--arm", run["arm"], "--seed", str(run["seed"]),
               "--run-id", rid, *common]
        t0 = U.utc(); log(f"{rid}: START")  # noqa: E702
        with open(logs / f"{rid}.log", "x") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        t1 = U.utc(); log(f"{rid}: END rc={rc}")  # noqa: E702
        chk_ok = False
        if rc == 0:
            ccmd = [sys.executable, str(HERE / "accuracy_check.py"), "--runs-dir", str(runs_dir), "--run-id", rid,
                    "--output-dir", str(checks)] + (["--smoke"] if a.smoke else [])
            r = subprocess.run(ccmd, capture_output=True, text=True)
            chk_ok = r.returncode == 0
            (logs / f"{rid}.check.log").write_text(r.stdout + r.stderr)
        comp = json.loads((d / "RUN_COMPLETE.json").read_text())["files_sha256"] if (d / "RUN_COMPLETE.json").exists() else {}
        with open(man, "a") as mf:
            mf.write("\t".join(map(str, [k + 1, rid, run["arm"], run["seed"], run["role"], t0, t1, rc, chk_ok,
                                         comp.get("run_summary.json", ""), comp.get("predictions.npz", "")])) + "\n")
        if rc != 0 or not chk_ok:
            stop(f"{rid}: rc={rc} check_pass={chk_ok}")
        st["completed"].append(rid)
        status_p.write_text(json.dumps(dict(st, status="RUNNING", last_completed=rid), indent=1))
    st.update(status="COMPLETE", finished_utc=U.utc())
    status_p.write_text(json.dumps(st, indent=1))
    log("accuracy campaign COMPLETE")
    if a.smoke:
        print(U.SMOKE_BANNER)


if __name__ == "__main__":
    main()
