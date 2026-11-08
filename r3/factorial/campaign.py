#!/usr/bin/env python3
"""Campaign runner for the factorial study (Python stdlib only). Every run is a fresh Python process, started ONE AT
A TIME in the fixed order below; the campaign stops at the first non-zero exit code (outputs are preserved).

  accuracy  22 runs: baseline (seed 0), core (canonical S3-Merge Core control, seed 0), then for each seed 0..4 the
            factorial arms in the order SW, SG, PW, PG. Each run: accuracy.py --arm A --seed S --run-id A_sS.
  timing    10 latency invocations (0..9), then 3 profile invocations (0..2), then 5 merge_micro invocations (0..4):
            timing.py --mode M --invocation i. Only the latency invocations enter paper Table 4 / Table S19.

Any argument this script does not recognise is passed unchanged to every driver call (e.g. --data-dir, --checkpoint,
--device, --smoke, --no-pretrained). Use --dry-run to print the commands without running them.
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, os, subprocess, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
FACT = ["SW", "SG", "PW", "PG"]
TIMING_SCHEDULE = [("latency", 10), ("profile", 3), ("merge_micro", 5)]


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def accuracy_plan(seeds):
    plan = [("baseline", 0), ("core", 0)] + [(arm, s) for s in seeds for arm in FACT]
    return [{"run_id": f"{arm}_s{s}", "argv": ["--arm", arm, "--seed", str(s), "--run-id", f"{arm}_s{s}"]} for arm, s in plan]


def timing_plan(modes):
    return [{"run_id": f"{m}_inv{i:02d}", "argv": ["--mode", m, "--invocation", str(i)]}
            for m, n in TIMING_SCHEDULE if m in modes for i in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the factorial accuracy or timing campaign (one process per run).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    ap.add_argument("campaign", choices=["accuracy", "timing"])
    ap.add_argument("--output-dir", default=None,
                    help="driver output directory (default: runs/r3_factorial/accuracy or runs/r3_factorial/timing)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="accuracy: seeds of the factorial arms")
    ap.add_argument("--modes", nargs="+", choices=[m for m, _ in TIMING_SCHEDULE],
                    default=[m for m, _ in TIMING_SCHEDULE], help="timing: which invocation series to run")
    ap.add_argument("--resume", action="store_true", help="skip runs whose outputs are already complete")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter for the driver processes")
    a, passthrough = ap.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if "--output-dir" in passthrough:
        ap.error("pass --output-dir to campaign.py itself (before any driver arguments)")
    out = Path(a.output_dir or f"runs/r3_factorial/{a.campaign}")
    driver = HERE / ("accuracy.py" if a.campaign == "accuracy" else "timing.py")
    plan = accuracy_plan(a.seeds) if a.campaign == "accuracy" else timing_plan(a.modes)
    for p in plan:
        p["cmd"] = [a.python, str(driver)] + p["argv"] + ["--output-dir", str(out)] + passthrough
    cfg = {"script": "r3/factorial/campaign.py", "campaign": a.campaign, "output_dir": str(out.resolve()),
           "driver": str(driver), "passthrough_args": passthrough, "resume": a.resume, "dry_run": a.dry_run,
           "n_runs": len(plan), "order": [p["run_id"] for p in plan]}
    print("RESOLVED CONFIGURATION:\n" + json.dumps(cfg, indent=1), flush=True)
    if a.dry_run:
        for p in plan:
            print(" ".join(p["cmd"]))
        return
    out.mkdir(parents=True, exist_ok=True)
    status_p = out / f"campaign_{a.campaign}_status.json"
    state = {"config": cfg, "started_utc": utc(), "runs": []}

    def save(status):
        state["status"] = status
        tmp = status_p.with_name(status_p.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, status_p)

    for p in plan:
        if a.campaign == "accuracy":
            done = (out / p["run_id"] / "RUN_COMPLETE.json").exists()
            partial = (out / p["run_id"]).exists() and not done
        else:
            done = (out / f"{p['run_id']}.json").exists()
            partial = False
        if done:
            if a.resume:
                print(f"{utc()}\t{p['run_id']}: already complete -> skip", flush=True)
                state["runs"].append({"run_id": p["run_id"], "skipped": True})
                continue
            save("STOPPED")
            raise SystemExit(f"{p['run_id']}: output already exists (use --resume to skip completed runs)")
        if partial:
            save("STOPPED")
            raise SystemExit(f"{p['run_id']}: partial run directory present ({out / p['run_id']}); remove it to rerun")
        t0 = utc()
        print(f"{t0}\t{p['run_id']}: START {' '.join(p['cmd'])}", flush=True)
        rc = subprocess.run(p["cmd"]).returncode
        state["runs"].append({"run_id": p["run_id"], "cmd": p["cmd"], "start_utc": t0, "end_utc": utc(), "rc": rc})
        print(f"{utc()}\t{p['run_id']}: END rc={rc}", flush=True)
        if rc != 0:
            save("STOPPED")
            raise SystemExit(f"{p['run_id']}: exit code {rc}; campaign stopped")
        save("RUNNING")
    save("COMPLETE")
    print(f"{utc()}\tcampaign {a.campaign} COMPLETE ({len(plan)} runs)", flush=True)


if __name__ == "__main__":
    main()
