#!/usr/bin/env python3
"""Unified timing CAMPAIGN for paper Table 1 / Fig. 5: ten latency invocations (0..9) and three profiling
invocations (0..2), each an INDEPENDENT Python process running timing_unified.py with its own randomized arm order,
then timing_analysis.py.

Before the campaign an OPTIONAL, warn-only GPU idle check samples nvidia-smi (--idle-check-samples). During every
invocation nvidia-smi telemetry is logged once per second to <output-dir>/monitor/ (disable with --no-monitor) and the
GPU compute processes are listed before and after each latency invocation. Nothing here signals or alters any other
process. Any failing invocation aborts the campaign.

Layout of <output-dir>: raw/ (latency_inv<NN>.json, profile_inv<NN>.json), monitor/, logs/, analysis/.
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, shutil, subprocess  # noqa: E401,E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unified_common as U  # noqa: E402

MONITOR_QUERY = ("timestamp,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,"
                 "clocks_event_reasons.active")


def build_parser():
    ap = argparse.ArgumentParser(description="Run the unified Table 1 timing campaign (independent invocations).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output-dir", required=True, help="campaign directory (raw/, monitor/, logs/, analysis/)")
    ap.add_argument("--invocations", type=int, default=U.N_LATENCY_INVOCATIONS, help="latency invocations 0..n-1")
    ap.add_argument("--profile-invocations", type=int, default=U.N_PROFILE_INVOCATIONS,
                    help="profiling invocations 0..n-1 (run after all latency invocations; 0 = none)")
    ap.add_argument("--arms", nargs="+", choices=U.ARMS, default=list(U.ARMS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=list(U.BATCH_SIZES))
    ap.add_argument("--warmup", type=int, default=U.WARMUP)
    ap.add_argument("--iters", type=int, default=U.TIMED)
    ap.add_argument("--profiled-forwards", type=int, default=U.PROFILED_FORWARDS)
    ap.add_argument("--idle-check-samples", type=int, default=60,
                    help="1-s nvidia-smi samples of the optional warn-only GPU idle check before the campaign (0 = skip)")
    ap.add_argument("--no-monitor", action="store_true", help="do not log nvidia-smi telemetry during the invocations")
    ap.add_argument("--accuracy-analysis", default=None, help="accuracy_analysis.json passed to timing_analysis.py")
    ap.add_argument("--skip-analysis", action="store_true", help="do not run timing_analysis.py at the end")
    U.add_model_args(ap)
    ap.add_argument("--smoke", action="store_true", help="SOFTWARE TEST ONLY: passes --smoke to every invocation")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = Path(a.output_dir)
    raw, mon, logs = out / "raw", out / "monitor", out / "logs"
    cfg = {"script": "timing_campaign.py", "output_dir": str(out), "invocations": a.invocations,
           "profile_invocations": a.profile_invocations, "arms": a.arms, "batch_sizes": a.batch_sizes,
           "warmup": a.warmup, "iters": a.iters, "profiled_forwards": a.profiled_forwards,
           "idle_check_samples": a.idle_check_samples, "monitor": not a.no_monitor, "device": a.device,
           "checkpoint": a.checkpoint, "expected_sha256": a.expected_sha256,
           "expected_state_sha256": a.expected_state_sha256, "pretrained": not a.no_pretrained,
           "tome_path": a.tome_path, "verify_tome": not a.no_verify_tome, "accuracy_analysis": a.accuracy_analysis,
           "orders": {m: {i: [k for k in o if k in a.arms] for i, o in d.items()} for m, d in U.invocation_orders().items()},
           "smoke": a.smoke}
    U.print_config(cfg)
    for d in (raw, mon, logs):
        d.mkdir(parents=True, exist_ok=True)
    have_smi = shutil.which("nvidia-smi") is not None

    def log(msg):
        line = f"{U.utc()}\t{msg}"
        print(line, flush=True)
        with open(logs / "timing_campaign.log", "a") as f:
            f.write(line + "\n")

    def apps(tag):
        (mon / f"apps_{tag}.txt").write_text(U.compute_apps() + "\n")

    if a.idle_check_samples > 0:
        log(f"optional GPU idle check ({a.idle_check_samples} samples)")
        chk = U.gpu_idle_check(a.idle_check_samples)
        U.write_atomic_json(logs / f"idle_check_{chk['utc'].replace(':', '')}.json", chk)
        for w in chk["warnings"]:
            log(f"WARNING: {w}")
    common = ["--output-dir", str(raw), "--arms", *a.arms, "--batch-sizes", *map(str, a.batch_sizes),
              "--warmup", str(a.warmup), "--iters", str(a.iters), "--profiled-forwards", str(a.profiled_forwards),
              "--device", a.device, "--expected-sha256", a.expected_sha256, "--expected-state-sha256", a.expected_state_sha256]
    for flag, val in (("--checkpoint", a.checkpoint), ("--tome-path", a.tome_path)):
        if val:
            common += [flag, val]
    for flag, on in (("--no-pretrained", a.no_pretrained), ("--no-verify-tome", a.no_verify_tome), ("--smoke", a.smoke)):
        if on:
            common.append(flag)
    runs = [("latency", i) for i in range(a.invocations)] + [("profile", i) for i in range(a.profile_invocations)]
    for mode, i in runs:
        tag = f"{mode}_inv{i:02d}"
        if mode == "latency":
            apps(f"before_{tag}")
        mp = mf = None
        if have_smi and not a.no_monitor:
            mf = open(mon / f"{tag}.csv", "w")
            mp = subprocess.Popen(["nvidia-smi", f"--query-gpu={MONITOR_QUERY}", "--format=csv,noheader", "-lms", "1000"],
                                  stdout=mf, stderr=subprocess.DEVNULL)
        log(f"{mode} invocation {i:02d}")
        try:
            with open(logs / f"{tag}.log", "w") as lf:
                rc = subprocess.run([sys.executable, str(HERE / "timing_unified.py"), "--mode", mode, "--invocation", str(i),
                                     *common], stdout=lf, stderr=subprocess.STDOUT).returncode
        finally:
            if mp is not None:
                mp.terminate()
                mp.wait()
                mf.close()
        if mode == "latency":
            apps(f"after_{tag}")
        if rc != 0:
            log(f"ABORT: {tag} exited with {rc} (see {logs / (tag + '.log')})")
            raise SystemExit(rc)
    if not a.skip_analysis:
        log("analysis")
        cmd = [sys.executable, str(HERE / "timing_analysis.py"), "--raw-dir", str(raw), "--output-dir", str(out / "analysis"),
               "--invocations", str(a.invocations), "--arms", *a.arms, "--batch-sizes", *map(str, a.batch_sizes)]
        if a.accuracy_analysis:
            cmd += ["--accuracy-analysis", a.accuracy_analysis]
        if a.smoke:
            cmd.append("--smoke")
        with open(logs / "analysis.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            log(f"ABORT: analysis exited with {rc} (see {logs / 'analysis.log'})")
            raise SystemExit(rc)
    log("timing campaign complete")
    print(json.dumps({"output_dir": str(out), "smoke": a.smoke}, indent=1))
    if a.smoke:
        print(U.SMOKE_BANNER)


if __name__ == "__main__":
    main()
