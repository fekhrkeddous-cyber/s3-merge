"""Factorial-study accuracy driver (paper Table 4 Top-1; Supplementary Tables S17-S18): one arm x one seed per process.

Full run: all 50,000 ImageNet-1K validation images in fixed order 0..49999 (no shuffle), labels read, Top-1/Top-5.

Protocol (identical for every arm): torch.manual_seed(s); np.random.seed(s) once at run start; batch 64;
shuffle=False; FP32; no autocast; no torch.compile; PA off; one forward boundary (no_grad, synchronize before and
after). On CUDA the CUDA RNG is verified untouched between seeding and the first forward, so every arm receives
the same priority draws per image for a given seed. Trajectory, attn_mask absence, FP32 logits and (factorial
arms) the post-forward merge invariants are checked on every batch; any violation stops the run.

Outputs in <output-dir>/<run-id>/ (written atomically, never overwritten; the run directory must not exist):
  predictions.npz   hf_index, argmax, top5_idx, top5_logit, margin_top1_top2, label, correct_top1, correct_top5
  targets_u8.npy    factorial arms: final 84 target coordinates per image (canonical topk order)
  run_summary.json  counts, Top-1/Top-5, invariants, environment, resolved configuration
  RUN_COMPLETE.json sha256 of the files above

--smoke is for SOFTWARE TESTING ONLY: a few synthetic images (random labels), CPU allowed; outputs are marked
"smoke": true and are not paper measurements.
"""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, io, json, time, traceback  # noqa: E401,E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factorial_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

BATCH_SIZE, NUM_WORKERS = 64, 8                  # paper protocol
SMOKE_BATCH_SIZE, SMOKE_NUM_WORKERS = 8, 0       # software test only


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Factorial-study accuracy run (one arm x one seed per process). Defaults = paper protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--arm", choices=C.ARMS, required=True,
                    help="baseline = unmerged; core = canonical S3-Merge Core (PA off, control); SW/SG/PW/PG = factorial arms")
    ap.add_argument("--seed", type=int, required=True, help="torch.manual_seed / np.random.seed applied once at run start")
    ap.add_argument("--run-id", default=None, help="run directory name (default: <arm>_s<seed>)")
    ap.add_argument("--output-dir", default="runs/r3_factorial/accuracy", help="parent directory of the run directory")
    ap.add_argument("--data-dir", default=None,
                    help="directory holding the Hugging Face arrow shards of ILSVRC/imagenet-1k (validation split); "
                         "required unless --smoke")
    ap.add_argument("--shard-pattern", default=C.DEFAULT_SHARD_PATTERN, help="glob for the validation shards inside --data-dir")
    ap.add_argument("--checkpoint", default=None,
                    help="local deit_small_patch16_224.fb_in1k model.safetensors (default: Hugging Face hub "
                         f"{C.HF_HUB_ID}, downloaded or taken from the hub cache)")
    ap.add_argument("--expected-sha256", default=C.CKPT_FILE_SHA, help="checkpoint file sha256 ('none' disables the check)")
    ap.add_argument("--expected-state-sha256", default=C.STATE_SHA,
                    help="sha256 of the loaded weights (state_sha method; 'none' disables the check)")
    ap.add_argument("--device", default="cuda", help="torch device (the paper used one H100 NVL)")
    ap.add_argument("--batch-size", type=int, default=None, help="default 64 (8 with --smoke)")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers; default 8 (0 with --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="SOFTWARE TEST ONLY: synthetic images, tiny count; results are not paper measurements")
    ap.add_argument("--smoke-images", type=int, default=20, help="number of synthetic images with --smoke")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random (seeded) weights, no checkpoint; only allowed with --smoke")
    a = ap.parse_args(argv)
    if a.no_pretrained and not a.smoke:
        ap.error("--no-pretrained is only allowed with --smoke")
    if not a.smoke and not a.data_dir:
        ap.error("--data-dir is required (or use --smoke for a software test)")
    return a


def resolved_config(a):
    return {
        "script": "r3/factorial/accuracy.py", "arm": a.arm, "seed": a.seed,
        "run_id": a.run_id or f"{a.arm}_s{a.seed}", "output_dir": str(Path(a.output_dir).resolve()),
        "smoke": bool(a.smoke), "device": a.device,
        "data": ({"source": "synthetic random RGB images (smoke)", "n_images": a.smoke_images} if a.smoke else
                 {"dataset": "Hugging Face ILSVRC/imagenet-1k, split validation", "data_dir": str(Path(a.data_dir).resolve()),
                  "shard_pattern": a.shard_pattern, "open_method": "datasets.Dataset.from_file per shard (sorted) + "
                  "concatenate_datasets", "n_images": C.N_VAL, "order": "hf_index 0..49999, shuffle=False"}),
        "preprocessing": "image.convert('RGB') -> Resize(256) -> CenterCrop(224) -> ToTensor -> "
                         "Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])",
        "loader": {"batch_size": a.batch_size if a.batch_size is not None else (SMOKE_BATCH_SIZE if a.smoke else BATCH_SIZE),
                   "num_workers": a.num_workers if a.num_workers is not None else (SMOKE_NUM_WORKERS if a.smoke else NUM_WORKERS),
                   "shuffle": False, "drop_last": False, "pin_memory": torch.device(a.device).type == "cuda",
                   "worker_generator_seed": 0},
        "model": {"name": C.MODEL_NAME, "weights": "none (random, smoke)" if a.no_pretrained else C.HF_HUB_ID,
                  "checkpoint": a.checkpoint, "expected_sha256": C.none_or(a.expected_sha256),
                  "expected_state_sha256": C.none_or(a.expected_state_sha256), "pretrained": not a.no_pretrained},
        "arm_definition": ({"core_apply_patch_kwargs": C.CORE_KWARGS} if a.arm == "core" else
                           {"merge_after_block_index": list(C.ARM_SPECS[a.arm][0]), "removals_per_window": list(C.ARM_SPECS[a.arm][1]),
                            "scope": C.ARM_SPECS[a.arm][2]} if a.arm in C.FACT else {"merge": None}),
        "expected_trajectory": C.EXPECT_TRAJ[a.arm],
        "inference": {"precision": "FP32", "autocast": False, "torch_compile": False, "tf32_matmul": False,
                      "tf32_cudnn": True, "cudnn_benchmark": False, "proportional_attention": False},
        "seeding": "torch.manual_seed(seed); np.random.seed(seed) once at run start",
    }


def main(argv=None):
    a = parse_args(argv)
    cfg = resolved_config(a)
    C.print_config(cfg)
    if a.smoke:
        print(C.SMOKE_NOTICE, flush=True)
    t_start = C.utc()
    out = Path(a.output_dir) / cfg["run_id"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()                                      # FileExistsError -> output-path collision -> stop
    try:
        summary = evaluate(a, cfg, out, t_start)
    except BaseException as e:
        C.write_atomic_json(out / "RUN_FAILED.json", {"run_id": cfg["run_id"], "utc": C.utc(), "error": repr(e),
                                                     "traceback": traceback.format_exc()})
        raise
    print(json.dumps({k: summary[k] for k in ("run_id", "arm", "seed", "smoke", "n_images", "correct_top1",
                                              "top1_pct", "invariants_all_pass", "logits_sha256", "duration_s")}, indent=1))
    if a.smoke:
        print(C.SMOKE_NOTICE, flush=True)


def evaluate(a, cfg, out, t_start):
    dev = torch.device(a.device)
    is_cuda = dev.type == "cuda"
    ckpt, ck_sha = C.resolve_checkpoint(a.checkpoint, C.none_or(a.expected_sha256), pretrained=not a.no_pretrained)
    bs, nw = cfg["loader"]["batch_size"], cfg["loader"]["num_workers"]
    if a.smoke:
        dataset, indices = C.SyntheticDS(a.smoke_images), list(range(a.smoke_images))
    else:
        files = C.shard_paths(a.data_dir, a.shard_pattern)
        indices = list(range(C.N_VAL))
        dataset = C.ValDS(files, indices, with_labels=True)
    C.set_backends()
    torch.manual_seed(a.seed)                         # seeding once at run start
    np.random.seed(a.seed)
    rng_after_seed = torch.cuda.get_rng_state(dev).clone() if is_cuda else None
    m = C.build_model(a.arm, ckpt, dev, pretrained=not a.no_pretrained,
                      expected_state_sha=None if a.no_pretrained else C.none_or(a.expected_state_sha256))
    loader = C.make_loader(dataset, bs, nw, pin_memory=is_cuda)
    gpu_start = C.gpu_snapshot()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    is_fact = a.arm in C.FACT

    idx_all, lab_all, top5i, top5v, margin, argmax_all, reps = [], [], [], [], [], [], []
    per_batch_correct, inv_all = [], {}
    n_cross, size_hist = 0, np.zeros(C.N_PATCH + 1, dtype=np.int64)
    h_logits = hashlib.sha256()
    t_busy = t_wait = 0.0
    first = True
    t_loop0 = time.perf_counter()
    t_w = time.perf_counter()
    for x, y, idx in loader:
        t_b0 = time.perf_counter()
        t_wait += t_b0 - t_w
        if first:
            if is_cuda and not torch.equal(torch.cuda.get_rng_state(dev), rng_after_seed):
                raise SystemExit("CUDA RNG consumed between seeding and the first forward")
            first = False
        x = x.to(dev, non_blocking=True)
        logits = C.forward(m, x)
        B = x.shape[0]
        v5, i5 = logits.topk(5, dim=1)
        am = logits.argmax(-1)
        h_logits.update(logits.detach().cpu().numpy().tobytes())
        top5i.append(i5.to(torch.int16).cpu())
        top5v.append(v5.float().cpu())
        margin.append((v5[:, 0] - v5[:, 1]).cpu())
        argmax_all.append(am.to(torch.int16).cpu())
        idx_all.append(idx.clone())
        yy = y.to(dev)
        per_batch_correct.append(int((am == yy).sum().item()))
        lab_all.append(y.to(torch.int16).clone())
        if is_fact:
            rep_u8, ok, st = C.factorial_state_checks(m, B)
            reps.append(rep_u8)
            for k, v in ok.items():
                inv_all[k] = inv_all.get(k, True) and bool(v)
            n_cross += st["n_cross_window_members"]
            size_hist += st["size_hist"]
        t_b1 = time.perf_counter()
        t_busy += t_b1 - t_b0
        t_w = time.perf_counter()
    t_loop = time.perf_counter() - t_loop0

    idx_np = torch.cat(idx_all).numpy().astype(np.int32)
    n = int(idx_np.shape[0])
    order_ok = idx_np.tolist() == list(indices)
    i5_np, v5_np = torch.cat(top5i).numpy(), torch.cat(top5v).numpy()
    am_np = torch.cat(argmax_all).numpy()
    inv_all["index_order_and_count"] = order_ok and n == len(indices)
    inv_all["argmax_equals_top1"] = bool((am_np == i5_np[:, 0]).all())
    arrays = {"hf_index": idx_np, "argmax": am_np, "top5_idx": i5_np, "top5_logit": v5_np,
              "margin_top1_top2": torch.cat(margin).numpy()}
    summary = {"run_id": cfg["run_id"], "arm": a.arm, "seed": a.seed, "smoke": bool(a.smoke), "n_images": n,
               "batch_size": bs, "n_batches": len(per_batch_correct), "num_workers": nw, "labels_read": True}
    if a.smoke:
        summary["notice"] = C.SMOKE_NOTICE
    lab_np = torch.cat(lab_all).numpy()
    c1 = am_np == lab_np
    c5 = (i5_np == lab_np[:, None]).any(1)
    arrays.update({"label": lab_np, "correct_top1": c1, "correct_top5": c5})
    assert sum(per_batch_correct) == int(c1.sum())
    summary.update({"correct_top1": int(c1.sum()), "top1_pct": 100.0 * float(c1.sum()) / n,
                    "correct_top5": int(c5.sum()), "top5_pct": 100.0 * float(c5.sum()) / n,
                    "per_batch_correct_sha256": hashlib.sha256(json.dumps(per_batch_correct).encode()).hexdigest(),
                    "per_batch_correct": per_batch_correct})
    files = {}
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    C.write_atomic_bytes(out / "predictions.npz", buf.getvalue())
    files["predictions.npz"] = C.sha256_file(out / "predictions.npz")
    if is_fact:
        rep_np = torch.cat(reps).numpy()
        buf = io.BytesIO()
        np.save(buf, rep_np)
        C.write_atomic_bytes(out / "targets_u8.npy", buf.getvalue())
        files["targets_u8.npy"] = C.sha256_file(out / "targets_u8.npy")
        summary["targets_sha256_raw"] = hashlib.sha256(rep_np.tobytes()).hexdigest()
        summary["n_cross_window_members"] = int(n_cross)
        summary["cross_window_member_rate"] = float(n_cross) / (n * C.N_PATCH)
        summary["cluster_size_hist"] = {str(k): int(v) for k, v in enumerate(size_hist) if v}
    summary.update({
        "invariants": inv_all, "invariants_all_pass": all(inv_all.values()),
        "logits_sha256": h_logits.hexdigest(),
        "expected_trajectory": C.EXPECT_TRAJ[a.arm], "trajectory_checked_every_batch": True,
        "state_sha": m["state_sha"], "param_dtypes": m["param_dtypes"], "has_propattn": m["has_propattn"],
        "s3merge_config": m["s3merge_config"],
        "peak_mem_allocated_mib": torch.cuda.max_memory_allocated(dev) / 2**20 if is_cuda else None,
        "peak_mem_reserved_mib": torch.cuda.max_memory_reserved(dev) / 2**20 if is_cuda else None,
        "timing_s": {"loop": t_loop, "busy_batches": t_busy, "loader_wait": t_wait},
        "duration_s": t_loop, "started_utc": t_start, "finished_utc": C.utc(),
        "gpu_start": gpu_start, "gpu_end": C.gpu_snapshot(), "env": C.env_record(dev),
        "source_sha256": C.source_hashes(__file__), "config": cfg,
        "checkpoint": ckpt, "checkpoint_file_sha256": ck_sha, "checkpoint_file_sha256_expected": C.none_or(a.expected_sha256),
        "argv": sys.argv,
    })
    if not summary["invariants_all_pass"]:
        C.write_atomic_json(out / "run_summary.json", summary)
        raise SystemExit(f"INVARIANT FAILURE: {inv_all}")
    C.write_atomic_json(out / "run_summary.json", summary)
    files["run_summary.json"] = C.sha256_file(out / "run_summary.json")
    C.write_atomic_json(out / "RUN_COMPLETE.json", {"run_id": cfg["run_id"], "utc": C.utc(), "smoke": bool(a.smoke),
                                                   "files_sha256": files})
    return summary


if __name__ == "__main__":
    main()
