#!/usr/bin/env python3
"""Full-validation accuracy driver for paper Table 1 (Top-1 column): ONE arm x ONE seed per process.
Run the complete list (baseline; official ToMe r=10/r=12 primary + determinism replicate; S3-Merge-Acc/-Speed seeds
0..4) with accuracy_campaign.py.

Protocol (identical for every arm): HF ImageNet-1K validation, hf_index 0..49999 in order; Resize(256) bilinear,
CenterCrop(224), ToTensor, ImageNet Normalize; batch 64, shuffle False, 8 workers; FP32, no autocast/compile, TF32
matmul off; torch.manual_seed(s); np.random.seed(s) once at run start; on CUDA the CUDA RNG is verified untouched until
the first forward (so every S3 run draws its random targets from the same stream for a given seed); forward boundary
no_grad + synchronize before and after. Per batch the arm's structural invariants are asserted (per-block token
trajectory, attn_mask pattern, final tokens, token mass) and any violation stops the run.

Outputs in <output-dir>/<run-id>/ (written atomically; the run directory must not exist):
  predictions.npz   hf_index, argmax, top5_idx, top5_logit, margin_top1_top2, label, correct_top1, correct_top5
  run_summary.json  counts, Top-1/Top-5, per-batch correct counts, invariants, logits sha256, environment, config
  RUN_COMPLETE.json sha256 of the files above            (RUN_FAILED.json with the traceback on failure)

--smoke is for SOFTWARE TESTING ONLY: a few random synthetic images (random labels) instead of ImageNet, CPU and
--no-pretrained allowed; outputs carry "smoke": true and are not paper measurements.
"""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, io, json, time, traceback  # noqa: E401,E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_common as U  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description="ImageNet-1K validation accuracy of one Table 1 arm and seed.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--arm", choices=U.ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0, help="torch.manual_seed / np.random.seed at run start "
                    "(paper: 0..4 for s3_acc/s3_speed, 0 for baseline and ToMe)")
    ap.add_argument("--run-id", default=None, help="run directory name (default: <arm>_s<seed>; the ToMe determinism "
                    "replicates use <arm>_s0_rep)")
    ap.add_argument("--output-dir", required=True, help="parent directory of the run directory")
    ap.add_argument("--data-dir", default=None, help="directory holding (or any parent of) the HF arrow shards "
                    f"{U.VAL_SHARD_NAME.format(i=0, n=U.N_SHARDS)} .. of ILSVRC/imagenet-1k validation (required "
                    "unless --smoke)")
    ap.add_argument("--num-shards", type=int, default=U.N_SHARDS, help="number of validation arrow shards")
    ap.add_argument("--batch-size", type=int, default=U.EVAL_BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=U.EVAL_NUM_WORKERS)
    U.add_model_args(ap)
    ap.add_argument("--smoke", action="store_true", help="SOFTWARE TEST ONLY: synthetic images instead of ImageNet")
    ap.add_argument("--smoke-images", type=int, default=8, help="number of synthetic images in --smoke")
    return ap


def batch_checks(m, name, hooks, logits, B):
    exp_in, exp_mask, exp_final = U.expected_block_inputs(name), U.expected_attn_mask_blocks(name), U.expected_final_tokens(name)
    ok = {"block_inputs": hooks.block_in == exp_in,
          "attn_mask_blocks": [i for i, v in enumerate(hooks.mask) if v] == exp_mask,
          "final_tokens": hooks.final == exp_final,
          "logits_fp32_finite": logits.dtype == torch.float32 and tuple(logits.shape) == (B, 1000) and bool(torch.isfinite(logits).all())}
    fam = U.METHODS[name]["family"]
    if fam == "official_tome":
        s = m["model"]._tome_info["size"].squeeze(-1)
        ok["mass_197"] = tuple(s.shape) == (B, exp_final) and bool((s.sum(1) == 197).all()) and bool((s >= 1).all())
    elif fam == "s3_canonical":
        mb = m["model"].blocks[U.METHODS[name]["apply_patch"]["late_start"]]
        s = mb._token_sizes
        ok["mass_197_cls_1"] = tuple(s.shape) == (B, exp_final) and bool((s[:, 0] == 1).all()) and bool((s.sum(1) == 197).all()) and bool((s >= 1).all())
    return ok


def main(argv=None):
    a = build_parser().parse_args(argv)
    dev = U.validate_run_args(a)
    if not a.smoke and not a.data_dir:
        raise SystemExit("--data-dir is required (ImageNet-1K validation arrow shards) unless --smoke")
    run_id = a.run_id or f"{a.arm}_s{a.seed}"
    known = U.ACCURACY_RUN_INDEX.get(run_id)
    if known is not None and (known["arm"] != a.arm or known["seed"] != a.seed):
        raise SystemExit(f"run id {run_id} is reserved for arm {known['arm']} seed {known['seed']}")
    out = Path(a.output_dir) / run_id
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    cfg = {"script": "accuracy_eval.py", "arm": a.arm, "seed": a.seed, "run_id": run_id,
           "run_role": known["role"] if known else "not in the paper run list", "output": str(out),
           "data_dir": a.data_dir, "num_shards": a.num_shards, "batch_size": a.batch_size, "num_workers": a.num_workers,
           "device": str(dev), "precision": "FP32 eager; autocast off; no torch.compile; TF32 matmul off",
           "checkpoint": a.checkpoint, "expected_sha256": a.expected_sha256,
           "expected_state_sha256": a.expected_state_sha256, "pretrained": not a.no_pretrained, "tome_path": a.tome_path,
           "verify_tome": not a.no_verify_tome, "method": U.METHODS[a.arm], "smoke": a.smoke,
           "smoke_images": a.smoke_images if a.smoke else None}
    U.print_config(cfg)
    t_start = U.utc()
    _, _, tome_hashes = U.load_tome(a.tome_path, verify=not a.no_verify_tome)
    ckpt = U.check_checkpoint(a)
    shards = None if a.smoke else U.resolve_val_shards(a.data_dir, a.num_shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    try:
        s = evaluate(a, dev, cfg, out, shards, ckpt, tome_hashes, t_start)
    except BaseException as e:
        U.write_atomic_json(out / "RUN_FAILED.json", {"utc": U.utc(), "error": repr(e), "traceback": traceback.format_exc()})
        raise
    print(json.dumps({k: s.get(k) for k in ("run_id", "method", "seed", "n_images", "top1_pct", "invariants_all_pass",
                                            "logits_sha256", "smoke")}, indent=1))
    if a.smoke:
        print(U.SMOKE_BANNER)


def evaluate(a, dev, cfg, out, shards, ckpt, tome_hashes, t_start):
    cuda = dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    U.set_backends()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng0 = torch.cuda.get_rng_state().clone() if cuda else None
    builder = U.ArmBuilder(dev, a.checkpoint, pretrained=not a.no_pretrained, expected_state_sha256=a.expected_state_sha256)
    m = builder.build(a.arm)
    hooks = U.Hooks(m["inner"], U.final_norm_of(m["model"]))
    ds = U.SyntheticImages(a.smoke_images) if a.smoke else U.ImageNetVal(shards, with_labels=True)
    indices = list(ds.indices)
    loader = U.make_loader(ds, a.batch_size, a.num_workers, pin_memory=cuda)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    inv, idx_all, am_all, t5i, t5v, marg, lab, pbc = {}, [], [], [], [], [], [], []
    h = hashlib.sha256()
    first, t0 = True, time.perf_counter()
    for x, y, idx in loader:
        if first:
            if cuda and not torch.equal(torch.cuda.get_rng_state(), rng0):
                raise SystemExit("CUDA RNG consumed before the first forward")
            first = False
        x = x.to(dev, non_blocking=True)
        hooks.reset()
        with torch.no_grad():
            sync()
            logits = m["model"](x)
            sync()
        ok = batch_checks(m, a.arm, hooks, logits, x.shape[0])
        for k, v in ok.items():
            inv[k] = inv.get(k, True) and bool(v)
        if not all(ok.values()):
            raise SystemExit(f"INVARIANT FAILURE {ok} block_in={hooks.block_in} mask={hooks.mask} final={hooks.final}")
        v5, i5 = logits.topk(5, 1)
        am = logits.argmax(-1)
        h.update(logits.detach().cpu().numpy().tobytes())
        idx_all.append(idx.clone()); am_all.append(am.to(torch.int16).cpu()); t5i.append(i5.to(torch.int16).cpu())  # noqa: E702
        t5v.append(v5.cpu()); marg.append((v5[:, 0] - v5[:, 1]).cpu())  # noqa: E702
        pbc.append(int((am == y.to(dev)).sum().item())); lab.append(y.to(torch.int16).clone())  # noqa: E702
    loop = time.perf_counter() - t0
    idx_np = torch.cat(idx_all).numpy().astype(np.int32)
    am_np, i5_np = torch.cat(am_all).numpy(), torch.cat(t5i).numpy()
    inv["index_order_and_count"] = idx_np.tolist() == list(indices)
    inv["argmax_equals_top1"] = bool((am_np == i5_np[:, 0]).all())
    arrays = {"hf_index": idx_np, "argmax": am_np, "top5_idx": i5_np, "top5_logit": torch.cat(t5v).numpy(),
              "margin_top1_top2": torch.cat(marg).numpy()}
    lab_np = torch.cat(lab).numpy()
    c1, c5 = am_np == lab_np, (i5_np == lab_np[:, None]).any(1)
    arrays.update({"label": lab_np, "correct_top1": c1, "correct_top5": c5})
    assert sum(pbc) == int(c1.sum())
    s = {"run_id": out.name, "method": a.arm, "seed": a.seed, "mode": "smoke" if a.smoke else "full",
         "smoke": a.smoke, "n_images": int(idx_np.shape[0]), "labels_read": True,
         "config": U.METHODS[a.arm], "build_info": m["info"],
         "expected_block_inputs": U.expected_block_inputs(a.arm), "expected_final_tokens": U.expected_final_tokens(a.arm),
         "expected_attn_mask_blocks": U.expected_attn_mask_blocks(a.arm),
         "correct_top1": int(c1.sum()), "top1_pct": 100.0 * float(c1.sum()) / len(c1), "correct_top5": int(c5.sum()),
         "top5_pct": 100.0 * float(c5.sum()) / len(c5), "per_batch_correct": pbc,
         "per_batch_correct_sha256": hashlib.sha256(json.dumps(pbc).encode()).hexdigest()}
    if a.smoke:
        s["label"] = U.SMOKE_BANNER + " Synthetic images with random labels; the 'accuracy' is meaningless."
    s.update({"invariants": inv, "invariants_all_pass": all(inv.values()), "logits_sha256": h.hexdigest(),
              "peak_mem_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if cuda else None, "loop_s": loop,
              "started_utc": t_start, "finished_utc": U.utc(), "env": U.env_record(dev), "gpu_end": U.gpu_snapshot(),
              "checkpoint": ckpt, "tome_source_sha256": tome_hashes, "resolved_config": cfg, "argv": sys.argv})
    buf = io.BytesIO(); np.savez(buf, **arrays)  # noqa: E702
    U.write_atomic_bytes(out / "predictions.npz", buf.getvalue())
    if not s["invariants_all_pass"]:
        U.write_atomic_json(out / "run_summary.json", s)
        raise SystemExit(f"INVARIANT FAILURE {inv}")
    U.write_atomic_json(out / "run_summary.json", s)
    U.write_atomic_json(out / "RUN_COMPLETE.json", {"run_id": out.name, "utc": U.utc(), "files_sha256": {
        "predictions.npz": U.sha256_file(out / "predictions.npz"), "run_summary.json": U.sha256_file(out / "run_summary.json")}})
    return s


if __name__ == "__main__":
    main()
