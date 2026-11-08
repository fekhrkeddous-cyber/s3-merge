"""Bounded GPU smoke test: EXECUTION CHECK ONLY.

Synthetic inputs, random weights (no downloads), a few warm-up and timed iterations. The latencies it prints are
NOT paper benchmarks and must not be compared with any reported number. Run from the repository root:

    python tests/smoke_gpu.py --output smoke_gpu.json
"""
import argparse
import json
import pathlib
import platform
import sys
import time

import torch
import timm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "r3" / "table7_vitb16"))
from models.sr_tome import apply_patch, PropAttnBlock  # noqa: E402
from models.sr_tome_downstream import SRToMeDownstream  # noqa: E402
from models.adaptive_k import PackGeometry, window_scores, allocate_uniform, adaptive_merge_pack, unpack_restore  # noqa: E402
import run_table7_vitb16 as t7  # noqa: E402


def timed(model, x, warmup, iters, amp=False):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def tokens_after(model, idx, x, amp=False):
    seen = {}
    h = model.blocks[idx].register_forward_hook(lambda m, i, o: seen.__setitem__("n", o.shape[1]))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        y = model(x)
    h.remove()
    return seen["n"], y


def main():
    ap = argparse.ArgumentParser(description="Bounded GPU smoke test (execution only; not a benchmark)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--output", default=None, help="optional JSON report path")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA device required"
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rep = {"note": "execution check only; latencies are not paper benchmarks",
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
                           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
           "warmup": a.warmup, "iters": a.iters, "checks": {}}
    chk = rep["checks"]
    torch.manual_seed(0)
    x224 = torch.randn(1, 3, 224, 224, device=dev)

    # 1. DeiT-S/16 224^2, FP32, B=1: baseline vs S3-Merge (late_start=8, T=84, w=7, PA off)
    base = timm.create_model("deit_small_patch16_224", pretrained=False).eval().to(dev)
    s3 = timm.create_model("deit_small_patch16_224", pretrained=False).eval().to(dev)
    s3.load_state_dict(base.state_dict())
    apply_patch(s3, target_tokens=84, late_start=8, window_size=7, prop_attn=False)
    rc = s3.s3merge_config
    n, y = tokens_after(s3, 8, x224)
    chk["deit_s_224_fp32"] = {
        "resolved": rc, "grid": s3.blocks[8].last_geometry["grid"], "tokens_after_merge": n,
        "pa_off": rc["prop_attn"] is False and not any(isinstance(b, PropAttnBlock) for b in s3.blocks),
        "merge_executed": s3.blocks[8].last_geometry["merges"] and n < 197,
        "logits_shape": list(y.shape), "finite": bool(torch.isfinite(y).all()), "dtype": str(y.dtype),
        "lat_ms_b1_baseline": timed(base, x224, a.warmup, a.iters), "lat_ms_b1_s3merge": timed(s3, x224, a.warmup, a.iters)}
    ok = (n == 85 and chk["deit_s_224_fp32"]["grid"] == 14 and chk["deit_s_224_fp32"]["pa_off"]
          and chk["deit_s_224_fp32"]["merge_executed"] and chk["deit_s_224_fp32"]["finite"])

    # 2. DeiT-S/16 at 384^2 (24x24 grid, w=8, T=360), PA off: the configuration that previously skipped silently
    m384 = timm.create_model("deit_small_patch16_224", pretrained=False, img_size=384).eval().to(dev)
    apply_patch(m384, target_tokens=360, late_start=8, window_size=8, prop_attn=False)
    n384, y384 = tokens_after(m384, 8, torch.randn(1, 3, 384, 384, device=dev))
    chk["deit_s_384_fp32"] = {"grid": m384.blocks[8].last_geometry["grid"], "tokens_after_merge": n384,
                              "finite": bool(torch.isfinite(y384).all())}
    ok &= n384 == 361 and chk["deit_s_384_fp32"]["grid"] == 24 and chk["deit_s_384_fp32"]["finite"]

    # 3. Table 7 ViT-B/16 configuration (AMP/FP16, PA off asserted), B=1
    cfg = t7.load_config(t7.DEFAULT_CONFIG)
    vb = t7.build_merged(cfg, dev, pretrained=False)
    vb_base = timm.create_model(cfg["model"], pretrained=False).eval().to(dev)
    with torch.cuda.amp.autocast():
        n7, geom7 = t7.check_runtime_tokens(vb, cfg, dev)
    _, y7 = tokens_after(vb, cfg["merge"]["late_start"], x224, amp=True)
    chk["table7_vitb16_amp"] = {"resolved": vb.s3merge_config, "tokens_after_merge": n7, "grid": geom7["grid"],
                                "output_dtype": str(y7.dtype), "finite": bool(torch.isfinite(y7).all()),
                                "lat_ms_b1_baseline": timed(vb_base, x224, a.warmup, a.iters, amp=True),
                                "lat_ms_b1_s3merge": timed(vb, x224, a.warmup, a.iters, amp=True)}
    ok &= n7 == 137 and vb.s3merge_config["prop_attn"] is False and chk["table7_vitb16_amp"]["finite"]

    # 4. Restore mapping (dense output) on the actual grid
    torch.manual_seed(0)
    dm = SRToMeDownstream(timm.create_model("deit_small_patch16_224", pretrained=False).eval(), target_tokens=84,
                          late_start=8, window_size=7).to(dev)
    with torch.no_grad():
        f = dm(x224)
    chk["restore_mapping"] = {"shape": list(f.shape), "finite": bool(torch.isfinite(f).all())}
    ok &= list(f.shape) == [1, 384, 14, 14]

    # 5. Packed exact-coordinate path on the GPU (64x56 grid, 8x8 windows, uniform K=56)
    geom = PackGeometry(64, 56, 8, dev)
    g = torch.randn(1, 64, 56, 384, device=dev)
    win = g.view(1, 8, 8, 7, 8, 384).permute(0, 1, 3, 2, 4, 5).reshape(1, 56, 64, 384)
    _, s_w = window_scores(win)
    packed, coords, restore, _ = adaptive_merge_pack(g, geom, allocate_uniform(s_w, 56, 56, 56 * 56))
    full = unpack_restore(packed, restore, 64, 56)
    chk["packed_path"] = {"packed": list(packed.shape), "coords": list(coords.shape), "restored": list(full.shape)}
    ok &= list(packed.shape) == [1, 3136, 384] and list(full.shape) == [1, 64, 56, 384]

    rep["status"] = "PASS" if ok else "FAIL"
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if a.output:
        pathlib.Path(a.output).write_text(txt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
