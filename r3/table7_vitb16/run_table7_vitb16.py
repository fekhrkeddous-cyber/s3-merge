#!/usr/bin/env python3
"""Paper Table 7, ViT-B/16 row: explicit, PA-off reconstruction of the historical configuration.

Configuration (config_table7_vitb16.yaml): ViT-B/16 at 224^2, merge after block 9 (late_start=8), T=136 retained
patch tokens (137 with [CLS]), 7x7 windows, proportional attention OFF, AMP/FP16, B in {1, 64}, 30 warm-up + 200
timed forward passes with CUDA synchronization before and after the timed loop, one invocation. Accuracy: full 50k
ImageNet-1K validation set, no shuffle, AMP/FP16, the preprocessing of the original script.

The measurement and evaluation functions follow the original model-scale script (`model_scale_study.py`) line for
line; only configuration handling, explicit PA-off assertions, logging and machine-readable output were added.
The paper's numbers come from a historical run of an earlier implementation; this script reconstructs the
configuration with the current implementation, so exact numerical identity is NOT guaranteed (latency is also
hardware- and software-dependent). See REPRODUCIBILITY.md.

Usage (from the repository root):
  python r3/table7_vitb16/run_table7_vitb16.py --output-dir results/table7_vitb16 --imagenet-cache /path/to/hf_cache
  python r3/table7_vitb16/run_table7_vitb16.py --print-config          # resolve and print the configuration only
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import timm
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.sr_tome import apply_patch, PropAttnBlock, SRToMeBlock  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_table7_vitb16.yaml"


def load_config(path):
    cfg = yaml.safe_load(open(path))
    merge = cfg.get("merge", {})
    if "prop_attn" not in merge:
        raise ValueError("config: merge.prop_attn must be set explicitly (no default is used for this row)")
    if merge["prop_attn"] is not False:
        raise ValueError("config: the Table 7 ViT-B/16 row requires merge.prop_attn: false")
    return cfg


def build_merged(cfg, device, pretrained):
    m = cfg["merge"]
    model = timm.create_model(cfg["model"], pretrained=pretrained, img_size=cfg["img_size"]).eval().to(device)
    apply_patch(model, target_tokens=m["target_tokens"], late_start=m["late_start"],
                window_size=m["window_size"], prop_attn=m["prop_attn"])
    rc = model.s3merge_config
    # explicit PA-off and geometry assertions (never rely on a default)
    assert rc["prop_attn"] is False, "proportional attention must be OFF for this configuration"
    assert not any(isinstance(b, PropAttnBlock) for b in model.blocks), "a PropAttnBlock was installed"
    assert isinstance(model.blocks[m["late_start"]], SRToMeBlock)
    exp = cfg["expected"]
    assert rc["L_s"] == exp["L_s"], rc
    assert rc["merges"] and rc["patch_tokens_out"] == exp["patch_tokens_out"], rc
    assert rc["total_tokens_out"] == exp["total_tokens_out"], rc
    return model


def check_runtime_tokens(model, cfg, device):
    """One forward pass that records the token count leaving the merge block (no silent fallback)."""
    seen = {}
    blk = model.blocks[cfg["merge"]["late_start"]]
    h = blk.register_forward_hook(lambda mod, inp, out: seen.__setitem__("tokens", out.shape[1]))  # returns None
    with torch.no_grad():
        model(torch.zeros(1, 3, cfg["img_size"], cfg["img_size"], device=device))
    h.remove()
    assert seen["tokens"] == cfg["expected"]["total_tokens_out"], seen
    assert blk.last_geometry["merges"] and blk.last_geometry["grid"] == cfg["img_size"] // 16
    return seen["tokens"], blk.last_geometry


# ---- unchanged from the original model-scale script ------------------------------------------------------------
def get_imagenet_loader(batch_size, cache_dir, num_workers, resize, crop, mean, std):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = load_dataset("ILSVRC/imagenet-1k", split='validation', cache_dir=cache_dir)

    def collate(batch):
        imgs = [transform(x['image'].convert('RGB')) for x in batch]
        labels = [x['label'] for x in batch]
        return torch.stack(imgs), torch.tensor(labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate), len(dataset)


def evaluate(model, dataloader, device, max_batches=None):
    correct = total = 0
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, (images, targets) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total * 100, total


def measure_latency(model, dummy_input, n_iter=200, warmup=30):
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(warmup):
            model(dummy_input)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(dummy_input)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000
# ----------------------------------------------------------------------------------------------------------------


def environment():
    env = {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "host": platform.node()}
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML configuration (default: %(default)s)")
    ap.add_argument("--output-dir", default="results/table7_vitb16", help="directory for JSON/CSV outputs")
    ap.add_argument("--imagenet-cache", default=os.environ.get("IMAGENET_CACHE"),
                    help="Hugging Face datasets cache directory holding ILSVRC/imagenet-1k (env IMAGENET_CACHE)")
    ap.add_argument("--skip-accuracy", action="store_true", help="timing only (no ImageNet access)")
    ap.add_argument("--max-batches", type=int, default=None, help="evaluate only this many batches (not the paper protocol)")
    ap.add_argument("--print-config", action="store_true", help="resolve and print the configuration, then exit (CPU, no weights)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # resolve the merge configuration on CPU with random weights (no download) and print it before anything runs
    probe = build_merged(cfg, "cpu", pretrained=False)
    resolved = {"config_file": os.path.relpath(args.config, ROOT) if args.config.startswith(str(ROOT)) else args.config,
                "config": cfg, "resolved_merge": probe.s3merge_config}
    print("RESOLVED CONFIGURATION\n" + json.dumps(resolved, indent=1))
    if args.print_config:
        return
    del probe
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (the protocol times AMP/FP16 on a GPU)")
    device = torch.device("cuda")
    print("NOTE: reconstruction of a historical configuration; numbers are not expected to be bit-identical to the paper.")
    t = cfg["timing"]
    inputs = {b: torch.randn(b, 3, cfg["img_size"], cfg["img_size"], device=device) for b in t["batch_sizes"]}
    loader = None
    if not args.skip_accuracy:
        a = cfg["accuracy"]
        loader, n_images = get_imagenet_loader(a["batch_size"], args.imagenet_cache, a["num_workers"], a["resize"],
                                               a["crop"], a["mean"], a["std"])
    results = []
    for arm in ("baseline", "s3merge"):
        if arm == "baseline":
            model = timm.create_model(cfg["model"], pretrained=True, img_size=cfg["img_size"]).eval().to(device)
            tok, geom = 197, None
        else:
            model = build_merged(cfg, device, pretrained=True)
            tok, geom = check_runtime_tokens(model, cfg, device)
        row = {"arm": arm, "tokens_after_merge": tok, "geometry": geom,
               "pretrained_cfg": {k: model.pretrained_cfg.get(k) for k in ("architecture", "tag", "hf_hub_id", "mean", "std")}}
        for b in t["batch_sizes"]:
            row[f"lat_ms_b{b}"] = measure_latency(model, inputs[b], t["iters"], t["warmup"])
        if loader is not None:
            row["top1"], row["n_eval"] = evaluate(model, loader, device, args.max_batches)
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "geometry"}))
        del model
        torch.cuda.empty_cache()
    base, s3 = results
    summary = {f"speedup_b{b}": base[f"lat_ms_b{b}"] / s3[f"lat_ms_b{b}"] for b in t["batch_sizes"]}
    if "top1" in base:
        summary["delta_top1"] = s3["top1"] - base["top1"]
        summary["full_validation_set"] = args.max_batches is None
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {"resolved": resolved, "environment": environment(), "results": results, "summary": summary,
               "paper_reference": {"top1_baseline": 81.08, "top1_s3merge": 80.92, "speedup_b1": 0.90, "speedup_b64": 1.02,
                                   "note": "historical run (Supplementary Section S1.2); not expected to be bit-identical"}}
    (out / "table7_vitb16.json").write_text(json.dumps(payload, indent=1))
    with open(out / "table7_vitb16.csv", "w", newline="") as f:
        cols = ["arm", "tokens_after_merge"] + [f"lat_ms_b{b}" for b in t["batch_sizes"]] + ["top1", "n_eval"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(results)
    print("SUMMARY " + json.dumps(summary))
    print(f"wrote {out / 'table7_vitb16.json'} and {out / 'table7_vitb16.csv'}")


if __name__ == "__main__":
    main()
