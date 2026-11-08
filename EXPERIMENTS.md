# Experiments: commands for the paper results included in this release

Numbering follows the final manuscript:
- main paper: Tables 1–9, Figures 1–7;
- supplement: Tables S1–S24, Figures S1–S4.

**Scope of this file.**
- It documents only commands whose script exists, whose flags exist in its parser, and that were exercised by the release tests (`--help` and smoke runs).
- Results without a command here are listed in the status table at the end and in `PAPER_CODE_MATRIX.md`.
- Run every command from the repository root.

**Reading the results.**
- Each campaign has its own protocol, and values from different campaigns must not be combined (see `README.md`).
- Latency depends on hardware, driver and load, so it is not expected to be reproduced bit-exactly.
- `--smoke` options exist only for software testing. They are documented in each sub-directory's README and are not paper measurements.

---

## 1. Table 1, Fig. 5, Tables S20–S22: unified comparison with official ToMe

Code: `r3/unified_tome/` (protocol details and output schema in `r3/unified_tome/README.md`).

| Item | Value |
|---|---|
| Model / checkpoint | DeiT-S/16, timm `deit_small_patch16_224.fb_in1k` (`--checkpoint` for a local file; file and state sha256 are checked) |
| Arms | baseline; official ToMe r=10 and r=12 (commit `af95e4b1…`, `--tome-path`, source hashes checked); S³-Merge-Acc (`late_start=8`, i.e. L_s=9; T=84; w=7; **PA on**); S³-Merge-Speed (`late_start=6`, i.e. L_s=7; T=84; w=7; **PA on**) |
| Data | ImageNet-1K validation, Hugging Face `ILSVRC/imagenet-1k` arrow shards (`--data-dir`), 50,000 images in index order (accuracy only) |
| Hardware / precision | NVIDIA H100 NVL (reference); FP32, no autocast, no `torch.compile`, TF32 matmul off |
| Batch sizes | B = 1, 8, 32 (timing); 64 (accuracy) |
| Warm-up / timed | 30 warm-up forwards (synchronized), then 200 timed forwards with synchronization around every forward |
| Invocations | 10 independent latency invocations in randomized arm order (plus 3 separate profiling invocations) |
| Aggregation | latency = mean of the 10 invocation means (95% t-CI); B=32 throughput = geometric mean over invocations; speed-ups paired within invocation (geometric mean, t-CI and bootstrap CI) |
| Accuracy | S³-Merge: 5 seeds (mean ± SD); baseline and ToMe: seed 0 (ToMe with a determinism replicate) |

```bash
python r3/unified_tome/accuracy_campaign.py --output-dir out/table1_accuracy --data-dir /path/to/hf_datasets_cache --tome-path /path/to/ToMe
python r3/unified_tome/accuracy_analysis.py --runs-dir out/table1_accuracy/runs --output-dir out/table1_accuracy/analysis
python r3/unified_tome/timing_campaign.py --output-dir out/table1_timing --tome-path /path/to/ToMe --accuracy-analysis out/table1_accuracy/analysis/accuracy_analysis.json
python r3/unified_tome/gmac.py --output-dir out/table1_gmac
```

## 2. Table 4, Tables S17–S19: controlled 2×2 factorial study

Code: `r3/factorial/` (details in `r3/factorial/README.md`). The arm implementation `factorial_merge.py` is the validated one; only comments differ.

| Item | Value |
|---|---|
| Model / checkpoint | DeiT-S/16, timm `deit_small_patch16_224.fb_in1k` |
| Arms | baseline; SW, SG, PW, PG (single-shot / progressive × windowed / global); all keep 84 patch tokens; **PA off** in every arm; Core anchor run `core` (S³-Merge, `late_start=8`, T=84, w=7, PA off) |
| Data | ImageNet-1K validation arrow shards (`--data-dir`), 50,000 images, fixed order, batch 64, 8 workers |
| Hardware / precision | NVIDIA H100 NVL (reference); FP32, TF32 matmul off |
| Seeds | 0–4 for SW/SG/PW/PG; baseline and `core` at seed 0 (22 runs) |
| Timing | B = 1, 8, 32; 30 warm-up + 200 timed forwards with synchronization around every forward; **10** latency invocations in randomized arm order (plus profiling and merge-micro invocations, not used by Table 4) |
| Aggregation | Top-1: five-seed mean ± SD and paired contrasts (95% t-CI, ±0.10-point margin); latency: mean over the 10 paired invocations, paired ratios with 95% CIs (±1% margin) |

```bash
python r3/factorial/campaign.py accuracy --output-dir runs/r3_factorial/accuracy -- --data-dir /path/to/imagenet_shards --device cuda
python r3/factorial/analyze_accuracy.py --runs-dir runs/r3_factorial/accuracy --output-dir runs/r3_factorial/analysis
python r3/factorial/campaign.py timing --modes latency --output-dir runs/r3_factorial/timing -- --device cuda
python r3/factorial/analyze_timing.py --raw-dir runs/r3_factorial/timing --output-dir runs/r3_factorial/analysis
python r3/factorial/test_factorial.py
```

## 3. Table 7, ViT-B/16 row: explicit PA-off configuration

Code: `r3/table7_vitb16/run_table7_vitb16.py` with `config_table7_vitb16.yaml`.

| Item | Value |
|---|---|
| Model / checkpoint | timm `vit_base_patch16_224`, pretrained (timm default weights; the resolved pretrained tag is written to the output) |
| Configuration | `late_start=8` (L_s=9), T=136, w=7, so **136 patch tokens and 137 with [CLS]** |
| PA | **off**: a required key in the config file; asserted; no `PropAttnBlock` installed |
| Data | ImageNet-1K validation through `datasets.load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=--imagenet-cache)`; Resize 256 → CenterCrop 224 → ImageNet mean/std; batch 64; no shuffle |
| Hardware / precision | NVIDIA H100 NVL (reference); **AMP/FP16** (`torch.cuda.amp.autocast`) for timing and accuracy |
| Batch sizes | B = 1 and B = 64 |
| Warm-up / timed | 30 warm-up + 200 timed forwards; `torch.cuda.synchronize()` before and after the timed loop only |
| Invocations / aggregation | one invocation; latency = mean per forward over the timed loop; speed-up = baseline / merged latency |

```bash
python r3/table7_vitb16/run_table7_vitb16.py --print-config
python r3/table7_vitb16/run_table7_vitb16.py --output-dir results/table7_vitb16 --imagenet-cache /path/to/hf_cache
```

The paper's row comes from a historical run with an earlier implementation (Supplementary Section S1.2). This script reconstructs the configuration, so exact numerical identity is **not** guaranteed. Its measurements are a different campaign from Supplementary Table S2 and must not be combined with it.

## 4. Section 5: Safe Mode calibration (DeiT-S/16, 224²)

| Item | Value |
|---|---|
| Model / checkpoint | DeiT-S/16, timm pretrained weights; S³-Merge `late_start=8` (L_s=9), T=84, w=7 |
| PA / precision | PA off (`SM_PA=0`, default); FP32 (`SM_FP32=1`, default) |
| Batch sizes | B ∈ {1, 2, 4, 8, 16, 32} |
| Protocol | per B, interleaved baseline/merged blocks of 10 warm-up + 50 timed iterations with per-iteration synchronization; percentile bootstrap (2,000 resamples) of the speed-up |
| Rule / output | activate at B when LCB95 > 1.02; B* = smallest qualifying B; result cached as JSON (`models/safemode_cache.json`) |
| Invocations | one cold calibration per process |

```bash
python -m models.safemode --validate
```

The calibrated B* is device- and configuration-specific. This repository does not include a dispatcher, and the paper reports no routed latency.

## 5. Tests

```bash
python tests/test_release.py
python tests/test_public.py
python tests/smoke_gpu.py
```

---

## Status of the other results

| Item | Status in this release |
|---|---|
| Table 2 (overhead decomposition, FP32, T=84, +PA) | producer not included (`profile_overhead.py` runs a different configuration: FP16 autocast, T=0.7N) |
| Table 3 (target selection, 30 seeds) | partial: `run_ablation_experiments.py` implements the five strategies; the 30-seed protocol and the selection-cost timing are not wired. No command documented |
| Table 5, Table S5 (COCO ViTDet-B, packed path) | merge core included (`models/adaptive_k.py`); detection drivers **not included** |
| Table 6, Table S9 (plain ViT-B/16 proxy) | not included |
| Table 8 (ViT-L, DINOv2-L, SigLIP-L) | not included (Hugging Face adapter and drivers) |
| Table 9, Tables S4, S10, S14 (cross-hardware: L40S, Jetson Orin) | not included. `benchmark_cross_hardware.py` is a simplified legacy check, not the paper protocol |
| Table S15 (LW-DETR-medium) | not included (wrapper around the official LW-DETR code) |
| Tables S1, S7 (384²) | producers not included. The release module merges 384² inputs (24×24 grid, w=8), and this is tested |
| Tables S2, S3, S12 (window and merge-layer sweeps) | producers not included or not re-verified (`sweep_window_size.py` is a legacy script) |
| Table S8 (PA on COCO, superseded implementation) | historical (`eval_coco_vitdet.py`, regular-grid path) |
| Table S11, Fig. 6 (separate batch-scaling campaign) | not included (`benchmark_batch_scaling.py` is a single-invocation legacy script) |
| Table S16 (Triton prototype) | kernel included (`models/triton_merge.py`); end-to-end harness not included |
| Table S23 (PiToMe, ATC; original campaign) | `eval_imagenet_baselines.py` is the original-campaign harness. The PiToMe compatibility adapter is **not included**, so the PiToMe arm stops with an explicit error |
| Table S24 (merge depth and PA, single runs) | `apply_patch` supports every variant (`keep_all_control=True` for the w=14 control); no driver |
| Fig. 3 (merge displacement) | `analyze_merge_displacement.py` (legacy script; protocol not re-verified for this release) |
| Fig. 7 (CUDA kernel breakdown) | `scripts/profile_kernel_breakdown.py` is a later port of the original producer; not re-verified for this release |
| Figs. 1, 2, 4, S1–S4 | diagrams or visualisations; producers not included |

**Legacy scripts.** These correspond to no result of the current paper and are kept for reference only:
- `eval_ade20k_upernet.py` and `configs/mmseg/`;
- `benchmark_sdpa_control.py`;
- `plot_amdahl_decomposition.py`;
- `generate_pareto_fixed.py` and `generate_batch_scaling_pa.py` (historical data);
- `ablation_merge_strategies.py` and `ablation_tome_components.py`;
- `models/adaptive_k_patch.py`;
- `baselines/dsm_impl.py`.
