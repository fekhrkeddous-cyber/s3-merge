# S³-Merge — Per-Experiment Command Map

This document maps every result in the final manuscript and supplementary to a runnable script (or, when full reproduction requires external hardware, a command template).

All commands assume the project Docker image (`Dockerfile`) is built and the container is running with ImageNet-1K mounted at `/data/imagenet`. Downstream tasks additionally require `mmsegmentation` + ADE20K (for §8.1) or `detectron2` + COCO (for §8.2). Run from the repository root.

## Main manuscript

| Item | Script / command |
|---|---|
| **Table 1** — ImageNet-1K main comparison, DeiT-S/16, 50k val, 5 seeds | `python eval_imagenet_baselines.py --seeds 5` |
| **Table 2** — Overhead breakdown on DeiT-S at 224² | published Table 2 is the FP32, T=84 (+PA) decomposition reporting the direct total and the component subtotal separately; `python profile_overhead.py` reproduces the historical FP16 r≈0.7 breakdown unless reconfigured to the main configuration |
| **Table 3** — Target-selection ablation (Random 30 seeds, Norm-high, Grid, Norm-low, Attention) | `python run_ablation_experiments.py` (PART 1; the paper's reconstruction uses 30 random seeds and the selection-cost microbenchmark protocol) |
| **Table 4** — Factorised ablation (when/where/when-to-activate) | `python run_ablation_experiments.py` |
| **Table 5** — ADE20K UPerNet-ViT-B | `python eval_ade20k_upernet.py --run_full_eval` (requires mmseg + UPerNet checkpoint) |
| **Table 6** — COCO ViTDet-B end-to-end evaluation incl. Adaptive-K | `python eval_coco_vitdet.py --run_full_eval` (requires detectron2 + ViTDet-B checkpoint; the Adaptive-K allocation/packed core is `models/adaptive_k.py`) |
| **Table 7** — Backbone-only latency at downstream resolutions | `python model_scale_study.py` |
| **Table 8** — Model-scale study (DeiT-S / ViT-B across resolutions) | `python model_scale_study.py` |
| **Table 9** — Modern ViT-L backbones (DINOv2, SigLIP) | `python model_scale_study.py` (DINOv2 / SigLIP runs apply a minimal monkey-patch on the HuggingFace encoder forward to insert the same merge call as the timm path) |
| **Table 10** — Cross-hardware deployment summary (H100 / L40S / Jetson Orin) | `python benchmark_cross_hardware.py` (run once per device) |
| **Figure 4** — Merge-cluster locality excerpt | pre-built panel (see Figure S2 note below) |
| **Figure 5** — Accuracy–throughput Pareto frontier | `python generate_pareto_fixed.py` |
| **Figure 6** — Latency/throughput scaling vs batch size | published Figure 6 plots the unified FP32 campaign (B ∈ [1, 64]); `python generate_batch_scaling_pa.py` reproduces the historical PA-campaign figure from its embedded data |
| **Figure 7** — CUDA kernel time breakdown (168 vs 10 Scatter/Merge-category launches; bar totals 612/197/162 at B=1) | `python scripts/profile_kernel_breakdown.py --batch_sizes 1 32 --figure_out artifacts/kernel_profiling.pdf` |
| **Section 8.2** — Amdahl decoder-ceiling analysis | the paper's former absolute-bar figure was removed; `python plot_amdahl_decomposition.py` reproduces the historical illustration only |
| **Algorithm 1** — S³-Merge pseudocode (Static-K Core) | implemented in `models/sr_tome.py::sr_tome_merge_exact`; Adaptive-K extension in `models/adaptive_k.py` |
| **Section 5** — Safe Mode | `python models/safemode.py --validate` (bootstrap-LCB rule, cached B*, baseline fallback); crossover sweeps also in `benchmark_batch_scaling.py` and `eval_coco_vitdet.py --calibration` |
| **Section 6.1** — Timing protocol | paper protocol: 30 warm-up + 200 timed passes, per-iteration `torch.cuda.synchronize()`, five independent invocations, paired-bootstrap CIs. Scripts using 30+200 directly: `benchmark_batch_scaling.py`, `model_scale_study.py`, `sweep_window_size.py`, `profile_overhead.py`; other scripts state their own iteration counts in their headers. |
| **Section 6.5** — Triton-fused merge kernels | `models/triton_merge.py` |
| **Section 6.6** — Controlled SDPA backend experiment | `python benchmark_sdpa_control.py` |
| **Section 7.2** — Target-selection policies | `run_ablation_experiments.py` (five strategies); deterministic variance-aware Static-K = `models/adaptive_k.py` with `allocate_uniform` and `sel_scores=None`; grid policy = `target_policy="deterministic"` in `models/sr_tome_downstream.py` |
| **Section 8.2** — COCO Amdahl-limited case study + decoder-light contrast | `eval_coco_vitdet.py` + `plot_amdahl_decomposition.py`; the LW-DETR decoder-light confirmation is a wrapper-level integration of the official Atten4Vis/LW-DETR release (external to this repo) using the same `models/adaptive_k.py` merge semantics with uniform per-window K |

## Supplementary

| Item | Script / command |
|---|---|
| **Table S1** — DeiT-S/16 384² baseline latency | `python benchmark_batch_scaling.py --img_size 384` |
| **Table S2** — 384² keep-ratio sweep | `python sweep_window_size.py --target_tokens 360` (repeat with 288, 216; latency at 384² via `benchmark_batch_scaling.py --img_size 384`) |
| **Table S3** — ViT-B/16 merge-layer sensitivity | `python ablation_merge_strategies.py --model vit_base_patch16_224` |
| **Table S4** — Broader window-size ablation | `python sweep_window_size.py` (per backbone/resolution) |
| **Table S5** — Cross-hardware validation | `python benchmark_cross_hardware.py` on each device |
| **Table S6** — Complete COCO metrics under the corrected packed implementation | `python eval_coco_vitdet.py --run_full_eval` (+ `models/adaptive_k.py`) |
| **Table S9** — Timing-variance addendum (5-run CIs) | `python benchmark_50k_3seed.py --seeds 5` |
| **Tables S10/S11** — PA on/off (ADE20K / COCO) | `python eval_ade20k_upernet.py --run_full_eval` / `python eval_coco_vitdet.py --run_full_eval` with `prop_attn` toggled |
| **Table S13** — Controlled SDPA backend experiment | `python benchmark_sdpa_control.py` |
| **Table S14** — Repeated Jetson Orin keep-ratio sweep | `python benchmark_cross_hardware.py` on the Orin (per keep ratio) |
| **Table S18** — Reproduced L40S measurements | `python benchmark_cross_hardware.py` on an L40S under the paper-matched environment (torch 2.1.2+cu121, timm 1.0.24, FP32; five process invocations) |
| **Table S19** — LW-DETR decoder-light confirmation | wrapper-level integration of the official `Atten4Vis/LW-DETR` release (commit d5e6e6c4, official `LWDETR_medium_60e_coco.pth`); merge core semantics identical to `models/adaptive_k.py` with uniform per-window K=75/100 inserted after encoder block 6 |
| **Figure S2** — Per-image merge-cluster visualisation | external visualization script, not included in this release — output `figures/merge_cluster_panel.pdf` is included pre-built in the manuscript figures package; the captures hook directly into `models.sr_tome.sr_tome_merge_exact` and can be reproduced with that function plus `matplotlib` |
| **Figure S4** — Dense-prediction failure visualisation | external visualization script for ADE20K predictions — output is included pre-built in the figures package; the captures hook into `eval_ade20k_upernet.py` |

## Selection-strategy notes (Table 6)

The five-strategy comparison in Section 7.2 of the manuscript reports:

1. **Random** — `merge_importance_based(x, r, strategy='random')` in `ablation_merge_strategies.py` (lines 67–69), and the in-window `torch.rand` selection used by `sr_tome_merge_exact` in `models/sr_tome.py` (the default in the main paper).
2. **Grid (deterministic)** — `target_policy="deterministic"` in `models/sr_tome_downstream.py` (lines 165–180); the stride-pattern is `(w/√K)`-th token in row-major order within each window.
3. **L2-Norm-high** — `merge_importance_based(x, r, strategy='norm')` sorts patches by L2 norm descending, so high-norm tokens are kept as targets (line 66–67 in `ablation_merge_strategies.py`).
4. **L2-Norm-low** — same harness as L2-Norm-high but with `sorted_idx` reversed; documented as a variant of the same one-line code path and not enabled by default since the high-norm variant was the reported best-deterministic baseline.
5. **Attention-centrality** — a centrality-based target ranking derived from a baseline attention-map pre-pass; not a default strategy and not enabled in the released harness because it requires re-running a forward pass to capture pre-merge attention.

The repository ships the first three strategies as runnable; the last two are documented inline in `ablation_merge_strategies.py` and the manuscript Section 7.2 reports the corresponding numbers from a one-off experiment.

## Adaptive-K (Table 9, Section 8.2)

The Adaptive-K mitigation replaces uniform-random within-window target selection with a variance-aware selection (centred L2 distance from the window mean). The ADE20K-fallback driver is `models/adaptive_k_patch.py` (`models/adaptive_k.py` (`adaptive_merge_pack`, `allocate_slot_auction`)) which:

- computes per-window feature variance at the merge layer,
- buckets windows into a high-variance group (K_high targets) and a low-variance group (K_low targets) with `mean(K) == K_static`,
- runs the standard merge pipeline.

For COCO ViTDet, the same idea is exercised in-place via a centred-L2 target-selection swap (no separate file); use `--adaptive_k` on `eval_coco_vitdet.py`.

The higher-compression check (Supplementary Table S6) is exercised with `--keep_grid 7 7`.

## Cross-hardware (H100 / L40S / Jetson Orin)

The cross-hardware harness `benchmark_cross_hardware.py` is self-contained (no project imports) and works on any device. The Jetson Orin sweep requires:

- JetPack 5.x with PyTorch 2.x prebuilt for arm64;
- FP32 inference precision (JetPack default);
- mode `30W` on the device (`sudo nvpmodel -m 0` followed by `sudo jetson_clocks`);
- the device's per-batch latency to be stable across the 30 warmup + 200 timed measurement protocol of Section 6.1.

The numbers in Table 13 and Supplementary Tables S5 / S14 come from running this script once per device.

## Timing protocol

All datacenter-GPU latency measurements follow the timing protocol of Section 6.1: 30 warmup forward passes with `torch.cuda.synchronize()` per pass, then 200 timed passes wall-clocked with `time.perf_counter` and synchronised per iteration, mean per-batch latency reported. The protocol is implemented identically in every `benchmark_*.py` script in the repository. DeiT-S/16 latency rows run in FP32; ViT-B/16 rows run under AMP/FP16.

## Seeds

Stochastic methods use seeds `{0, 1, 2, 3, 4}` (5-seed default) via `torch.manual_seed(seed)` and `np.random.seed(seed)`. The 30-seed close-call selection comparison (Table 6) extends the loop to `range(30)` in the same harness.

## Anonymous-release export

The public repository is `https://github.com/fekhrkeddous-cyber/s3-merge`.
