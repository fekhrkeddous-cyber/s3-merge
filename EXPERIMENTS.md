# S³-Merge — Per-Experiment Command Map

This document maps every result in the final manuscript and supplementary to a runnable script (or, when full reproduction requires external hardware, a command template).

All commands assume the project Docker image (`Dockerfile`) is built and the container is running with ImageNet-1K mounted at `/data/imagenet`. Downstream tasks additionally require `mmsegmentation` + ADE20K (for §8.1) or `detectron2` + COCO (for §8.2). Run from the repository root.

## Main manuscript

| Item | Script / command |
|---|---|
| **Table 1** — Design comparison (ToMe vs S³-Merge) | non-experimental (qualitative summary) |
| **Table 2** — ImageNet-1K main comparison, DeiT-S/16, 50k val, 5 seeds | `python eval_imagenet_baselines.py --mode full` |
| **Table 3** — Throughput scaling across batch sizes | `python benchmark_batch_scaling.py` |
| **Table 4** — Overhead breakdown on DeiT-S at 224² | `python profile_overhead.py` |
| **Table 5** — Window-size sweep | `python sweep_window_size.py` |
| **Table 6** — Target-selection ablation (Random, Grid, L2-Norm-high, L2-Norm-low, Attention-centrality) | `python ablation_merge_strategies.py` runs the Random and L2-Norm-high variants out of the box; the Grid (deterministic) variant is exposed via `target_policy="deterministic"` in `models/sr_tome_downstream.py`; L2-Norm-low and Attention-centrality are documented inline as variants of the same one-line code path (see *Selection-strategy notes* below) |
| **Table 7** — Factorised ablation (when/where/when-to-activate) | `python run_ablation_experiments.py` |
| **Table 8** — ADE20K UPerNet-ViT-B backbone throughput | `python eval_ade20k_upernet.py` (requires mmseg + UPerNet checkpoint) |
| **Table 9** — COCO ViTDet-B end-to-end evaluation, with Adaptive-K row | `python eval_coco_vitdet.py --adaptive_k_main` (requires detectron2 + ViTDet-B checkpoint) |
| **Table 10** — Backbone-only latency at downstream resolutions | `python model_scale_study.py --resolutions 224 384 1024` |
| **Table 11** — ViT-B/16 latency promoted from supplementary | `python model_scale_study.py --backbone vit_base_patch16_224` |
| **Table 12** — Modern ViT-L backbones (DINOv2, SigLIP) | `python model_scale_study.py --backbone {vit_large_patch16_224,dinov2_vitl14,siglip_vit_l16_256}` (DINOv2 / SigLIP runs apply a minimal monkey-patch on the HuggingFace encoder forward to insert the same merge call as the timm path) |
| **Table 13** — Cross-hardware deployment summary (H100 / L40S / Jetson Orin) | `python benchmark_cross_hardware.py` (run once per device; Jetson Orin requires JetPack and FP32 — see Cross-Hardware section below) |
| **Figure 4** — Accuracy–throughput Pareto frontier | `python generate_pareto_fixed.py` |
| **Figure 5** — Latency/throughput scaling vs batch size | `python generate_batch_scaling_pa.py` |
| **Figure 6** — CUDA kernel time breakdown (168 launches vs 10) | `python scripts/profile_kernel_breakdown.py --batch_sizes 1 32 --figure_out artifacts/kernel_profiling.pdf` |
| **Figure 7** — Amdahl decomposition (ViTDet backbone vs decoder) | `python plot_amdahl_decomposition.py` |
| **Algorithm 1** — S³-Merge pseudocode | implemented in `models/sr_tome.py::sr_tome_merge_exact` |
| **Section 5** — Safe Mode dispatch | calibration code in `benchmark_batch_scaling.py` and `eval_coco_vitdet.py` (search for `b_star`) |
| **Section 6.1** — Timing protocol | implemented in every `benchmark_*.py` script: one-shot tensor placement, 30 warmup + 200 timed passes, per-iteration `torch.cuda.synchronize()`, `time.perf_counter` |
| **Section 6.5** — Triton-fused merge kernels | `models/triton_merge.py` (three `@triton.jit` kernels: `_fused_cosine_argmax_kernel`, `_fused_weighted_merge_kernel`, `_fused_restore_index_kernel`). Block-size, tiling-strategy, and shared-memory parameters are documented as `tl.constexpr` arguments inside each kernel definition. |
| **Section 6.6** — Controlled SDPA backend experiment | `python benchmark_sdpa_control.py` |
| **Section 6.7** — Algorithmic complexity + kernel profiling | discussed in `profile_overhead.py`'s output |
| **Section 7.2** — Target-selection / deterministic Grid | `models/sr_tome.py` (Random by default) + `models/sr_tome_downstream.py` (`target_policy="deterministic"` is the Grid variant) |
| **Section 8.1** — ADE20K dense prediction | `eval_ade20k_upernet.py` |
| **Section 8.2** — COCO Amdahl-limited case study | `eval_coco_vitdet.py` + `plot_amdahl_decomposition.py` |

## Supplementary

| Item | Script / command |
|---|---|
| **Table S1** — DeiT-S/16 384² baseline latency | `python benchmark_batch_scaling.py --resolution 384` |
| **Table S2** — 384² keep-ratio sweep (K∈{40, 32, 24}) | `python sweep_window_size.py --resolution 384 --K 40 32 24` |
| **Table S3** — ViT-B/16 merge-layer sensitivity (Ls∈{6,8,10}) | `python ablation_merge_strategies.py --backbone vit_base_patch16_224 --ls 6 8 10` |
| **Table S4** — Broader window-size ablation across backbones / resolutions | `python sweep_window_size.py --backbone {deit_small,deit_iii_small,vit_base} --resolution {224,384}` |
| **Table S5** — Cross-hardware validation, original Jetson Orin observed measurement | `python benchmark_cross_hardware.py --device {h100,l40s,orin}` (requires the named device) |
| **Table S6** — Adaptive-K at higher compression (COCO) | `python eval_coco_vitdet.py --adaptive_k --keep_grid 7 7` |
| **Table S9** — Timing-variance addendum (5-run CIs on close-call cells) | `python benchmark_50k_3seed.py --seeds 5 --variance_addendum` (the script supports the 5-run variance addendum protocol) |
| **Table S10** — ADE20K PA on/off | `python eval_ade20k_upernet.py --pa_sweep` |
| **Table S11** — COCO ViTDet PA cross-task | `python eval_coco_vitdet.py --pa_sweep` |
| **Table S13** — Controlled SDPA backend experiment (full table) | `python benchmark_sdpa_control.py --full_table` |
| **Table S14** — Repeated Jetson Orin keep-ratio sweep (r∈{0.87, 0.70, 0.56, 0.43}) | `python benchmark_cross_hardware.py --device orin --repeated_sweep --r 0.87 0.70 0.56 0.43` |
| **Figure S2** — Per-image merge-cluster visualisation | external visualisation script (`exp1_visualize_merge_clusters.py` in the development repo) — output `figures/merge_cluster_panel.pdf` is included pre-built in the manuscript figures package; the captures hook directly into `models.sr_tome.sr_tome_merge_exact` and can be reproduced with that function plus `matplotlib` |
| **Figure S4** — Dense-prediction failure visualisation | external visualisation script for ADE20K predictions (input / GT / baseline / S³-Merge / disagreement) — output is included pre-built in the figures package; the captures hook into `eval_ade20k_upernet.py` |

## Selection-strategy notes (Table 6)

The five-strategy comparison in Section 7.2 of the manuscript reports:

1. **Random** — `merge_importance_based(x, r, strategy='random')` in `ablation_merge_strategies.py` (lines 67–69), and the in-window `torch.rand` selection used by `sr_tome_merge_exact` in `models/sr_tome.py` (the default in the main paper).
2. **Grid (deterministic)** — `target_policy="deterministic"` in `models/sr_tome_downstream.py` (lines 165–180); the stride-pattern is `(w/√K)`-th token in row-major order within each window.
3. **L2-Norm-high** — `merge_importance_based(x, r, strategy='norm')` sorts patches by L2 norm descending, so high-norm tokens are kept as targets (line 66–67 in `ablation_merge_strategies.py`).
4. **L2-Norm-low** — same harness as L2-Norm-high but with `sorted_idx` reversed; documented as a variant of the same one-line code path and not enabled by default since the high-norm variant was the reported best-deterministic baseline.
5. **Attention-centrality** — a centrality-based target ranking derived from a baseline attention-map pre-pass; not a default strategy and not enabled in the released harness because it requires re-running a forward pass to capture pre-merge attention.

The repository ships the first three strategies as runnable; the last two are documented inline in `ablation_merge_strategies.py` and the manuscript Section 7.2 reports the corresponding numbers from a one-off experiment.

## Adaptive-K (Table 9, Section 8.2)

The Adaptive-K mitigation replaces uniform-random within-window target selection with a variance-aware selection (centred L2 distance from the window mean). The ADE20K-fallback driver is `models/adaptive_k_patch.py` (`apply_adaptive_k_to_srtome`) which:

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

The repository at `https://anonymous.4open.science/r/s3-merge-E000` is exported as `git archive HEAD --format=zip` so that the `.git` directory is excluded.
