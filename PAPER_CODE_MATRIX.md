# Paper–code matrix

Every table and figure of the final paper and supplement that reports an experiment is listed below.

**Public status:**
- **included**: an adapted reproduction harness for the published result is in this repository;
- **reconstruction**: the configuration is rebuilt with the current implementation;
- **partial**: only part of the implementation is included (e.g. the S³-Merge core, or a legacy script covering part of the protocol);
- **not included**: no implementation in this release;
- **analytical**: no experiment code is involved.

**Test status** refers to the software tests run for this release: unit tests on synthetic tensors, CPU smoke runs, reduced GPU functional smoke runs, and static checks. **No paper number was re-measured**, and no implementation here is claimed to reproduce a historical measurement bit-exactly. `REPRODUCIBILITY.md` §5a separates the adapted harnesses from the historical scripts.

## Main paper

| Paper item | Script | Config | Public status | Test status | Known limitation |
|---|---|---|---|---|---|
| Table 1 (unified comparison with official ToMe) | `r3/unified_tome/` (`accuracy_campaign.py`, `accuracy_analysis.py`, `timing_campaign.py`, `gmac.py`) | `r3/unified_tome/config_table1.yaml` (mirrors the CLI defaults) | **included** (adapted harness; see `PROVENANCE.md`) | `--help`; CPU smoke of the campaigns and analyses; reduced GPU smoke (timing and accuracy entry points); regenerated inputs and arm orders equal the paper's; GMAC and token trajectories equal Table S20 | Needs ImageNet-1K, official ToMe `af95e4b1`, an H100-class GPU; latency not bit-exact. **The sixth, non-Table-1 configuration of the paper campaign (factorial SW arm) is not timed**, so invocations run five arms in the paper's relative order |
| Fig. 5 | as Table 1 (timing analysis) | as Table 1 | **included** | as Table 1 | as Table 1 |
| Table 2 (overhead decomposition, FP32, T=84, +PA) | — (`profile_overhead.py` is a legacy script with a different configuration) | — | **not included** | — | The legacy script uses FP16 autocast and T=0.7N |
| Table 3 (per-window target selection, 30 seeds) | `run_ablation_experiments.py` (legacy) | — | **partial** | static checks only | The five strategies are implemented; the 30-seed protocol and the selection-cost timing are not wired |
| Table 4 (2×2 factorial: schedule × scope) | `r3/factorial/` (`campaign.py`, `analyze_accuracy.py`, `analyze_timing.py`) | `r3/factorial/config_table4.yaml`; all arms PA off; 84 patch tokens | **included** (adapted harness; arm code AST-identical to the validated implementation) | 14 unit tests; CPU smoke; reduced GPU smoke (timing and accuracy entry points); the analysis re-applied to the paper's raw outputs matches the original analysis exactly and the published Table S17 counts | Needs ImageNet-1K and an H100-class GPU; latency not bit-exact. The analysis labels the accuracy interaction EQUIVALENT-WITHIN-MARGIN, where the paper says "not statistically resolved" (same interval) |
| Table 5 (COCO ViTDet-B, packed exact-coordinate path) | `models/adaptive_k.py` (merge, packing, restore, packed global block) | — | **partial**: S³-Merge core only; detection drivers **not included** | packed-path unit tests (including a detectron2 global block with relative positions); GPU smoke of packing and restore | COCO results cannot be reproduced from this repository |
| Table 6 (plain ViT-B/16 proxy, 512² / 1024²) | — | — | **not included** | — | random-weight proxy driver not released |
| Table 7, DeiT-S row | values from the unified campaign (`r3/unified_tome/`) | as Table 1 | **included** (via Table 1) | as Table 1 | large-batch cell is B=32 |
| Table 7, ViT-B/16 row | `r3/table7_vitb16/run_table7_vitb16.py` | `config_table7_vitb16.yaml`: L_s=9, T=136 (137 with [CLS]), w=7, **PA off**, AMP/FP16, B ∈ {1, 64}, 30+200, sync around the loop | **reconstruction** | config unit tests; GPU smoke of the configuration; reduced GPU run of the runner (pretrained weights, timing only, reduced iterations) | The historical run used an earlier implementation; exact identity is not guaranteed. A different campaign from Table S2 |
| Table 7, proxy rows | — | — | **not included** | — | as Table 6 |
| Table 8 (ViT-L, DINOv2-L, SigLIP-L) | — | — | **not included** | — | Hugging Face adapter and drivers not released |
| Table 9 (cross-hardware: H100 / L40S / Jetson Orin) | — (`benchmark_cross_hardware.py` is a simplified legacy check) | — | **not included** | — | The device-campaign runners are not released |
| Fig. 3 (merge displacement) | `analyze_merge_displacement.py` (legacy) | — | **partial** | static checks only | protocol not re-verified for this release |
| Fig. 6 (separate batch-scaling campaign) | — (`benchmark_batch_scaling.py` is a single-invocation legacy script) | — | **not included** | — | The five-invocation campaign producer is not released |
| Fig. 7 (CUDA kernel breakdown) | `scripts/profile_kernel_breakdown.py` (later port of the original producer) | — | **partial** | static checks only | not re-verified for this release |
| Section 5 (Safe Mode calibration) | `models/safemode.py` | `python -m models.safemode --validate` | **included** (calibrator) | import and `--help` | No dispatcher; B* is device-specific; the paper reports no routed latency |

## Supplement

| Paper item | Script | Config | Public status | Test status | Known limitation |
|---|---|---|---|---|---|
| S1, S7 (384² sweep; timing-variance addendum) | `models/sr_tome.py` (24×24 grid, w=8) | `apply_patch(..., window_size=8)` | **partial** (core only) | 384² merge tested on CPU and GPU; bit-identical to the R3 module | Sweep and variance drivers not released |
| S2 (ViT-B/16 merge-layer sweep) | — | — | **not included** | — | — |
| S3, S12 (window-size sweeps) | `sweep_window_size.py` (legacy) | — | **partial** | static checks only | The legacy script uses autocast; the keep-all controls now require `keep_all_control=True` |
| S4, S10, S14 (cross-hardware; Jetson Orin sweep; **L40S**) | — | — | **not included** | — | The L40S runner depends on a host-specific payload; Orin harness not released |
| S5 (complete COCO metrics) | as Table 5 | — | **partial** (core only) | as Table 5 | as Table 5 |
| S6, S13 (complexity; design comparison) | — | — | **analytical** | — | — |
| S8 (PA on COCO, superseded implementation) | `eval_coco_vitdet.py` (legacy, regular-grid path) | — | **partial** (historical) | static checks only | superseded implementation; not used for current main-paper numbers |
| S9 (Safe Mode B* on the ViT-B/16 proxy) | — | — | **not included** | — | as Table 6 |
| S11 (throughput across batch sizes) | as Fig. 6 | — | **not included** | — | as Fig. 6 |
| **S15 (LW-DETR-medium)** | — | — | **not included** | native 10×10 window geometry unit-tested | The wrapper around the official LW-DETR code is not released |
| S16 (Triton prototype) | `models/triton_merge.py` (kernel) | — | **partial** | import | end-to-end harness not released |
| S17–S19 (factorial: per seed, contrasts, latency) | as Table 4 | as Table 4 | **included** | as Table 4 | as Table 4 |
| S20–S22 (unified comparison: configurations, invocations, ratios) | as Table 1 | as Table 1 | **included** | as Table 1 | as Table 1 |
| S23 (PiToMe and ATC, original campaign) | `eval_imagenet_baselines.py` (legacy original-campaign harness) | — | **partial**: ATC arm via an external clone; **PiToMe adapter not included** | the PiToMe arm stops with an explicit error | The archived local PiToMe adapter is not distributed (see `REPRODUCIBILITY.md`) |
| S24 (merge depth and PA, single runs) | `models/sr_tome.py` (`apply_patch` covers every variant) | `keep_all_control=True` for the w=14 control | **partial** | unit tests of the variants | no driver |
| Figs. 1, 2, 4, S1–S4 | — | — | **not included** (illustrations) | — | — |
