# Reproducibility

This file describes:
- what the repository can reproduce;
- under which environment and protocol;
- what is not included.

The numbers reported in the paper are authoritative. The code here is the current implementation used to re-run the included campaigns. It does **not** guarantee bit-exact latency reproduction: latency depends on the GPU, driver, clocks, power cap, concurrent load and software versions.

## 1. Environment

| Component | Paper (H100 experiments) | Pinned here |
|---|---|---|
| Python | 3.10.13 | `requirements.txt` |
| PyTorch / torchvision | 2.1.2 (CUDA 12.1, cuDNN 8.9.2) / 0.16.2 | `torch==2.1.2`, `torchvision==0.16.2` |
| timm | 1.0.24 | `timm==1.0.24` |
| numpy / datasets / safetensors | 1.26.4 / 4.5.0 / 0.7.0 | pinned |
| Official ToMe (Table 1 only) | commit `af95e4b1befa172dadccd8c81e223b10090f9579` | installed separately (licence CC BY-NC 4.0; not redistributed) |
| detectron2 (packed ViTDet path only) | commit `fd27788985af0f4ca800bca563acdb700bb890e2` | installed separately |
| GPU (reference) | NVIDIA H100 NVL, default boost clocks, 400 W power cap, clocks not locked | — |

The `Dockerfile` builds this environment (`ENV PYTHONPATH=/workspace`). Jetson Orin runs in the paper used JetPack 6 (L4T R36.4), PyTorch 2.5, CUDA 12.6 and timm 0.9.12, in the 30 W mode with unlocked clocks. No Orin or L40S runner is included in this release.

## 2. Data and checkpoints

- **ImageNet-1K validation:** Hugging Face `ILSVRC/imagenet-1k` (gated; accept its terms), split `validation`, 50,000 images in index order, no shuffling.
  - `r3/unified_tome` and `r3/factorial` read the cached arrow shards (`--data-dir`) in shard order and refuse to run unless they contain the 50,000 validation images.
  - `r3/table7_vitb16` uses `datasets.load_dataset(..., cache_dir=--imagenet-cache)`.
  - The paper used dataset revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`.
- **DeiT-S/16:** timm `deit_small_patch16_224.fb_in1k`, checkpoint file sha256 `1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b`. Both harnesses check the file hash and the hash of the loaded weights (`--expected-sha256`, `--expected-state-sha256`).
- **ViT-B/16 (Table 7):** timm `vit_base_patch16_224` with timm's default pretrained weights. The resolved pretrained configuration is written to the output. The original run used the same model name; its weight file hash was not recorded.
- **Synthetic timing inputs:** `torch.randn` from fixed seeds. The harnesses record each tensor's sha256 and compare it with the paper's inputs.

## 3. Campaigns, settings and output schema

Values from different campaigns must **not** be combined, even for the same configuration.

| Campaign (paper items) | Script | Model, resolution | Hardware | Precision | PA | Batch sizes | Warm-up / timed | Synchronization scope | Invocations | Aggregation | Outputs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Unified official-ToMe comparison (Table 1, Fig. 5, S20–S22) | `r3/unified_tome/` | DeiT-S/16, 224² | H100 NVL | FP32 | on for S³-Merge-Acc and -Speed | 1, 8, 32 (timing); 64 (accuracy) | 30 / 200 | around every timed forward | 10 (randomized arm order) + 3 profiling | mean of invocation means (t-CI); geometric-mean throughput; paired speed-ups (t and bootstrap CIs) | per-invocation JSON; analysis JSON/CSV/Markdown (see `r3/unified_tome/README.md`) |
| Accuracy, unified comparison | `r3/unified_tome/` | same | any CUDA GPU | FP32 | as above | 64 | — | — | S³-Merge: 5 seeds; baseline and ToMe: seed 0 | mean ± SD over seeds | per-run JSON with correct counts; analysis JSON/CSV |
| Factorial study (Table 4, S17–S19) | `r3/factorial/` | DeiT-S/16, 224² | H100 NVL | FP32 | off in every arm | 1, 8, 32 (timing); 64 (accuracy) | 30 / 200 | around every timed forward | 10 latency invocations (randomized order); 5 seeds for accuracy | five-seed mean ± SD; paired contrasts with prespecified margins (±0.10 points, ±1% latency) | per-run and per-invocation JSON; analysis CSV/JSON/Markdown (see `r3/factorial/README.md`) |
| Table 7, ViT-B/16 row (historical; Supplementary S1.2) | `r3/table7_vitb16/` | ViT-B/16, 224², 136 patch tokens (137 with [CLS]), L_s=9 | H100 NVL | AMP/FP16 | **off** (asserted) | 1, 64 | 30 / 200 | before and after the timed loop only | 1 | mean per forward; speed-up = baseline / merged | `table7_vitb16.json` (resolved config, environment, per-arm latency and Top-1, summary) and `.csv` |
| Safe Mode calibration (Section 5) | `models/safemode.py` | DeiT-S/16, 224² | per device | FP32 | off | 1, 2, 4, 8, 16, 32 | 10 / 50 per block, interleaved | every iteration | 1 cold calibration per process | percentile-bootstrap LCB95 of the speed-up; activate if > 1.02 | `models/safemode_cache.json` |
| Separate batch-scaling campaign (Fig. 6, S11) | not included | — | — | FP32 | — | 1–64 | — | — | 5 | — | — |
| Downstream (Table 5, S5, S15) | not included (merge core only) | — | — | FP32 | off | — | — | — | 5 | — | — |
| Cross-hardware (Table 9, S4, S10, S14) | not included | — | — | — | — | — | — | — | — | — | — |

## 4. Paper item → script

`PAPER_CODE_MATRIX.md` lists every main item with its script, configuration, public status, test status and known limitations. `EXPERIMENTS.md` gives the exact commands.

## 5. Current implementation and historical results

- **The S³-Merge module (`models/sr_tome.py`)** reads the patch grid from the model, or from the token count, and validates it. It raises an error for invalid geometries instead of skipping the merge. Its merge computation is unchanged: on synthetic inputs it is bit-identical to the module used for the R3 experiments. That covers merge outputs and token sizes at 14×14 and 24×24, and full DeiT-S logits for the Acc, Speed and PA-off configurations. The extra per-forward Python work is a cached dictionary lookup. Two CPU measurements of the merge-block wrapper gave +1 to +2 µs per call (canonical about 145 µs), which is within run-to-run noise.
- **The Table 7 ViT-B/16 row** was produced by an earlier implementation and a single historical run. `r3/table7_vitb16` reconstructs its configuration with the current implementation, so exact numerical identity is not guaranteed.
- **The Table 1 and Table 4 harnesses** are cleaned ports of the validated scripts. The protocol is unchanged; the deviations, such as removed host-specific checks and renamed outputs, are listed in each `PROVENANCE.md`. The Table 1 port no longer times a sixth configuration that the paper campaign interleaved, but it keeps the paper's relative arm order.
- **Reported numbers are authoritative.** A re-run yields new measurements; it neither replaces nor corrects the reported values.

## 5a. Historical scripts and adapted reproduction harnesses

The repository contains two kinds of experiment code. Neither reproduces the paper's historical measurements bit-exactly.

| Kind | Files | What they are |
|---|---|---|
| **Adapted reproduction harnesses** (R3) | `r3/unified_tome/`, `r3/factorial/` | Cleaned ports of the validated scripts that produced Table 1 / Fig. 5 / S20–S22 and Table 4 / S17–S19. The protocol is unchanged. Host-specific checks are removed, paths are arguments, and outputs are renamed. File-level mapping and deviations are in each `PROVENANCE.md` |
| **Configuration reconstruction** | `r3/table7_vitb16/` | A new runner that rebuilds the configuration of the historical Table 7 ViT-B/16 run with the current implementation. Its measurement and evaluation functions are copied from the historical script |
| **Historical (legacy) scripts** | repository root (`model_scale_study.py`, `eval_imagenet_baselines.py`, `eval_coco_vitdet.py`, `benchmark_*.py`, `sweep_window_size.py`, `run_ablation_experiments.py`, `analyze_merge_displacement.py`, `profile_overhead.py`, `failure_analysis.py`, …) and `scripts/profile_kernel_breakdown.py` | Public versions of scripts from earlier campaigns. Some were later edited (for example, parameterized data paths). They are kept for reference. Except where `EXPERIMENTS.md` documents a command, they are not claimed to reproduce a result of the final paper |

**Removed sixth arm in the Table 1 harness.** The paper's unified timing campaign interleaved a sixth configuration in every invocation: the single-shot/windowed (SW) arm of the factorial study, built with the factorial implementation (84 patch tokens, no proportional attention). It is not a Table 1 row. The adapted harness no longer times it. Its position is still drawn from the same random sequence, so the five Table 1 arms run in the paper's relative order. Each invocation therefore runs five arms instead of six, and the timing context differs slightly from the paper campaign.

## 6. Components not included in this release

- **PiToMe compatibility adapter (Supplementary Table S23).** The PiToMe comparison used an archived local adapter that is not included in this release. The reported result remains documented in the supplementary material, but the corresponding adapter is not currently distributed.
- **COCO ViTDet-B detection drivers** for the packed exact-coordinate path (Table 5, Table S5).
- **LW-DETR-medium wrapper** (Table S15).
- **Cross-hardware campaign runners** for L40S and Jetson Orin (Table 9, Tables S4, S10, S14).
- **Hugging Face DINOv2/SigLIP adapter and ViT-L drivers** (Table 8).
- **Plain ViT-B/16 proxy driver** (Table 6, Table S9).
- **Producers of the separate batch-scaling campaign** (Fig. 6, Table S11), the 384² sweeps (Tables S1, S7), the ViT-B merge-layer sweep (Table S2), the overhead decomposition (Table 2) and the Triton end-to-end harness (Table S16).

These results remain documented in the paper and supplement. This release makes no claim that they can be reproduced from this repository.
