# Unified S³-Merge vs. official ToMe comparison (paper Table 1, Fig. 5)

This directory contains the harness behind the paper's **Table 1** (ImageNet-1K accuracy and inference efficiency of
DeiT-S/16 under the unified H100 NVL FP32 protocol), **Fig. 5** (accuracy–throughput trade-off) and the
**Supplementary Tables S20–S22** (configurations, per-invocation latencies, paired speed-ups and ratios).
It is a cleaned port of the harness that produced those numbers; [PROVENANCE.md](PROVENANCE.md) maps every file to its
original and lists every deliberate deviation.

Accuracy and latency are **separate** measurements: accuracy on the full 50,000-image ImageNet-1K validation set;
latency on fixed, device-resident synthetic input tensors.

## Files

| File | Role |
|---|---|
| `unified_common.py` | shared code: arm registry, model builders, official-ToMe import + source-hash check, structural hooks, synthetic timing inputs, randomized orders, ImageNet loader |
| `timing_unified.py` | ONE timing invocation (`--mode latency` = paper measurement; `--mode profile` = separate profiler invocations) |
| `timing_campaign.py` | runs the 10 latency + 3 profiling invocations as independent processes, then `timing_analysis.py` |
| `timing_analysis.py` | aggregates invocations: mean latency + 95 % t-CI, geometric-mean throughput, paired speed-ups / ratios (t-CI + bootstrap) |
| `accuracy_eval.py` | ONE accuracy run (one arm, one seed) on the full validation set |
| `accuracy_campaign.py` | runs the paper run list (15 runs) one at a time, each followed by `accuracy_check.py` |
| `accuracy_check.py` | post-run integrity check of one accuracy run |
| `accuracy_analysis.py` | aggregates the accuracy runs (Table 1 Top-1, mean ± SD over seeds for S³-Merge) |
| `gmac.py` | analytical GMAC and token trajectories (Table 1 GMAC/Tok. columns, Table S20) |
| `config_table1.yaml` | reference mirror of the defaults (not read by the scripts) |

## Arms

All arms are built from one DeiT-S/16 checkpoint (timm `deit_small_patch16_224.fb_in1k`).

| Key | Table 1 row | Build | PA (proportional attention) | Attention tokens per block (incl. [CLS]) | Final tokens | GMAC |
|---|---|---|---|---|---|---|
| `baseline` | Baseline | unmodified timm model | n/a | 197 × 12 | 197 | 4.5989 |
| `tome_r10` | ToMe r=10 | official ToMe @ `af95e4b1`: `tome.patch.timm(model, trace_source=False, prop_attn=True)`; `model.r = 10` | native ToMe size bias (blocks 1–11, 0-indexed) | 197, 187, …, 87 (MLP 187, …, 77) | 77 | 3.1286 |
| `tome_r12` | ToMe r=12 | same, `model.r = 12` | as above | 197, 185, …, 65 (MLP 185, …, 53) | 53 | 2.8439 |
| `s3_acc` | S³-Merge-Acc (L_s=9 + PA) | `models.sr_tome.apply_patch(model, target_tokens=84, late_start=8, window_size=7, weighted_merge=False, tau=0.3, content_adaptive=False, prop_attn=True, grid_size=14)` | log-size mask in blocks 9–11 (0-indexed) | 197 × 9, then 85 × 3 | 85 | 3.9316 |
| `s3_speed` | S³-Merge-Speed (L_s=7 + PA) | same with `late_start=6` | log-size mask in blocks 7–11 | 197 × 7, then 85 × 5 | 85 | 3.4867 |

`late_start` is the 0-indexed block whose output is merged (paper L_s = late_start + 1); 4 windows of 7×7 with
K = 21 targets each give 84 patch tokens + [CLS]. Every build checks that the S³-Merge configuration recorded by
`apply_patch` (`model.s3merge_config`) is exactly this operating point and that patching leaves the weights unchanged.
Attention implementation (Table S20, measured by the profiling invocations, not asserted): unmerged and S³-Merge use
timm's fused SDPA (S³-Merge with an additive mask); official ToMe uses explicit matrix products and softmax.

## Requirements

* **Hardware (paper):** one NVIDIA H100 NVL; clocks not locked (default boost clocks under a 400 W power cap; no thermal
  slowdown observed). Latency is hardware-, driver- and load-dependent and is **not bit-exactly reproducible**; expect
  the same qualitative picture, not identical milliseconds.
* **Software (paper environment):** Python 3.10.13, PyTorch 2.1.2, CUDA 12.1, timm 1.0.24, torchvision 0.16.2,
  datasets 4.5.0, numpy 1.26.4, scipy 1.15.3, Pillow 10.0.1. Every output records the versions actually used.
* **Official ToMe:** a checkout of <https://github.com/facebookresearch/ToMe> at commit
  `af95e4b1befa172dadccd8c81e223b10090f9579`, passed with `--tome-path` (or that commit installed as the `tome`
  package). Before import, `tome/patch/timm.py`, `tome/merge.py`, `tome/utils.py`, `tome/__init__.py`,
  `tome/patch/__init__.py` and `LICENSE` are checked against the pinned sha256 (disable with `--no-verify-tome`;
  results are then not comparable). The ToMe package is used unmodified.
* **Checkpoint:** timm `deit_small_patch16_224.fb_in1k` (`model.safetensors`, sha256
  `1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b`). Either let timm resolve it through the Hugging
  Face hub/cache, or pass the file with `--checkpoint` (its sha256 is then checked, `--expected-sha256`). In both cases
  the loaded weights must have the paper state hash (`--expected-state-sha256`, default `a17d0075…`).
  The paper runs used `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` with a pre-populated cache.
* **ImageNet-1K validation (accuracy only):** the Hugging Face dataset `ILSVRC/imagenet-1k` (gated; dataset revision
  `49e2ee26f3810fb5a7536bbf732a7b07389a47b5` in the paper), split `validation`, as the arrow files that `datasets`
  stores in its cache: `imagenet-1k-validation-00000-of-00012.arrow` … `-00011-of-00012.arrow`. `--data-dir` is that
  directory or any parent of it (e.g. the `datasets` cache directory; it is searched for the first shard). The shards
  are opened read-only with `datasets.Dataset.from_file` in shard order and concatenated (hf_index 0..49999); the
  script refuses to run unless exactly `--num-shards` shards with 50,000 rows in total are found.

## Protocol

**Timing** (`timing_unified.py --mode latency`, one process per invocation):
* FP32 eager, no autocast, no `torch.compile`; TF32 matmul off, cuDNN TF32 on (patch-embedding conv), cuDNN benchmark off.
* Inputs: `torch.randn(B, 3, 224, 224)` float32 from a CPU `torch.Generator` seeded 20260718, drawn for B = 1, 8, 32 in
  that order (sha256 of each tensor is recorded and compared with the paper inputs); moved to the GPU once per
  invocation and reused for every arm.
* For each arm in the invocation's randomized order: build the arm, then for B = 1, 8, 32: `torch.manual_seed(1000)`;
  **30 warm-up** forwards (synchronize after each); **200 timed** forwards, each
  `synchronize; t0 = perf_counter_ns(); model(x); synchronize; t1`. The timed region is the whole forward pass
  (including target draw, matching, aggregation, size tracking and the PA bias), under `torch.no_grad()`. No profiler,
  no hooks. nvidia-smi telemetry is recorded before and after every (arm, B) cell, outside the timed region.
* **Ten invocations** (0..9), each in a randomized arm order: `random.Random(20261104)` draws latency orders 0..9 then
  profiling orders 0..2 as `random.sample` over the six configurations of the paper campaign; the paper campaign also
  timed a sixth, non-Table-1 configuration, which is drawn but filtered out, so the relative order of the five arms
  in every invocation is that of the paper (`--order` overrides; other invocation indices require `--order`).
* Three separate profiling invocations (`--mode profile`, 30 warm-up + 20 profiled forwards per cell, merge entry
  points wrapped in `record_function`) give kernel/merge mechanics; they are never combined with latency.
* Every invocation is an independent process. In the paper each one ran as a fresh container of the paper image
  (12 CPUs); `timing_campaign.py` starts a fresh Python process per invocation in the current environment.

**Aggregation** (`timing_analysis.py`): per invocation and (arm, B), the mean of the 200 timed forwards.
Latency = mean of the ten invocation means with a 95 % t-interval (df = 9) → Table 1 B=1 / B=8 ms, Table S21.
Throughput = geometric mean over invocations of B / latency → Table 1 B=32 img/s.
Speed-up vs the baseline = paired within invocation, geometric mean exp(mean ln(L_base / L_arm)) with 95 % t-CI and a
percentile bootstrap CI (10,000 resamples, `numpy.random.default_rng(0)`) → Table S22; paired latency ratios between
operating points likewise, with a ±1 % equivalence-margin reading.

**Accuracy** (`accuracy_eval.py`, one process per run): full validation in hf_index order; `image.convert('RGB')` →
`Resize(256)` (bilinear) → `CenterCrop(224)` → `ToTensor` → `Normalize(ImageNet mean/std)`; batch 64, no shuffle,
8 workers, DataLoader generator seeded 0; FP32; `torch.manual_seed(s); np.random.seed(s)` once at run start; the CUDA
RNG is verified untouched until the first forward; forward under `no_grad` with synchronization before and after.
Every batch asserts the arm's token trajectory, attn_mask pattern, final token count and token mass (197).
Run list: `baseline_s0`; official ToMe r=10 and r=12 with seed 0 (primary run + determinism replicate; the primary is
reported); S³-Merge-Acc and -Speed with **seeds 0–4** (random target selection; Table 1 reports mean ± SD).

## Commands (full Table 1 campaign)

```bash
git clone https://github.com/facebookresearch/ToMe.git /path/to/ToMe
git -C /path/to/ToMe checkout af95e4b1befa172dadccd8c81e223b10090f9579

# 1. Accuracy: 15 sequential runs (~full ImageNet validation each), then the analysis.
python r3/unified_tome/accuracy_campaign.py --output-dir out/table1_accuracy \
    --data-dir /path/to/hf_datasets_cache --tome-path /path/to/ToMe
python r3/unified_tome/accuracy_analysis.py --runs-dir out/table1_accuracy/runs \
    --output-dir out/table1_accuracy/analysis

# 2. Timing: 10 latency + 3 profiling invocations (independent processes), then the analysis joined with accuracy.
#    Run on an otherwise idle GPU (an optional warn-only idle check samples nvidia-smi for 60 s first).
python r3/unified_tome/timing_campaign.py --output-dir out/table1_timing --tome-path /path/to/ToMe \
    --accuracy-analysis out/table1_accuracy/analysis/accuracy_analysis.json

# 3. Analytical GMAC / token trajectories.
python r3/unified_tome/gmac.py --output-dir out/table1_gmac
```

Add `--checkpoint /path/to/model.safetensors` to the campaign commands to use a local checkpoint file. Individual
steps: `timing_unified.py --mode latency --invocation <i> --output-dir <dir> --tome-path <ToMe>` runs one invocation;
`timing_analysis.py --raw-dir out/table1_timing/raw --output-dir <new dir> [--accuracy-analysis …]` re-aggregates;
`accuracy_eval.py --arm <arm> --seed <s> --output-dir <dir> --data-dir <data> --tome-path <ToMe>` runs one accuracy
run; `accuracy_campaign.py --resume` skips completed and checked runs. Outputs are never overwritten (a script refuses
to write into an existing output file or run directory). Every entry point prints its fully resolved configuration as
JSON before executing; all flags are listed by `--help`.

## Outputs

* `timing_campaign.py --output-dir D`: `D/raw/latency_inv00.json` … `latency_inv09.json`, `D/raw/profile_inv00..02.json`,
  `D/monitor/` (1 s nvidia-smi telemetry per invocation, compute processes before/after each latency invocation),
  `D/logs/`, `D/analysis/`.
* `latency_invNN.json`: `config` (resolved), `inputs` (tensor sha256 + match with the paper inputs),
  `checkpoint`, `tome_source_sha256`, `compute_apps_at_start_before_cuda_init`, `result.order`,
  `result.builds[arm]` (build info incl. `s3merge_config` / ToMe settings and the state hash),
  `result.cells["<arm>|<B>"]` = {`warmup_ms`[30], `times_ms`[200], `mean_ms`, `median_ms`, `start_utc`, `end_utc`,
  `telemetry_before`, `telemetry_after`}, `env` (python, torch, CUDA, cuDNN, timm, torchvision, datasets, numpy, GPU
  name, TF32/SDPA flags, autocast), `smoke`.
* `profile_invNN.json`: `result.cells["<arm>|<B>"]` with kernel launches and kernel time per forward, merge events /
  ranges / kernel launches / kernel time per forward, attention kernel names, top kernels.
* `analysis/timing_analysis.json`: `latency["<arm>|<B>"]` = {`mean_ms`, `sd_ms`, `ci95_ms`, `per_invocation_ms`,
  `throughput_img_s_geomean`}; `speedup_vs_baseline["<arm>|<B>"]` = {`speedup_geomean`, `ci95_t`, `ci95_boot`,
  `arith_mean_paired_ratio`, `ratio_of_means`, `net_latency_change_ms`}; `paired_ratios["<a>/<b>|<B>"]` = {`ratio`,
  `ci95_t`, `ci95_boot`, `reading`}; `profile`; `baseline_drift_cv`; `invocation_orders`.
  `analysis/timing_per_invocation.csv`: `invocation,method,B,mean_ms,order`. `analysis/unified_table.md`: readable table.
* `accuracy_campaign.py --output-dir D`: `D/runs/<run-id>/{predictions.npz, run_summary.json, RUN_COMPLETE.json}`,
  `D/checks/<run-id>.json`, `D/logs/`, `D/accuracy_run_manifest.tsv`, `D/accuracy_campaign_status.json`.
  `predictions.npz`: `hf_index, argmax, top5_idx, top5_logit, margin_top1_top2, label, correct_top1, correct_top5`.
  `run_summary.json`: `correct_top1, top1_pct, correct_top5, top5_pct, per_batch_correct, invariants, logits_sha256,
  build_info, env, checkpoint, tome_source_sha256, resolved_config`.
* `accuracy_analysis.py`: `accuracy_analysis.json` (`methods[arm]` = {`correct`, `top1_values_pct`, `top1_mean_pct`,
  `top1_sd_pp`, `top1_ci95`, `top5_mean_pct`, `runs`, `determinism` (ToMe), `analytical_gmac`, `disclosure`}),
  `accuracy_table.csv`, `accuracy_report.md`.

Table 1 columns: Tok. = `final_total` (Table S20); GMAC = `gmac.py`; Top-1 = `top1_mean_pct` (± `top1_sd_pp` for
S³-Merge); ΔTop-1 = arm − baseline; B=1 / B=8 ms = `latency["<arm>|1"/"<arm>|8"].mean_ms`; B=32 img/s =
`latency["<arm>|32"].throughput_img_s_geomean`.

Paper values (H100 NVL), for orientation only: Baseline 197 tok, 4.60 GMAC, 79.71 %, 2.37 ms, 3.92 ms, 2544 img/s;
ToMe r=10 77, 3.13, 79.09, 7.45, 7.64, 3040; ToMe r=12 53, 2.84, 78.90, 7.46, 7.66, 3287; S³-Merge-Acc 85, 3.93,
79.50 ± 0.02, 2.92, 3.64, 2948; S³-Merge-Speed 85, 3.49, 79.01 ± 0.06, 3.04, 3.37, 3330.

## Smoke mode (software testing only)

`--smoke` exists on `timing_unified.py`, `timing_campaign.py`, `accuracy_eval.py`, `accuracy_campaign.py` and the
analysis/check scripts. It uses tiny counts (3 warm-up, 10 timed, 3 profiled forwards), allows `--device cpu` and
`--no-pretrained` (random weights), adds a correctness check to timing (two independent builds per arm, structural
trajectory/mask/final-token checks), and replaces ImageNet with a few random synthetic images with random labels.
Smoke outputs carry `"smoke": true` and the scripts print that they are **not paper measurements**; the analysis
scripts accept them only with `--smoke`. Smoke never changes a default. Example (no GPU, no dataset, no download):

```bash
python r3/unified_tome/timing_campaign.py --output-dir /tmp/tu_smoke_timing --invocations 2 --profile-invocations 1 \
    --idle-check-samples 0 --smoke --device cpu --no-pretrained --tome-path /path/to/ToMe
python r3/unified_tome/accuracy_campaign.py --output-dir /tmp/tu_smoke_acc --smoke --device cpu --no-pretrained \
    --tome-path /path/to/ToMe --smoke-images 8 --batch-size 4 --num-workers 2
python r3/unified_tome/accuracy_analysis.py --runs-dir /tmp/tu_smoke_acc/runs --output-dir /tmp/tu_smoke_acc/analysis --smoke
```

## Not included

The ToMe fidelity checks of the original harness (per-block trajectory, proportional attention and logits of official
ToMe compared against the same release on its original timm version), the S³-Merge operating-point fidelity check,
and the resource and scheduling checks specific to the original compute environment are not part of this directory;
see [PROVENANCE.md](PROVENANCE.md).
