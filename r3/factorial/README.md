# Controlled 2×2 factorial study: merge schedule × matching scope

This directory holds the code behind paper **Table 4** and **Supplementary Tables S17–S19**: a controlled
comparison of when to merge (single-shot vs. progressive) and where to match (windowed vs. global) on DeiT-S/16,
within one implementation, at a matched final budget of 84 patch tokens and **without proportional attention (PA)**.

## Arms

Block indices below are 0-indexed, as in the code; the paper counts blocks from 1, so index 8 is the paper's "block 9".

| Arm | Schedule | Scope | Merge after block index | Tokens removed per window and step | Token trajectory, blocks 1–12 (incl. [CLS]) |
|---|---|---|---|---|---|
| `baseline` | none | – | – | – | 197 × 12 |
| `SW` | single-shot | windowed | 8 | 28 | 197 × 9, then 85 × 3 |
| `SG` | single-shot | global | 8 | 28 | 197 × 9, then 85 × 3 |
| `PW` | progressive | windowed | 5, 6, 7, 8 | 7, 7, 7, 7 | 197 × 6, 169, 141, 113, then 85 × 3 |
| `PG` | progressive | global | 5, 6, 7, 8 | 7, 7, 7, 7 | 197 × 6, 169, 141, 113, then 85 × 3 |
| `core` (control) | single-shot | windowed | 8 | 28 | 197 × 9, then 85 × 3 |

* All factorial arms remove the same 112 patch tokens (28 per 7×7 window) and keep 84 patch tokens (21 per window) plus [CLS].
* `core` is the canonical S³-Merge Core path. It is called through `models.sr_tome.apply_patch(target_tokens=84,
  late_start=8, window_size=7, weighted_merge=False, tau=0.3, content_adaptive=False, prop_attn=False,
  grid_size=14, keep_all_control=False)`. It is a control that the factorial `SW` arm must reproduce, not a fifth arm.
  The paper reports that SW seed 0 and Core seed 0 give identical correct counts.

The factorial arms are implemented in `factorial_merge.py`:

* **Priority draw.** Each forward makes exactly one priority draw, `torch.rand(B*4, 49)`, at its first merge event.
  The call, shape and window layout are those of the canonical `sr_tome_merge_exact`.
* **Nested removal.** Within each window, the survivors are kept in priority order. Each stage removes the k
  lowest-priority survivors. Every arm therefore ends with the same 84 targets for a given seed.
* **Scope.** Scope only masks the candidates. The windowed arms exclude candidates outside the source's original
  7×7 window; the global arms do not mask.
* **Aggregation.** Tokens are fused with a size-weighted, mass-conserving mean, computed as a dense one-hot matrix
  product. In the single-shot arms every size is 1, so this equals the paper's Eq. 1. **In the progressive arms,
  previously merged tokens carry their sizes: each final token is the mean of all original patches it represents.**
* **Ties.** An argmax tie goes to the lowest original patch coordinate.
* **Output order.** After the last stage, the 21 survivors per window are put back into the canonical
  `torch.topk(p, 21, sorted=False)` order.
* **Attention.** PA is off: the post-merge blocks are plain timm Blocks, called without `attn_mask`.

## Files

| File | Role |
|---|---|
| `factorial_merge.py` | Arm implementation (`ARM_SPECS`, `merge_stage`, `FactorialModel`) |
| `factorial_common.py` | Data loading, model builders, invariant checks, environment record |
| `accuracy.py` | One accuracy run (one arm × one seed per process) on the full ImageNet-1K validation set |
| `timing.py` | One timing invocation (latency; optionally profile or merge_micro) |
| `campaign.py` | Runs the full accuracy campaign (22 runs) or timing campaign (10 + 3 + 5 invocations), one process per run |
| `analyze_accuracy.py` | Accuracy statistics: Table 4 Top-1, Tables S17–S18 |
| `analyze_timing.py` | Latency statistics: Table 4 latency columns, Table S19 |
| `test_factorial.py` | CPU unit tests on synthetic tensors |
| `config_table4.yaml` | Reference copy of the protocol defaults. The scripts do not read it; the unit test checks it against the scripts |

Run every command from the repository root. Each entry point prints its fully resolved configuration (JSON)
before it runs, and lists its options with `--help`.

## Requirements

* **Reference environment:** Python 3.10, PyTorch 2.1.2 (CUDA 12.1), torchvision 0.16.2, timm 1.0.24,
  datasets 4.5.0, Pillow 10.0.1, numpy 1.26.4, scipy 1.15.3, one NVIDIA H100 NVL.
  * The repository `Dockerfile` pins torch, timm, datasets and numpy to these versions.
  * scipy is needed only by the two analysis scripts.
  * huggingface_hub (a dependency of datasets) downloads the checkpoint.
* **Data.** The ImageNet-1K validation split of the Hugging Face dataset `ILSVRC/imagenet-1k`, as its 12 arrow
  shards `imagenet-1k-validation-000NN-of-00012.arrow`.
  * The dataset is gated: accept its terms on the Hub first.
  * Prepare it once, for example with `datasets.load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=...)`.
    The shards then sit in `<cache_dir>/ILSVRC___imagenet-1k/default/0.0.0/<revision>/`; the paper used revision
    `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`.
  * Pass that directory as `--data-dir`. The drivers open the shards read-only with `datasets.Dataset.from_file`,
    in shard order, and check that there are 50,000 images.
  * Labels are the dataset's `label` column. Preprocessing: `image.convert('RGB')` → Resize(256) → CenterCrop(224)
    → ToTensor → Normalize(ImageNet mean/std).
  * The order is fixed (index 0..49,999, no shuffle).
* **Checkpoint.** timm `deit_small_patch16_224.fb_in1k`.
  * By default `model.safetensors` is fetched from the Hugging Face Hub (`timm/deit_small_patch16_224.fb_in1k`;
    the paper used snapshot `91327a9c99f98fe6b524cd4d397b7226b80e1365`) or taken from the local Hub cache.
  * Use `--checkpoint /path/to/model.safetensors` to point to a local copy.
  * The file must have sha256 `1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b`
    (`--expected-sha256`).
  * The loaded weights must hash to `a17d00754ef1e5401cf33cf61af23094c48b8f4a4f5f8898d8ae36b9d81925e1`
    (`--expected-state-sha256`).
  * Pass `none` to either flag to disable that check.

## Accuracy campaign (Table 4 Top-1, Tables S17–S18)

**Protocol** (the command-line defaults):

* **Precision and backends:** FP32 weights, inputs and logits; TF32 matmul off; cuDNN TF32 on (the PyTorch
  default; it affects only the patch-embedding convolution); `cudnn.benchmark` off; no autocast; no `torch.compile`; PA off.
* **Data loading:** batch size 64, 8 DataLoader workers, `shuffle=False`, `drop_last=False`.
* **Seeding:** `torch.manual_seed(s); np.random.seed(s)` once, at run start. On CUDA the run checks that the CUDA
  generator is untouched between seeding and the first forward, so every arm receives the same priority draws per
  image for a given seed.
* **Checks on every batch:** the token trajectory, the absence of `attn_mask`, and finite FP32 logits. For the
  factorial arms they also cover the final targets, masses, membership and windowed zero cross-window membership.
  Any violation stops the run.
* **Seeds and order:** seeds 0–4 for the factorial arms; the unmerged baseline is deterministic and runs once,
  at seed 0. The run order is `baseline_s0`, `core_s0`, then SW, SG, PW, PG for seed 0, 1, 2, 3, 4.

```bash
# full campaign: 22 runs, one process each, stops at the first failure
python r3/factorial/campaign.py accuracy --output-dir runs/r3_factorial/accuracy -- \
    --data-dir /path/to/ILSVRC___imagenet-1k/default/0.0.0/<revision> --device cuda
# (add --checkpoint /path/to/model.safetensors after "--" to use a local checkpoint; --resume skips completed runs)

# a single run
python r3/factorial/accuracy.py --arm SW --seed 0 --data-dir /path/to/shards --output-dir runs/r3_factorial/accuracy

# analysis (CPU)
python r3/factorial/analyze_accuracy.py --runs-dir runs/r3_factorial/accuracy --output-dir runs/r3_factorial/analysis
```

**Per-run outputs** in `<output-dir>/<arm>_s<seed>/`. Files are written atomically and never overwritten; an
existing run directory stops the run.

* `predictions.npz`: `hf_index`, `argmax`, `top5_idx`, `top5_logit`, `margin_top1_top2`, `label`, `correct_top1`, `correct_top5`.
* `targets_u8.npy` (factorial arms only): the final 84 target coordinates per image.
* `run_summary.json`, which records:
  * counts: `correct_top1`, `top1_pct`, `correct_top5`, `top5_pct`, `per_batch_correct` and its sha256;
  * checks: `invariants`, `logits_sha256`, `state_sha`, `cross_window_member_rate`, `cluster_size_hist`;
  * environment: `env` (versions, backend flags, GPU), `gpu_start` and `gpu_end` (nvidia-smi);
  * provenance: `config` (the resolved configuration), `source_sha256`, and the checkpoint path and sha256.
* `RUN_COMPLETE.json`: the sha256 of each file above.

**Statistics** (`analyze_accuracy.py`):

* The statistical unit is one complete 50,000-image run per (arm, seed).
* Contrasts are computed within each seed from integer correct counts, with exact rational arithmetic:
  * E_schedule = ((SG+SW) − (PG+PW))/2
  * E_scope = ((PW+SW) − (PG+SG))/2
  * I = SW − SG − PW + PG
  * simple effects: SW−SG, PW−PG, SW−PW, SG−PG
* Across the 5 seeds the analysis reports the mean, the sample SD and a 95% Student-t interval (df = 4).
* Readings use a **±0.10-point margin**, fixed before the runs:
  * **EQUIVALENT-WITHIN-MARGIN**: the CI lies inside the margin.
  * **EFFECT**: the CI excludes 0 and |mean| ≥ 0.10.
  * **INCONCLUSIVE**: otherwise.
  * These labels are mechanical. For the accuracy interaction I, the paper's data give a 95% CI of [−0.0553, 0.0777], so the rule prints EQUIVALENT-WITHIN-MARGIN. The paper describes the same interval as "not statistically resolved", because the CI includes zero. Both statements describe the same numbers; the paper's wording is the authoritative interpretation.
* It also reports:
  * the Core control: SW vs. core discordance at seed 0, with a tolerance of 10 images;
  * the final targets, which must be identical across SW/SG/PW/PG within each seed;
  * the difference between each correct count and the published count in Table S17;
  * whether the baseline's per-batch correct-count sha256 matches the paper's run.
* Outputs: `accuracy_seed_level_results.csv`, `accuracy_paired_contrasts.csv`, `accuracy_discordance.csv`,
  `accuracy_analysis.json`, `accuracy_report.md`. Existing files are never overwritten.

## Timing campaign (Table 4 latency columns, Table S19)

**Protocol** (the command-line defaults; the paper's hardware is one NVIDIA H100 NVL, FP32):

* **Arms and batch sizes:** baseline, SW, SG, PW, PG (all PA off), at B ∈ {1, 8, 32}.
* **Inputs:** fixed synthetic, device-resident tensors: `torch.randn` float32 [B, 3, 224, 224] from a CPU generator
  seeded 20260718, drawn for B = 1, 8, 32 in that order. The output records their sha256 and whether they equal
  the paper's inputs.
* **Invocations:** 10 independent latency invocations, each a fresh Python process. Within an invocation the arms
  run in a randomized order (`random.Random(20261103)`, the same per-invocation orders as the paper), each arm is
  built once, and B runs in the order 1, 8, 32.
* **Per (arm, B) cell:**
  * `torch.manual_seed(1000)`;
  * 30 warm-up forwards, each followed by `torch.cuda.synchronize()`;
  * 200 timed forwards, each `synchronize(); t0 = perf_counter_ns(); model(x); synchronize(); t1 = perf_counter_ns()`
    under `no_grad`.
* **What the timed region covers:** the whole forward, including the priority draw, the matching, the aggregation
  and all bookkeeping.
* **Telemetry:** nvidia-smi is sampled before and after each cell, never inside a timed loop. No profiler runs in latency mode.
* **Clocks:** not locked. Under load the paper's GPU ran at default boost clocks under a 400 W power cap.
* **Exclusive GPU:** use one. The driver warns, but does not stop, if other compute processes are present at start.

```bash
# full campaign: latency 0-9, then profile 0-2, then merge_micro 0-4 (one process each)
python r3/factorial/campaign.py timing --output-dir runs/r3_factorial/timing -- --device cuda
# only the latency series used by Table 4 / Table S19
python r3/factorial/campaign.py timing --modes latency --output-dir runs/r3_factorial/timing -- --device cuda

# a single invocation
python r3/factorial/timing.py --mode latency --invocation 0 --output-dir runs/r3_factorial/timing

# analysis (CPU)
python r3/factorial/analyze_timing.py --raw-dir runs/r3_factorial/timing --output-dir runs/r3_factorial/analysis
```

**Per-invocation output:** `<output-dir>/<mode>_inv<NN>.json`, never overwritten. It records `config`, `inputs`
(tensor sha256s), `compute_apps_at_start_before_cuda_init`, `env` and `source_sha256`, plus a `result` that
depends on the mode:

* **`latency`:** `result.cells["<arm>|<B>"]` with `warmup_ms`, `times_ms` (200 values), `mean_ms`, `median_ms`,
  `telemetry_before` and `telemetry_after`.
* **`profile` and `merge_micro`:** supplementary diagnostics that are not reported in Table 4 or Tables S17–S19.
  * `profile`: kernel launches and kernel time per forward, with the merge stage tagged.
  * `merge_micro`: the isolated cost of each merge stage.

**Statistics** (`analyze_timing.py`):

* The unit is one invocation (n = 10). Each (arm, B) value is the **mean** of its 200 timed iterations; medians are descriptive only.
* Factorial contrasts are taken on ln(latency) within each invocation:
  * they use the same E_schedule, E_scope, I and simple-effect definitions as the accuracy analysis;
  * they are summarized as ratios exp(mean) with a 95% Student-t interval (df = 9);
  * a robustness interval comes from a percentile bootstrap over invocations (10,000 resamples, `numpy.random.default_rng(0)`).
* Readings use a **±1% latency margin**: EQUIVALENT-WITHIN-MARGIN if the ratio CI lies inside [0.99, 1.01];
  EFFECT if it excludes 1 and |ln ratio| ≥ ln 1.01; otherwise INCONCLUSIVE.
* Speedup vs. baseline is the geometric mean of the paired ratios, with a t-CI. Throughput is the geometric mean of B/latency.
* Isolated merge cost and profiler counts are reported in their own tables and never combined with wall-clock latency.
* Outputs: `timing_latency_per_invocation.csv`, `timing_factorial_contrasts.csv`, `timing_speedup_throughput.csv`,
  `timing_analysis.json`, `timing_report.md`.

**Latency depends on the hardware and is not bit-exactly reproducible.** Expect absolute values to differ on other
GPUs, drivers, clocks or software versions. Only the paired, within-invocation ratios are designed to be compared.

## Reference values (paper Table 4; H100 NVL, FP32, PA off)

| Config. | GMAC | Top-1 (%) | B=1 (ms) | B=8 (ms) | B=32 (ms) |
|---|---|---|---|---|---|
| Unmerged | 4.5989 | 79.708 | 2.36 | 3.92 | 12.56 |
| PG | 3.5919 | 78.815 ± 0.055 | 4.92 | 5.52 | 11.05 |
| PW | 3.5919 | 78.815 ± 0.067 | 5.16 | 5.78 | 11.25 |
| SG | 3.9316 | 78.976 ± 0.071 | 3.34 | 4.33 | 11.20 |
| SW | 3.9316 | 78.987 ± 0.037 | 3.40 | 4.40 | 11.22 |

* Top-1 is the five-seed mean ± sample SD; the seed-level counts are in Table S17.
* The schedule contrast (S − P) is +0.1664 points, 95% CI [+0.1194, +0.2134]. The scope contrast (W − G) is
  +0.0056 points, 95% CI [−0.0374, +0.0486].

## Smoke mode and unit tests (software testing only)

`--smoke` exists only to test the software:

* it uses tiny counts and synthetic images or inputs;
* `--no-pretrained` gives random, seeded weights and needs no download;
* it can run on CPU;
* the timing driver also checks that the timing-path logits are bitwise equal to the accuracy-path logits.

Smoke outputs carry `"smoke": true`, the scripts print that they are **not paper measurements**, and the analysis
scripts refuse to mix smoke and non-smoke runs.

```bash
python r3/factorial/test_factorial.py                       # CPU unit tests, exit code 0 = all passed
python r3/factorial/campaign.py accuracy --output-dir /tmp/smoke/acc -- --smoke --device cpu --no-pretrained
python r3/factorial/analyze_accuracy.py --runs-dir /tmp/smoke/acc --output-dir /tmp/smoke/analysis
python r3/factorial/campaign.py timing --output-dir /tmp/smoke/timing -- --smoke --device cpu --no-pretrained --batch-sizes 1 8
python r3/factorial/analyze_timing.py --raw-dir /tmp/smoke/timing --output-dir /tmp/smoke/analysis
```

The unit tests cover:

* the token counts after every schedule step, and the per-block trajectories inside a 12-block timm ViT;
* window-scope masking;
* that the size-weighted aggregation equals the mean of each token's member patches;
* determinism under a fixed seed, and identical final targets across arms;
* that SW equals the canonical Core merge;
* the lowest-coordinate tie rule;
* the randomized timing orders;
* that this YAML file agrees with the command-line defaults.

See `PROVENANCE.md` for how these files relate to the code used for the paper.
