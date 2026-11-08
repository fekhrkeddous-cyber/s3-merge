# S³-Merge: Spatially-Restricted Single-Shot Token Merging for Efficient Vision Transformers

> Training-free token merging that replaces per-layer global matching with **one late,
> spatially windowed merge**, converting token reduction into **measured wall-clock
> acceleration** while preserving the 2D layout that dense prediction requires.

<p align="center">
  <img src="figures/pipeline_diagram.png" width="90%" alt="S³-Merge pipeline overview"/>
</p>

Public repository: https://github.com/fekhrkeddous-cyber/s3-merge

---

## Method overview

**S³-Merge Core** is a single-shot merge after a late transformer block:
- the patch tokens are partitioned into w×w spatial windows;
- K = ⌊T / n_windows⌋ target tokens are kept per window;
- each remaining token is assigned to its most similar target by hard cosine argmax;
- each cluster is fused by uniform averaging (paper Eq. 1).

[CLS] is excluded from windowing. A restore mapping reconstructs the full token grid for dense heads. Windowing bounds merge displacement by √2(w−1) patches.

**Proportional attention (PA)** is an *optional, task-dependent* correction and is not part of Core. It recovers 0.48% Top-1 on DeiT-S classification (paper Table S24) and does not help COCO detection (Table S8). The dense experiments and the factorial study run Core without PA.

**Target selection:**
- *seeded random Static-K*: the classification default;
- *deterministic variance-aware Static-K*: used for dense prediction;
- *Adaptive-K*: variable per-window counts under an exact per-image budget, with packed no-padding execution and exact-coordinate relative positions (`models/adaptive_k.py`).

**Safe Mode** is an empirical routing policy, not a guarantee. `models/safemode.py` implements its **calibration step** for DeiT-S/16 at 224²:
- it measures baseline and merged latency for B ∈ {1, 2, 4, 8, 16, 32};
- a batch size qualifies when the bootstrap 95% lower confidence bound of the speed-up exceeds 1.02;
- it caches B*.

This repository ships the calibrator only, not a dispatcher. The paper does not report routed end-to-end latency.

## Main results (paper Table 1)

ImageNet-1K, DeiT-S/16, H100 NVL, FP32, unified protocol with official ToMe:

| Method | Tok. | Top-1 (%) | ΔTop-1 | B=1 (ms) ↓ | B=8 (ms) ↓ | B=32 (img/s) ↑ |
|--------|-----:|-----------|--------|-----------:|-----------:|---------------:|
| Baseline | 197 | 79.71 | — | 2.37 | 3.92 | 2544 |
| ToMe r=10 | 77 | 79.09 | −0.62 | 7.45 | 7.64 | 3040 |
| ToMe r=12 | 53 | 78.90 | −0.81 | 7.46 | 7.66 | 3287 |
| **S³-Merge-Acc (L_s=9, +PA)** | 85 | **79.50 ± 0.02** | **−0.21** | 2.92 | 3.64 | 2948 |
| S³-Merge-Speed (L_s=7, +PA) | 85 | 79.01 ± 0.06 | −0.69 | 3.04 | **3.37** | **3330** |

- **Latencies** are means over **ten** independent invocations in randomized method order (30 synchronized warm-up passes, then 200 timed passes with synchronization around every iteration). B=32 throughput is the geometric mean over invocations.
- **Accuracy** is evaluated separately on the full 50k validation set; S³-Merge Top-1 is the five-seed mean ± std.
- **Environment:** PyTorch 2.1.2, CUDA 12.1, timm 1.0.24, no autocast, no `torch.compile`. Clocks were not locked (default boost, 400 W cap).
- **Reproduction:** latency is hardware- and software-dependent, so these values are not expected to be reproduced bit-exactly.

## Experimental campaigns: do not combine values across them

The paper's results come from **separate campaigns** with different protocols. Values from different campaigns should not be combined or compared directly, even for the same configuration.

| Campaign | Paper items | Protocol | Code in this repository |
|---|---|---|---|
| Unified official-ToMe comparison | Table 1, Fig. 5, Tables S20–S22 | H100 NVL, FP32, B ∈ {1, 8, 32}, 30+200 with per-iteration sync, **10 invocations**, randomized order; accuracy 50k val (5 seeds for S³-Merge) | `r3/unified_tome/` (adapted harness; the sixth, non-Table-1 configuration that the paper campaign interleaved is not timed) |
| Controlled 2×2 factorial study (schedule × scope) | Table 4, Tables S17–S19 | DeiT-S/16 at 224², 84 patch tokens in every arm, **no PA**, FP32, H100 NVL; Top-1 five-seed mean on the full 50k val; latency mean over **ten paired invocations** | `r3/factorial/` (adapted harness) |
| ViT-B/16 model-scale row | Table 7 (row b) | Historical AMP/FP16 run, PA off, B ∈ {1, 64}, 30+200 with sync around the timed loop, one invocation (Supplementary Section S1.2) | `r3/table7_vitb16/` (reconstruction of the configuration) |
| Separate batch-scaling campaign | Fig. 6, Table S11 | FP32, five invocations, B ∈ [1, 64] | not included |
| Downstream: COCO ViTDet-B (packed path), LW-DETR-medium | Table 5, Tables S5, S15 | FP32, five invocations, PA off | merge core only (`models/adaptive_k.py`); drivers not included |
| Cross-hardware: L40S, Jetson Orin | Table 9, Tables S4, S10, S14 | per-device campaigns, B* device-specific | not included |
| Other ablations and original-campaign baselines | e.g. Tables S2, S3, S12, S23, S24 | protocol stated per table in the supplement | partly (see `EXPERIMENTS.md`) |

`PAPER_CODE_MATRIX.md` lists, for every main result, the script, the configuration, the public status, the test status and the known limitations. `REPRODUCIBILITY.md` (§5a) separates the adapted reproduction harnesses from the historical scripts. None of them reproduces historical measurements bit-exactly.

## Configurations

`late_start` is the 0-indexed block whose output is merged, so **paper L_s = `late_start` + 1**.

| Configuration | `apply_patch(...)` | Paper |
|---|---|---|
| S³-Merge-Acc | `target_tokens=84, late_start=8, window_size=7, prop_attn=True` | L_s=9, T=84, w=7, PA |
| S³-Merge-Speed | `target_tokens=84, late_start=6, window_size=7, prop_attn=True` | L_s=7, T=84, w=7, PA |
| Core without PA | as above with `prop_attn=False` | factorial study, dense prediction |
| Table 7 ViT-B/16 | `target_tokens=136, late_start=8, window_size=7, prop_attn=False` (AMP/FP16) | L_s=9, 136 patch tokens (137 with [CLS]), no PA |
| 384² (patch 16, 24×24 grid) | `window_size=8`; the grid is read from the model | Tables S1, S7 |

**Grid and window.**
- The patch grid is taken from the model, or derived from the token count, and validated.
- The window size must divide the grid. Otherwise `apply_patch` raises a `ValueError` instead of silently skipping the merge.
- The window-sweep keep-all controls (w ≥ grid, or w = 1) perform no merge by construction. They must be requested explicitly with `keep_all_control=True`.

**Token budget.** K = ⌊T / n_windows⌋ per window. For example, `target_tokens=171` at 224² with w=7 (4 windows) keeps 168 patch tokens plus [CLS].

**Resolved configuration.** `model.s3merge_config` holds the resolved configuration.

## Installation

```bash
pip install -r requirements.txt     # tested environment: Python 3.10, torch 2.1.2 (CUDA 12.1), timm 1.0.24
pip install "git+https://github.com/facebookresearch/ToMe.git@af95e4b1befa172dadccd8c81e223b10090f9579"   # only for r3/unified_tome
# or, as a container:
docker build -f Dockerfile -t s3merge:cu121 .
docker run --gpus all --shm-size=16g -it -v $PWD:/workspace s3merge:cu121
```

ImageNet-1K is read through Hugging Face `datasets` (`ILSVRC/imagenet-1k`), which requires accepting the dataset's terms. Data locations are passed to each script as arguments; see `REPRODUCIBILITY.md`.

## Quick start (classification)

```python
import timm, torch
from models.sr_tome import apply_patch

model = timm.create_model("deit_small_patch16_224", pretrained=True).cuda().eval()
apply_patch(model, target_tokens=84, late_start=8, window_size=7, prop_attn=True)   # S³-Merge-Acc (L_s=9)
print(model.s3merge_config)        # resolved grid, windows, K per window, output tokens, PA setting
x = torch.randn(8, 3, 224, 224, device="cuda")
with torch.no_grad():
    y = model(x)
```

For Adaptive-K and variance-aware Static-K on grid features, see `models/adaptive_k.py`. Its functions are exercised in `tests/test_release.py`.

## Tests

```bash
python tests/test_release.py    # synthetic-tensor unit tests (CPU; no dataset or checkpoint downloads)
python tests/test_public.py     # import, syntax, shape and hygiene checks
python tests/smoke_gpu.py       # bounded GPU execution check (not a benchmark)
```

## Scope and limitations

- **Model structure:** `apply_patch` targets timm-style ViTs with exactly one [CLS] token and a square patch grid. Other layouts raise an error.
- **PA:** requires timm's `Block.forward(x, attn_mask=...)` (timm 1.0.24 is pinned).
- **Packed variable-K:** requires every post-merge block to be global-attention compatible.
- **Not included in this release:**
  - the COCO ViTDet drivers (Table 5);
  - the LW-DETR wrapper (Table S15);
  - the L40S and Jetson Orin campaign runners (Tables 9, S4, S10, S14);
  - the Hugging Face DINOv2/SigLIP adapter (Table 8);
  - the PiToMe compatibility adapter (Table S23).

  `REPRODUCIBILITY.md` lists all unavailable components.
- **Evaluated scope:**
  - classification: DeiT-S/16, ViT-B/16, ViT-L/16, DINOv2-L/14, SigLIP-L/16;
  - dense prediction: ViTDet-B and LW-DETR-medium on COCO;
  - deployment: H100 NVL, L40S, Jetson Orin.

  Claims are restricted to this scope.

## Citation

The manuscript is under review at *Pattern Recognition* (revision R3). A BibTeX entry will be added upon acceptance.

## License

MIT (see `LICENSE`). Third-party projects are not redistributed; see `THIRD_PARTY_NOTICES.md`.
