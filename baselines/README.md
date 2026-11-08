# Baselines

This directory provides a unified API for evaluating training-free token merging baselines.

## Setup

S³-Merge is self-contained in `models/`. The third-party methods below are installed separately and are not redistributed here (check each project's licence; ToMe and PiToMe are CC BY-NC 4.0).

### ToMe (ICLR 2023)

```bash
pip install "git+https://github.com/facebookresearch/ToMe.git@af95e4b1befa172dadccd8c81e223b10090f9579"   # commit used for paper Table 1
```

### PiToMe (NeurIPS 2024)

```bash
git clone https://github.com/hchautran/PiToMe.git baselines/PiToMe && git -C baselines/PiToMe checkout 550b5deed94aadfeac28bfbe381d87e672044a40
```

The timm >= 1.0 compatibility adapter used for Supplementary Table S23 is **not included** in this release.

### ATC (ECCV 2024)

```bash
git clone https://github.com/JoakimHaurum/ATC.git baselines/ATC
```

### DSM (NeurIPS 2024 Workshop)

`dsm_impl.py` is an exploratory re-implementation from the paper description (no official code was available).
DSM is **not** compared quantitatively in the paper, and this code produced none of its results.

## Usage

```python
import timm
from baselines import apply_method

model = timm.create_model('deit_small_patch16_224', pretrained=True)

# Apply any baseline
model = apply_method(model, 'pitome', ratio=0.55)
# model = apply_method(model, 'atc', reduction_ratio=0.5)
# model = apply_method(model, 'dsm', delay_layer=6, merge_r=12)
```

## Notes

- `eval_imagenet_baselines.py` is the harness of the original (earlier) campaign. Paper Table 1 (Baseline, ToMe r=10/12, S³-Merge-Acc/Speed) comes from the separate ten-invocation unified FP32 comparison in `r3/unified_tome/` with official ToMe; PiToMe and ATC appear only in Supplementary Table S23 (original-campaign measurements, not comparable with Table 1).
- The PiToMe comparison used an archived local adapter that is not included in this release (see `REPRODUCIBILITY.md`); the PiToMe arm of `eval_imagenet_baselines.py` therefore stops with an explicit error
- DSM: exploratory re-implementation, released for inspection only; not used for any number in the paper
