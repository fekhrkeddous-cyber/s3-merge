# Baselines

This directory provides a unified API for evaluating training-free token merging baselines.

## Setup

S³-Merge is self-contained in `models/`. To reproduce baseline comparisons from the paper, install the following third-party methods:

### ToMe (ICLR 2023)

```bash
pip install git+https://github.com/facebookresearch/ToMe.git
```

### PiToMe (NeurIPS 2024)

```bash
git clone https://github.com/hchautran/PiToMe.git baselines/PiToMe
```

### ATC (ECCV 2024)

```bash
git clone https://github.com/JoakimHaurum/ATC.git baselines/ATC
```

### DSM (NeurIPS 2024 Workshop)

Our DSM re-implementation is included in `dsm_impl.py` — no official code was available at time of writing.

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

- All baselines run through one common harness (`eval_imagenet_baselines.py`: same model, dataset, hardware, timing procedure). In the paper's Table 1, Baseline/ToMe/S³-Merge timings come from the unified FP32 campaign, while PiToMe/ATC/DSM retain their clearly labelled original-campaign measurements
- DSM is our vectorized re-implementation from the paper description; we release the code for verification
