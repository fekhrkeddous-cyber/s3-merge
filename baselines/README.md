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
git clone https://github.com/piotrmwojcik/PiToMe.git baselines/PiToMe
```

### ATC (ECCV 2024)

```bash
git clone https://github.com/JoakimHauworWorWorrum/ATC.git baselines/ATC
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

- All baselines are evaluated under identical conditions (same model, dataset, hardware, timing protocol) via `eval_imagenet_baselines.py`
- DSM is our vectorized re-implementation from the paper description; we release the code for verification
