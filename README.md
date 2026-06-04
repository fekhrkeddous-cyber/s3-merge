# S³-Merge: Spatially-Restricted Single-Shot Token Merging for Efficient Vision Transformers

> **Training-free** token merging framework that replaces per-layer global matching with a single, spatially-constrained late merge — delivering real wall-clock speedups without retraining.

<p align="center">
  <img src="figures/pipeline_diagram.png" width="90%" alt="S³-Merge pipeline overview"/>
</p>

---

## Highlights

| Feature | Description |
|---------|-------------|
| **9× lower overhead** | Single-shot merging eliminates per-layer matching cost |
| **Safe Mode** | Predictive crossover estimator (B*) prevents regression at any batch size |
| **Spatial fidelity** | Windowed matching bounds merge displacement (Proposition 1) |
| **Dense-task ready** | Restore mapping enables detection/segmentation out of the box |

### Main Results (ImageNet-1K, DeiT-S/16, H100 NVL)

| Method | Top-1 (%) | ΔTop-1 | B=1 (ms) ↓ | B=32 (img/s) ↑ |
|--------|-----------|--------|-------------|-----------------|
| Baseline | 79.71 | — | 2.40 | 2680 |
| ToMe r=10 | 79.09 | −0.62 | 7.44 | 3054 |
| **S³-Merge-Acc (l8+PA)** | **79.50** | **−0.21** | **2.92** | **2998** |
| S³-Merge-Speed (l6+PA) | 79.01 | −0.70 | 3.03 | **3377** |

---

## Installation

### Option A: pip (recommended)

```bash
git clone https://github.com/anonymous-s3merge/s3-merge.git
cd s3-merge
pip install -r requirements.txt
```

### Option B: Docker (full reproducibility)

```bash
docker build -f Dockerfile -t s3merge:cu121 .
docker run --gpus all --shm-size=16g -it \
  -v /path/to/imagenet:/data/imagenet \
  -v $PWD:/workspace \
  s3merge:cu121
```

### Baseline Methods (optional)

To reproduce baseline comparisons (ToMe, PiToMe, ATC), install the required packages:

```bash
# ToMe (ICLR 2023)
pip install git+https://github.com/facebookresearch/ToMe.git

# PiToMe (NeurIPS 2024)
git clone https://github.com/piotrmwojcik/PiToMe.git baselines/PiToMe

# ATC (ECCV 2024)
git clone https://github.com/JoakimHauworWorWorrum/ATC.git baselines/ATC
```

See [`baselines/README.md`](baselines/README.md) for details.

---

## Quick Start

### Classification (DeiT-S, training-free)

```python
import torch
from models.sr_tome import create_sr_tome_v3

# Wrap a pretrained DeiT-S with S³-Merge (zero training, inference-time only)
model = create_sr_tome_v3(
    model_name='deit_small_patch16_224',
    target_tokens=84,      # Keep 84 out of 196 tokens (~43%)
    late_start=8,           # Merge at layer 8 of 12
    window_size=7,          # 7×7 spatial matching windows
    prop_attn=True,         # Proportional attention (log-size bias)
    pretrained=True,
).cuda().eval()

x = torch.randn(32, 3, 224, 224).cuda()
with torch.no_grad():
    logits = model(x)  # (32, 1000)
```

### Dense Prediction (detection/segmentation)

```python
from models.sr_tome_downstream import create_sr_tome_downstream

# Wrap any timm ViT for dense tasks — includes unmerge/restore mapping
backbone = create_sr_tome_downstream(
    model_name='deit_small_patch16_224',
    target_tokens=84,
    late_start=8,
    window_size=7,
    pretrained=True,
).cuda().eval()

x = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    features = backbone(x)  # (1, C, H, W) — dense grid for FPN/decoder
```

### Triton-Fused Merge (advanced)

```python
from models.triton_merge import sr_tome_merge_triton
# Drop-in replacement for sr_tome_merge_exact with 1.23× lower overhead
```

---

## Reproducing Paper Results

All experiments can be reproduced with the scripts below. Results match those reported in the paper when using the same hardware (H100 NVL) and Docker environment.

```bash
# Table 2 — Main ImageNet-1K comparison (DeiT-S, 50k val, 5 seeds)
python eval_imagenet_baselines.py --mode full

# Table 3 — Throughput scaling across batch sizes
python benchmark_batch_scaling.py

# Table 4 — Overhead decomposition
python profile_overhead.py

# Table 8 — ADE20K UPerNet-ViT-B backbone throughput (requires mmseg)
python eval_ade20k_upernet.py

# Table 9 — COCO ViTDet-B end-to-end evaluation (requires detectron2)
python eval_coco_vitdet.py

# Figure 4 — Accuracy–throughput Pareto frontier
python generate_pareto_fixed.py

# Figure 5 — Latency/throughput scaling vs batch size
python generate_batch_scaling_pa.py

# Figure 6 — CUDA kernel time breakdown (168 launches vs 10)
python scripts/profile_kernel_breakdown.py --batch_sizes 1 32 --figure_out artifacts/kernel_profiling.pdf

# Figure 7 — Amdahl decomposition (ViTDet backbone vs decoder)
python plot_amdahl_decomposition.py

# Supplementary Table S13 — Controlled SDPA backend experiment
python benchmark_sdpa_control.py

# Supplementary Table S5 / S14 — Cross-hardware validation (H100 / L40S / Orin)
python benchmark_cross_hardware.py
```

See [`scripts/reproduce_tables.sh`](scripts/reproduce_tables.sh) for a one-click script.

---

## Method Overview

**S³-Merge** modifies two axes of progressive token merging:

1. **Single-shot schedule**: Merge tokens once at layer $L_s$ instead of every layer → 9× lower overhead
2. **Windowed spatial matching**: Match within $w × w$ windows instead of globally → bounded displacement (Proposition 1)

Combined with **proportional attention** (log-size bias) and **Safe Mode** (predictive crossover $B^*$), this yields:
- Near-baseline B=1 latency (2.92 ms vs 2.40 ms)
- Up to 1.26× throughput at B=32
- Direct backbone acceleration for dense tasks (+21% on ADE20K)

---

## File Structure

```
s3-merge/
├── models/
│   ├── sr_tome.py              # Core S³-Merge for classification
│   ├── sr_tome_downstream.py   # S³-Merge with unmerge for dense tasks
│   └── triton_merge.py         # Fused Triton kernels (1.23× lower overhead)
├── baselines/
│   ├── __init__.py             # Unified API: apply_method(model, 'pitome', ...)
│   └── dsm_impl.py             # DSM re-implementation (no official code)
├── configs/                    # YAML configurations
├── eval_imagenet_baselines.py  # Unified 50k ImageNet benchmark
├── eval_ade20k_upernet.py      # ADE20K segmentation
├── eval_coco_vitdet.py         # COCO detection (ViTDet)
├── profile_overhead.py         # Latency decomposition
├── Dockerfile                  # Full repro environment
└── requirements.txt
```

---

## Citation

```
Paper under double-blind review.
```

## License

MIT License — see [LICENSE](LICENSE).
