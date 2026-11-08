# Table 7, ViT-B/16 row: explicit PA-off configuration

This directory reconstructs the configuration of the ViT-B/16 row of paper Table 7 (footnote b; protocol in Supplementary Section S1.2) with the current implementation.

| Setting | Value |
|---|---|
| Model | timm `vit_base_patch16_224` (pretrained), 224² |
| Merge | after block 9 (`late_start=8`), T=136, 7×7 windows, so **136 patch tokens and 137 with [CLS]** |
| Proportional attention | **off**: `merge.prop_attn` is required in the YAML and must be `false`; asserted after patching |
| Precision | AMP/FP16 (`torch.cuda.amp.autocast`) for timing and accuracy |
| Timing | B = 1 and 64; 30 warm-up + 200 timed forwards; CUDA synchronization before and after the timed loop; one invocation |
| Accuracy | ImageNet-1K validation (50,000 images, no shuffle), Resize 256 → CenterCrop 224 → ImageNet mean/std, batch 64 |
| Reference hardware | NVIDIA H100 NVL |

```bash
python r3/table7_vitb16/run_table7_vitb16.py --print-config        # resolve and print the configuration (CPU, no weights)
python r3/table7_vitb16/run_table7_vitb16.py --output-dir results/table7_vitb16 --imagenet-cache /path/to/hf_cache
python r3/table7_vitb16/run_table7_vitb16.py --skip-accuracy        # timing only
```

**Outputs.** `table7_vitb16.json` holds the resolved configuration, environment, per-arm latencies, Top-1 and summary speed-ups. `table7_vitb16.csv` is also written.

**The paper row and this script.**
- **Paper values:** 80.92% vs 81.08% Top-1, 0.90× at B=1, 1.02× at B=64.
- **Their origin:** a historical run with an earlier implementation that had no proportional attention. The original call passed `target_tokens=137`, which gives the same K=34 per window.
- **What this script does:** it rebuilds that configuration with the current implementation.
- **Not guaranteed:** exact numerical identity, because the implementation and software environment have evolved and latency depends on the hardware.
- **Do not combine:** this campaign is distinct from the ViT-B merge-layer sweep (Supplementary Table S2).
