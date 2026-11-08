# Third-party software

The code in this repository is released under the MIT licence (`LICENSE`). It does **not** redistribute third-party source code. The projects below are installed or downloaded separately by the user, under their own licences.

| Project | Used by | How it is used | Licence (as published by the project) |
|---|---|---|---|
| [ToMe](https://github.com/facebookresearch/ToMe), commit `af95e4b1befa172dadccd8c81e223b10090f9579` | `r3/unified_tome/` (paper Table 1 baseline) | imported as the installed `tome` package and applied through `tome.patch.timm`; not modified and not copied | CC BY-NC 4.0 (non-commercial) |
| [PiToMe](https://github.com/hchautran/PiToMe), commit `550b5deed94aadfeac28bfbe381d87e672044a40` | `eval_imagenet_baselines.py` (original-campaign harness, Supplementary Table S23) | optional external clone in `baselines/PiToMe`; the compatibility adapter used for the paper is **not included** in this release | CC BY-NC 4.0 (non-commercial) |
| [ATC](https://github.com/JoakimHaurum/ATC) | `eval_imagenet_baselines.py` (Supplementary Table S23) | optional external clone in `baselines/ATC` | MIT |
| [timm](https://github.com/huggingface/pytorch-image-models) 1.0.24 | all model code | dependency (`pip`) | Apache-2.0 |
| [detectron2](https://github.com/facebookresearch/detectron2), commit `fd27788985af0f4ca800bca563acdb700bb890e2` | `models/adaptive_k.py` (packed ViTDet path: `get_rel_pos`), `eval_coco_vitdet.py` | dependency | Apache-2.0 |
| [PyTorch](https://pytorch.org) 2.1.2, torchvision 0.16.2 | everything | dependency | BSD-3-Clause |
| [Hugging Face `datasets`](https://github.com/huggingface/datasets) 4.5.0 | ImageNet-1K loading | dependency | Apache-2.0 |

**Model weights and datasets.**
- Pretrained weights are downloaded by timm (for example `deit_small_patch16_224.fb_in1k`, `vit_base_patch16_224`) under the terms stated on their model cards.
- ImageNet-1K (`ILSVRC/imagenet-1k`) must be obtained by the user under its own terms of access.

**Licence restrictions.** The ToMe and PiToMe licences restrict commercial use of those projects. Running the baseline comparisons is therefore subject to their terms. The MIT licence of this repository does not change them.
