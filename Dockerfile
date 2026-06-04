# SR-ToMe Downstream Docker Environment
#
# Provides: detectron2 (COCO/ViTDet) + mmseg (ADE20K/UPerNet) with nvcc
#
# Build:
#   docker build -f Dockerfile.srtome_cv -t srtome-cv:cu121 .
#
# Run:
#   docker run --gpus all --shm-size=16g -it \
#     -v /path/to/imagenet:/data/imagenet \
#     -v $PWD:/workspace \
#     srtome-cv:cu121
#
# Note: Using PyTorch 2.1.2 for better prebuilt wheel compatibility

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install -U pip setuptools wheel

WORKDIR /workspace

# ===== Verify torch is available =====
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# ===== Detectron2 dependencies first =====
RUN python -m pip install 'fvcore>=0.1.5' 'iopath>=0.1.9' 'hydra-core>=1.1' 'omegaconf>=2.1' cloudpickle pycocotools

# ===== Detectron2 (build from source with --no-build-isolation) =====
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# ===== OpenMMLab Stack =====
RUN python -m pip install -U openmim
RUN mim install mmengine

# Prebuilt mmcv wheel for cu121/torch2.1.x (avoids compilation)
RUN python -m pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# mmseg from source (python-only once mmcv is installed)
RUN git clone --depth 1 https://github.com/open-mmlab/mmsegmentation.git /opt/mmseg && \
    python -m pip install -e /opt/mmseg

# ===== Additional dependencies =====
# Pin numpy<2 for compatibility with PyTorch 2.1.2 (compiled with numpy 1.x)
RUN python -m pip install 'numpy<2' timm datasets scikit-learn prettytable ftfy einops

# ===== Sanity check =====
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
RUN python -c "import detectron2; print('detectron2 OK')"
RUN python -c "import mmcv, mmseg; print(f'mmcv: {mmcv.__version__}, mmseg: {mmseg.__version__}')"

# Default command
CMD ["/bin/bash"]
