# PETR / PETRv2 (camera-based 3D detection) inference image.
# Stack follows PETR's install.md: torch 1.9.0+cu111, mmcv-full 1.4.0, mmdet 2.24.1,
# mmsegmentation 0.20.2, mmdetection3d v0.17.1.
# RTX 40xx (sm_89) runs the sm_86 SASS shipped in the cu111 wheels; CUDA extensions
# are compiled with TORCH_CUDA_ARCH_LIST=8.6 because nvcc 11.1 does not know 8.9.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip python-is-python3 \
        build-essential git wget ca-certificates ninja-build \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Old build tooling: setuptools>=60 breaks the legacy `setup.py develop` builds.
RUN python -m pip install -q "pip==23.0.1" "setuptools==59.5.0" "wheel==0.40.0"

RUN pip install -q torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html

# numpy<1.20 is required by mmdet3d 0.17 (np.float / np.int); numba==0.48.0 as pinned by mmdet3d (uses numba.errors).
RUN pip install -q \
        "numpy==1.19.5" \
        "numba==0.48.0" "llvmlite==0.31.0" \
        "yapf==0.40.1" \
        "opencv-python-headless==4.5.5.64" \
        "matplotlib==3.5.3" \
        "scipy==1.6.3" \
        "scikit-image==0.19.3" \
        "scikit-learn==1.0.2" \
        "shapely==1.8.5" \
        "pandas==1.3.5" \
        "networkx==2.2" \
        "trimesh==2.35.39" \
        "plyfile==0.7.4" \
        "pycocotools==2.0.7" \
        "terminaltables==3.1.10" \
        "tensorboard==2.11.2" \
        "protobuf==3.20.3" \
        "einops==0.6.1" \
        "pyquaternion==0.9.9" \
        "descartes==1.1.0" \
        "tqdm==4.66.1" \
        "gdown==5.2.0" \
        "timm==0.6.13" \
    && pip install -q --no-deps "nuscenes-devkit==1.1.9" "lyft_dataset_sdk==0.0.8" \
    && pip install -q "cachetools" "fire" "Pillow==9.5.0"

RUN pip install -q mmcv-full==1.4.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

WORKDIR /opt
# mmdetection v2.24.1
RUN git clone -q https://github.com/open-mmlab/mmdetection.git \
    && cd mmdetection && git checkout -q 73b4e65a6a30435ef6a35f405e3474a4d9cfb234 \
    && pip install -q --no-deps -e .

RUN pip install -q --no-deps mmsegmentation==0.20.2 \
    && pip install -q "prettytable==3.9.0" "wcwidth==0.2.13"

# mmdetection3d v0.17.1 (builds CUDA ops; no GPU needed at build time thanks to FORCE_CUDA=1)
RUN git clone -q https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d && git checkout -q f1107977dfd26155fc1f83779ee6535d2468f449 \
    && pip install -q --no-deps -e .

# PETR (pinned commit), with mmdetection3d symlinked inside as its configs expect.
RUN git clone -q https://github.com/megvii-research/PETR.git \
    && cd PETR && git checkout -q f7525f93467a33707ef401c587a52d5e7b34de74 \
    && ln -s /opt/mmdetection3d /opt/PETR/mmdetection3d

ENV PETR_ROOT=/opt/PETR \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /workspace
