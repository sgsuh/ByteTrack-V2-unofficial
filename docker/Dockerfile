# ByteTrackV2 (unofficial) runtime image.
# CUDA 12.1 supports Ada GPUs (sm_89, e.g. RTX 40xx).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install --no-deps nuscenes-devkit==1.1.11 \
    # motmetrics 1.1.3 (required by the nuScenes tracking eval) predates Python 3.10/3.11
    && sed -i \
        -e 's/from collections import OrderedDict, Iterable/from collections import OrderedDict\nfrom collections.abc import Iterable/' \
        -e 's/inspect.getargspec(/inspect.getfullargspec(/' \
        "$(python -c 'import site; print(site.getsitepackages()[0])')/motmetrics/metrics.py" \
    # ... and pandas>=1.0 (MultiIndex `labels` was renamed to `codes`)
    && sed -i 's/labels=\[\[\],\[\]\]/codes=[[],[]]/' \
        "$(python -c 'import site; print(site.getsitepackages()[0])')/motmetrics/mot.py"

WORKDIR /workspace
ENV PYTHONPATH=/workspace
