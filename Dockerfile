# syntax=docker/dockerfile:1.7
#
# FitCheck VTON worker for RunPod Serverless.
#
#   # lean image, weights from RunPod cached models / network volume:
#   docker build -t fitcheck-vton .
#
#   # self-contained image with weights baked in (fastest, most predictable cold start):
#   docker build -t fitcheck-vton:qwen --build-arg BAKE_MODELS=1 --build-arg VTON_ENGINE=qwen_edit \
#       --secret id=hf_token,env=HF_TOKEN .
#
# Layer order = change frequency: CUDA/torch -> deps -> weights -> code, so a
# code push only re-uploads a few KB instead of tens of GB of weights.

ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    HF_XET_HIGH_PERFORMANCE=1 \
    MEDIAPIPE_DIR=/models/mediapipe \
    TORCHINDUCTOR_CACHE_DIR=/models/inductor-cache \
    RUNPOD_DEBUG_LEVEL=INFO \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MediaPipe links against GL/EGL even when running on CPU.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Optional SVDQuant INT4/FP4 kernels (24 GB cards / max throughput).
ARG INSTALL_NUNCHAKU=0
ARG NUNCHAKU_VERSION=1.2.1
RUN if [ "$INSTALL_NUNCHAKU" = "1" ]; then \
      PY=$(python -c 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")'); \
      TV=$(python -c 'import torch;print(".".join(torch.__version__.split("+")[0].split(".")[:2]))'); \
      pip install "https://github.com/nunchaku-ai/nunchaku/releases/download/v${NUNCHAKU_VERSION}/nunchaku-${NUNCHAKU_VERSION}+cu12.8torch${TV}-${PY}-${PY}-linux_x86_64.whl"; \
    fi

# Masker models: 25 MB, Apache-2.0, always baked.
RUN mkdir -p "$MEDIAPIPE_DIR" \
 && curl -fsSL -o "$MEDIAPIPE_DIR/pose_landmarker_full.task" \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task \
 && curl -fsSL -o "$MEDIAPIPE_DIR/selfie_multiclass_256x256.tflite" \
      https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite

# Diffusion weights (optional bake). Only the files the engine actually loads
# are fetched, e.g. CatVTON-FLUX skips FLUX-Fill's ~10 GB of text encoders.
ARG VTON_ENGINE=qwen_edit
ARG BAKE_MODELS=0
COPY src/vton/config.py /app/src/vton/config.py
COPY scripts/fetch_models.py /app/scripts/fetch_models.py
RUN --mount=type=secret,id=hf_token,required=false \
    if [ "$BAKE_MODELS" = "1" ]; then \
      HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" python scripts/fetch_models.py --engine "$VTON_ENGINE"; \
    fi

COPY src/ /app/src/
COPY handler.py /app/handler.py

ENV VTON_ENGINE=${VTON_ENGINE}
CMD ["python", "-u", "/app/handler.py"]
