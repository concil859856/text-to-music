# syntax=docker/dockerfile:1.7
#
# text-to-music — ACE-Step v1-3.5B music generation HTTP API for the
# Vocence /studio/ops fleet manager. Runs on NVIDIA RTX 4090 (24 GB).
#
# Build:
#   docker build -t docker.io/<ns>/text-to-music:latest .
# Run (on rented box):
#   docker run -d --gpus all --restart=unless-stopped -p 8115:8115 \
#     -e MUSIC_API_KEY=<key> \
#     -v ace_checkpoints:/app/checkpoints \
#     -v ace_outputs:/app/api_outputs \
#     docker.io/<ns>/text-to-music:latest
#
# Replaces the original Dockerfile that cloned from GitHub on every build
# and ran the Gradio GUI. This one COPIES local source (so PR-merge-to-main
# builds the committed code), runs api.py (the JSON HTTP API the Vocence
# backend talks to), and binds 0.0.0.0:8115 by default.

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/cache/hf \
    HOST=0.0.0.0 \
    PORT=8115

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev \
        build-essential git curl wget ca-certificates \
        libsndfile1 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install deps first (slow torch/peft/etc. layer caches across iterations).
WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --upgrade pip \
    && pip3 install hf_transfer peft \
    && pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# Copy the actual project source.
COPY . .

# Install the acestep package itself (it's a pip-installable project).
RUN pip3 install --no-deps -e .

# Persistent volumes for the ~7 GB ACE-Step checkpoints + output WAVs +
# logs. Mount host directories here so the model isn't re-downloaded on
# every container recreation.
VOLUME ["/app/checkpoints", "/app/api_outputs", "/cache/hf"]

EXPOSE 8115

# HEALTHCHECK targets /health (always open) so Docker doesn't need bearer.
# 300 s start-period covers first-boot model download (~7 GB) + load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8115}/health || exit 1

# api.py is the JSON HTTP API that the Vocence backend calls. PORT comes
# from env (default 8115); use the click default lambda we added in api.py.
CMD ["python3", "api.py"]
