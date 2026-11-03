# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
LABEL maintainer="nesli <neslierdogmus@iyte.edu.tr>"

ENV PIP_DEFAULT_TIMEOUT=300

# System deps
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        libgl1 \
        gcc \
        g++ \
        tmux \
        libglib2.0-0 \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 1: torch nightly (works)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --pre --no-cache-dir \
        torch==2.12.0.dev20260408+cu128 \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        --timeout 300 --retries 10

# Step 1b: matching torchvision nightly (timm needs it at runtime)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --pre --no-cache-dir --no-deps \
        torchvision==0.27.0.dev20260407+cu128 \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        --timeout 300 --retries 10

# Step 2: timm WITHOUT its deps, so it can't pull torchvision or stable torch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-deps timm==1.0.12

# Step 3: everything else, PyPI only, with torch frozen by constraints
COPY requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt -c /tmp/constraints.txt \
        --timeout 300 --retries 10

# Pin the HF Hub cache to a fixed, home-independent path BEFORE the
# pre-fetch below writes anything. Without this, cache resolution falls
# through to ~/.cache/huggingface, which depends on $HOME -- fine at
# `docker build` time (root, $HOME=/root), but Apptainer/Singularity on the
# target cluster runs as the invoking (non-root) user and bind-mounts that
# user's real home directory by default, shadowing whatever $HOME was at
# build time. The baked-in weights would then sit untouched at
# /root/.cache/huggingface while runtime code looks in the real user's
# (empty) home dir instead -- silently defeating this entire fix. Pinning
# HF_HOME to an absolute path outside any user's home sidesteps that.
ENV HF_HOME=/opt/hf_cache

# Step 4: pre-download pretrained encoder weights so they're baked into the
# image layer, using the exact parameters the model factory actually uses
# at runtime (src/models/factory.py::_build_encoder) -- not placeholders.
# M2 always calls ResNet50Encoder(pretrained=True, in_channels=3). M3
# always calls SwinV2TEncoder(in_channels=3, img_size=config.input_size),
# and every task's input_size (FC 512x512, FMD 374x388, FOE 478x448) pads
# up to (512,512) inside SwinV2TEncoder (stride=256) -- confirmed by
# computing the padding for all three tasks' base.yaml, not assumed.
# M1 is always pretrained=False (scratch) and never touches the network,
# so it needs no pre-fetch entry here.
#
# Must come AFTER Step 3, not right after Step 2: timm.create_model()
# transitively needs numpy (timm/layers/cond_conv2d.py imports it), and
# numpy is deliberately deferred to Step 3's requirements.txt (Step 2
# installs timm with --no-deps specifically so it can't pull its own,
# unpinned copy). Placing this block between Step 2 and Step 3 -- as an
# earlier version of this file did -- fails the build with
# "ModuleNotFoundError: No module named 'numpy'" at this exact RUN step.
RUN python3 -c "\
import timm; \
timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=(1,2,3,4), in_chans=3); \
timm.create_model('swinv2_tiny_window8_256', pretrained=True, features_only=True, out_indices=(0,1,2,3), img_size=(512,512))"

# From here on, never attempt a network call for these weights -- fail fast
# from the local cache instead of hanging on an air-gapped node waiting on
# DNS/connection timeouts. Must come after the download step above, not
# before, or the download itself would refuse to run. Runtime (Apptainer on
# the cluster) inherits this from the image; if that's ever not the case,
# it must be set explicitly in the `apptainer exec` invocation instead.
ENV HF_HUB_OFFLINE=1

ENV PROJECT_DIR=/workspace
WORKDIR $PROJECT_DIR
CMD ["/bin/bash"]
