# CUDA runtime base (no dev tools needed)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_VERBOSITY=error \
    PYTORCH_ALLOC_CONF=expandable_segments:True

# OS deps (jq is handy for tiny unit-data prep), git-lfs for HF
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv git git-lfs jq && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# Copy only the lock/requirements first for layer caching
COPY requirements.txt ./requirements.txt

# Install PyTorch that matches CUDA 12.1 first
RUN python -m pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio

# Then everything else from your requirements
RUN pip install -r requirements.txt

# Now copy the rest of the repo
COPY . /workspace

# Make sure HF cache dir exists
RUN mkdir -p ${HF_HOME}
