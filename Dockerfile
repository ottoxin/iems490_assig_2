FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_VERBOSITY=error \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true

RUN apt-get update && apt-get install -y python3.10 python3-pip git git-lfs && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    pip install --upgrade pip

# Torch matching CUDA 12.1
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Core libs (pinned)
RUN pip install \
    transformers==4.57.1 tokenizers==0.22.1 peft==0.12.0 \
    datasets==2.19.0 scikit-learn==1.5.2 accelerate==0.33.0

WORKDIR /workspace
COPY . /workspace

RUN mkdir -p $HF_HOME && git lfs install

# Default: show unit test help
CMD ["python", "src/unit_test_lora_qwen.py", "--help"]
