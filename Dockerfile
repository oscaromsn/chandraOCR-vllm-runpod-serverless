FROM nvidia/cuda:12.4.1-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip poppler-utils

RUN ldconfig /usr/local/cuda-12.4/compat/

# Install vLLM with FlashInfer - use CUDA 12.4 PyTorch wheels (compatible with RTX 4090 RunPod drivers)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm[flashinfer]==0.16.0" --extra-index-url https://download.pytorch.org/whl/cu124



# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Suppress Ray metrics agent warnings (not needed in containerized environments)
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    # Prevent rayon thread pool panic in containers where ulimit -u < nproc
    # (tokenizers uses Rust's rayon which tries to spawn threads = CPU cores)
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
    pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly && \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    pip install git+https://github.com/huggingface/transformers.git; \
fi

# ChandraOCR / RTX 4090 optimal defaults (users can override via env vars)
ENV MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.95 \
    TRUST_REMOTE_CODE=true \
    DTYPE=half \
    KV_CACHE_DTYPE=fp8 \
    MAX_NUM_SEQS=16 \
    ENABLE_PREFIX_CACHING=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LIMIT_MM_PER_PROMPT="image=1" \
    MAX_CONCURRENCY=16 \
    DEFAULT_BATCH_SIZE=10 \
    MM_PROCESSOR_KWARGS='{"max_pixels": 802816, "min_pixels": 3136}' \
    MAX_NUM_BATCHED_TOKENS=8192

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
