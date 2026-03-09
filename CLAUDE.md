# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**worker-vllm** is a RunPod Serverless worker that deploys OpenAI-compatible LLM inference endpoints powered by vLLM. It runs on RunPod's serverless infrastructure with CUDA 12.9 and vLLM 0.16.0 (with FlashInfer).

## Build & Run

This project runs inside Docker containers on RunPod infrastructure. There is no local test suite or linter configured.

```bash
# Build pre-built image (model downloads at runtime)
docker build -t worker-vllm .

# Build with model baked in
docker build -t worker-vllm --build-arg MODEL_NAME="org/model" --build-arg BASE_PATH="/models" .

# Build with private/gated model
DOCKER_BUILDKIT=1 docker build -t worker-vllm --secret id=HF_TOKEN --build-arg MODEL_NAME="org/model" .

# Build with vLLM nightly
docker build -t worker-vllm --build-arg VLLM_NIGHTLY=true .
```

Python dependencies are in `builder/requirements.txt`. Docker bake configuration is in `docker-bake.hcl`.

## Architecture

### Request Flow
```
RunPod Request → handler.py → JobInput → Engine Selection → vLLM Generation → Streaming Response
```

### Key Source Files (`src/`)

- **`handler.py`** — Entry point. Async handler for RunPod serverless. Routes to `vLLMEngine` or `OpenAIvLLMEngine` based on `job_input.openai_route`. CUDA errors trigger `sys.exit(1)` to restart the worker.

- **`engine.py`** — Two engine classes:
  - `vLLMEngine`: Base class wrapping `AsyncLLMEngine`. Handles tokenizer init, dynamic batch streaming, and token counting.
  - `OpenAIvLLMEngine(vLLMEngine)`: OpenAI-compatible wrapper. **Defers initialization to first request** (not startup) to avoid event loop mismatch with RunPod's serverless handler. Supports LoRA adapters, chat/completion/models endpoints.

- **`engine_args.py`** — Configuration management. Auto-discovers all `AsyncEngineArgs` fields from UPPERCASED env vars (e.g., `MAX_MODEL_LEN` → `max_model_len`). Config hierarchy: `DEFAULT_ARGS` → env vars (auto-discovered) → env aliases (`MODEL_NAME` → `model`) → `/local_model_args.json` (baked models). Also handles speculative decoding config, multi-GPU tensor parallelism detection, and deprecated env var migration.

- **`utils.py`** — `JobInput` (request parsing with defaults), `BatchSize` (dynamic batching with growth factor), `DummyRequest` (stub for vLLM's raw_request parameter), error response formatting.

- **`tokenizer.py`** — `TokenizerWrapper` around HuggingFace `AutoTokenizer`. Applies chat templates (custom via `CUSTOM_CHAT_TEMPLATE` env var, or model-provided). Mistral models skip this wrapper and use vLLM's native tokenizer.

- **`constants.py`** — Default values for batch size, concurrency, etc.

- **`download_model.py`** — Downloads model during Docker build for baked images. Saves args to `/local_model_args.json`.

### Dual API Design

The same codebase serves two API modes:

1. **OpenAI-compatible** (`/openai/v1/`): Chat completions, text completions, models list. Uses vLLM's `OpenAIServingChat` and `OpenAIServingCompletion`.
2. **Native vLLM**: Direct `prompt` or `messages` with `sampling_params`. Custom streaming with configurable batch sizes.

### Configuration System

All configuration is via environment variables. Key patterns:
- Any `AsyncEngineArgs` field works as an UPPERCASED env var (auto-discovered)
- Backward-compat aliases: `MODEL_NAME`→`model`, `MODEL_REVISION`→`revision`, `TOKENIZER_NAME`→`tokenizer`
- Boolean env vars accept: `true`/`false`, `1`/`0`, `yes`/`no`
- Complex types (dicts, lists) are parsed as JSON strings
- `.runpod/hub.json` defines 80+ env vars for the RunPod console UI — **must stay in sync with code defaults**

### Important Design Decisions

- **Deferred OpenAI engine init**: `OpenAIvLLMEngine` initializes serving engines on first request, not at startup, because `asyncio.run()` creates a temporary event loop that conflicts with RunPod's handler loop.
- **Dynamic batching**: Streaming uses `min_batch_size * growth_factor^n` up to `max_batch_size` for adaptive throughput.
- **Multi-GPU auto-detection**: `device_count() > 1` automatically sets `tensor_parallel_size` and overrides `max_parallel_loading_workers`.
- **Guard against subprocess re-init**: `handler.py` checks `__name__ == "__main__"` or `MainProcess` to prevent vLLM worker subprocesses from re-initializing engines.

## CI/CD

- **Dev builds**: All PRs trigger `dev-refs-pull-<PR#>-merge` Docker images (`.github/workflows/dev.yml`)
- **Release builds**: Git tags (`v*.*.*`) trigger versioned images + GitHub releases (`.github/workflows/release.yml`)
- **No automatic builds on main** — main is a staging area
- Release process: `git checkout main && git tag v2.8.0 && git push origin v2.8.0`
- Version format: `vMAJOR.MINOR.PATCH` (semantic versioning)

## Key Env Vars Reference

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `facebook/opt-125m` | HuggingFace model ID |
| `MAX_MODEL_LEN` | auto | Maximum context length |
| `GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of GPU memory |
| `TENSOR_PARALLEL_SIZE` | `1` (auto if multi-GPU) | Number of GPUs |
| `MAX_CONCURRENCY` | `30` | Max concurrent requests |
| `QUANTIZATION` | — | `awq`, `gptq`, `bitsandbytes` |
| `CUSTOM_CHAT_TEMPLATE` | — | Jinja2 chat template |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | — | Override model name in API |
| `RAW_OPENAI_OUTPUT` | `1` | Stream raw SSE strings vs parsed JSON |
| `ENABLE_AUTO_TOOL_CHOICE` | `false` | Enable automatic tool selection |
| `SPECULATIVE_CONFIG` | — | Full JSON for speculative decoding |
