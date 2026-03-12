<div align="center">

# ChandraOCR — vLLM Serverless Endpoint for RunPod

Deploy a blazing-fast OCR endpoint powered by [ChandraOCR](https://huggingface.co/datalab-to/chandra) and [vLLM](https://github.com/vllm-project/vllm) on RunPod Serverless. Accepts PDF and image inputs and returns structured markdown with tables, LaTeX math, and preserved formatting.

Built on the [worker-vllm](https://github.com/runpod-workers/worker-vllm) framework with full OpenAI API compatibility.

</div>

## Table of Contents

- [Features](#features)
- [Setting up the Serverless Worker](#setting-up-the-serverless-worker)
  - [Option 1: Deploy Using Pre-Built Docker Image [Recommended]](#option-1-deploy-using-pre-built-docker-image-recommended)
  - [Option 2: Build Docker Image with Model Inside](#option-2-build-docker-image-with-model-inside)
- [Configuration](#configuration)
- [Usage: ChandraOCR Input](#usage-chandraocr-input)
  - [PDF Input](#pdf-input)
  - [Image Input](#image-input)
  - [Multiple Images](#multiple-images)
  - [PDF from URL](#pdf-from-url)
- [Usage: OpenAI Compatibility](#usage-openai-compatibility)
  - [Modifying your OpenAI Codebase](#modifying-your-openai-codebase)
  - [Chat Completions](#chat-completions)
  - [Getting Available Models](#getting-available-models)
- [Usage: Standard (Non-OpenAI)](#usage-standard-non-openai)
  - [Request Input Parameters](#request-input-parameters)
  - [Sampling Parameters](#sampling-parameters)
- [Architecture](#architecture)

## Features

- **OCR-optimized**: Built-in PDF and image preprocessing for ChandraOCR (Qwen3-VL based)
- **Multiple input formats**: Base64 PDF, base64 images, PDF URLs, or raw prompts
- **Structured output**: Markdown with tables, LaTeX math, headings, and lists
- **Page range selection**: Process specific pages from multi-page PDFs
- **OpenAI-compatible API**: Drop-in replacement using `/openai/v1/` routes
- **Native vLLM API**: Direct prompt/messages with full sampling parameter control
- **Streaming support**: Both OpenAI SSE and native batch streaming
- **Multi-GPU auto-detection**: Automatically configures tensor parallelism
- **Dynamic batching**: Adaptive token batching (1 → 3 → 9 → 27 → 50) for throughput

## Setting up the Serverless Worker

### Option 1: Deploy Using Pre-Built Docker Image [Recommended]

**Docker Image**: `runpod/worker-v1-vllm:<version>`

- **Available Versions**: See [GitHub Releases](https://github.com/runpod-workers/worker-vllm/releases)
- **CUDA Compatibility**: Built with CUDA 12.4.1 (compatible with RTX 4090 and newer)

Set `MODEL_NAME=datalab-to/chandra` (or your preferred model) as an environment variable when creating the endpoint.

### Option 2: Build Docker Image with Model Inside

Build an image with the ChandraOCR model baked in for faster cold starts.

#### Prerequisites

- Docker

#### Build Commands

```bash
# Build with ChandraOCR model baked in (recommended)
docker build -t chandraocr-worker --build-arg MODEL_NAME="datalab-to/chandra" --build-arg BASE_PATH="/models" .

# Build pre-built image (model downloads at runtime)
docker build -t chandraocr-worker .

# Build with a private/gated model
DOCKER_BUILDKIT=1 docker build -t chandraocr-worker --secret id=HF_TOKEN --build-arg MODEL_NAME="datalab-to/chandra" .

# Build with vLLM nightly (latest unreleased features)
docker build -t chandraocr-worker --build-arg VLLM_NIGHTLY=true --build-arg MODEL_NAME="datalab-to/chandra" --build-arg BASE_PATH="/models" .
```

#### Build Arguments

- **Required**
  - `MODEL_NAME`: Hugging Face model ID (default: `datalab-to/chandra`)
- **Optional**
  - `MODEL_REVISION`: Model revision to load (default: `main`)
  - `BASE_PATH`: Storage directory for model weights (default: `/runpod-volume`). Set to `/models` when baking the model into the image.
  - `QUANTIZATION`: Quantization method (`awq`, `gptq`, `bitsandbytes`)
  - `TOKENIZER_NAME`: Custom tokenizer repository (default: uses model's tokenizer)
  - `TOKENIZER_REVISION`: Tokenizer revision (default: `main`)
  - `VLLM_NIGHTLY`: Set to `true` to use the latest nightly vLLM build (default: `false`)

#### (Optional) Including Hugging Face Token

For private or gated models, pass your token as a Docker secret:

```bash
export DOCKER_BUILDKIT=1
export HF_TOKEN="your_token_here"
docker build -t chandraocr-worker --secret id=HF_TOKEN --build-arg MODEL_NAME="datalab-to/chandra" .
```

## Configuration

Configure the worker using environment variables. The Dockerfile sets ChandraOCR-optimized defaults:

| Environment Variable         | Default (ChandraOCR)   | Description                                              |
| ---------------------------- | ---------------------- | -------------------------------------------------------- |
| `MODEL_NAME`                 | `datalab-to/chandra`   | Hugging Face model ID                                    |
| `MAX_MODEL_LEN`              | `4096`                 | Maximum context length                                   |
| `GPU_MEMORY_UTILIZATION`     | `0.95`                 | Fraction of GPU memory to use                            |
| `TENSOR_PARALLEL_SIZE`       | `1` (auto if multi-GPU)| Number of GPUs for tensor parallelism                    |
| `MAX_NUM_SEQS`               | `16`                   | Maximum sequences per iteration                          |
| `MAX_CONCURRENCY`            | `16`                   | Maximum concurrent requests                              |
| `DEFAULT_BATCH_SIZE`         | `10`                   | Maximum tokens per streaming batch                       |
| `DTYPE`                      | `half`                 | Model data type (FP16)                                   |
| `TRUST_REMOTE_CODE`          | `true`                 | Required for ChandraOCR model                            |
| `ENABLE_PREFIX_CACHING`      | `true`                 | Cache common prompt prefixes                             |
| `LIMIT_MM_PER_PROMPT`        | `image=1`              | Multimodal inputs per prompt                             |
| `MAX_NUM_BATCHED_TOKENS`     | `8192`                 | Maximum tokens batched together                          |
| `MM_PROCESSOR_KWARGS`        | (see below)            | Multimodal processor settings                            |
| `HF_TOKEN`                   |                        | Hugging Face token for gated models                      |
| `QUANTIZATION`               |                        | Quantization method (`awq`, `gptq`, `bitsandbytes`)     |
| `CUSTOM_CHAT_TEMPLATE`       |                        | Jinja2 chat template override                           |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` |                 | Override model name in OpenAI API responses              |
| `RAW_OPENAI_OUTPUT`          | `1`                    | Stream raw SSE strings vs parsed JSON                    |
| `ENABLE_AUTO_TOOL_CHOICE`    | `false`                | Enable automatic tool selection                          |
| `TOOL_CALL_PARSER`           |                        | Parser for tool calls (`mistral`, `hermes`, etc.)        |
| `SPECULATIVE_CONFIG`         |                        | Full JSON config for speculative decoding                |

**Pass any vLLM engine arg** by setting an UPPERCASED environment variable matching the field name:

| Environment Variable      | vLLM Engine Arg          | Example Value |
| ------------------------- | ------------------------ | ------------- |
| `MAX_MODEL_LEN`           | `max_model_len`          | `4096`        |
| `ENFORCE_EAGER`           | `enforce_eager`          | `true`        |
| `ENABLE_CHUNKED_PREFILL`  | `enable_chunked_prefill` | `true`        |

Any env var whose name matches a valid `AsyncEngineArgs` field (uppercased) is applied automatically. Boolean env vars accept `true`/`false`, `1`/`0`, `yes`/`no`. Complex types (dicts, lists) are parsed as JSON strings.

For the complete environment variable reference: **[docs/configuration.md](docs/configuration.md)**

## Usage: ChandraOCR Input

ChandraOCR accepts PDF and image inputs directly. The worker automatically detects OCR requests and processes them through the ChandraOCR pipeline.

### PDF Input

Send a base64-encoded PDF:

```json
{
  "input": {
    "pdf_base64": "<BASE64_ENCODED_PDF>",
    "pages": "1-3",
    "prompt": "OCR this document and extract all text as markdown."
  }
}
```

### Image Input

Send a single base64-encoded image:

```json
{
  "input": {
    "image_base64": "<BASE64_ENCODED_IMAGE>",
    "prompt": "Extract all text from this image."
  }
}
```

### Multiple Images

Send an array of base64-encoded images:

```json
{
  "input": {
    "images_base64": ["<BASE64_IMAGE_1>", "<BASE64_IMAGE_2>"],
    "prompt": "OCR these images."
  }
}
```

### PDF from URL

Fetch and process a PDF from a URL:

```json
{
  "input": {
    "pdf_url": "https://example.com/document.pdf",
    "pages": "1,3-5",
    "prompt": "Extract all text preserving layout."
  }
}
```

**Default OCR prompt** (used when `prompt` is omitted): Extracts all text preserving layout, formatting, and structure as clean markdown. Tables use markdown table syntax, math uses LaTeX notation.

**Page range syntax**: `"1-3"`, `"1,3,5"`, `"1-3,5,7-9"`, or omit to process all pages.

## Usage: OpenAI Compatibility

The worker is fully compatible with OpenAI's API. Supported routes: **Chat Completions**, **Completions**, and **Models** — with both streaming and non-streaming.

### Modifying your OpenAI Codebase

Change `api_key` to your RunPod API Key and `base_url` to `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1`:

**Python**:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("RUNPOD_API_KEY"),
    base_url="https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1",
)
```

**curl**:

```bash
curl https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer <YOUR RUNPOD API KEY>" \
-d '{
  "model": "datalab-to/chandra",
  "messages": [
    {
      "role": "user",
      "content": "Why is RunPod the best platform?"
    }
  ],
  "temperature": 0,
  "max_tokens": 100
}'
```

### Chat Completions

- **Streaming**:
  ```python
  response_stream = client.chat.completions.create(
      model="datalab-to/chandra",
      messages=[{"role": "user", "content": "Describe this document."}],
      temperature=0,
      max_tokens=100,
      stream=True,
  )
  for chunk in response_stream:
      print(chunk.choices[0].delta.content or "", end="", flush=True)
  ```
- **Non-Streaming**:
  ```python
  response = client.chat.completions.create(
      model="datalab-to/chandra",
      messages=[{"role": "user", "content": "Describe this document."}],
      temperature=0,
      max_tokens=100,
  )
  print(response.choices[0].message.content)
  ```

### Getting Available Models

```python
models_response = client.models.list()
list_of_models = [model.id for model in models_response]
print(list_of_models)
```

## Usage: Standard (Non-OpenAI)

### Request Input Parameters

<details>
  <summary>Click to expand table</summary>

You may use a `prompt`, a list of `messages`, or ChandraOCR-specific inputs (`pdf_base64`, `image_base64`, `images_base64`, `pdf_url`) as input.

| Argument                     | Type                 | Default                                    | Description                                                         |
| ---------------------------- | -------------------- | ------------------------------------------ | ------------------------------------------------------------------- |
| `prompt`                     | str                  |                                            | Prompt string to generate text based on.                            |
| `messages`                   | list[dict[str, str]] |                                            | List of messages with chat template applied automatically.          |
| `pdf_base64`                 | str                  |                                            | Base64-encoded PDF for OCR processing.                              |
| `image_base64`               | str                  |                                            | Single base64-encoded image for OCR processing.                     |
| `images_base64`              | list[str]            |                                            | Array of base64-encoded images for OCR processing.                  |
| `pdf_url`                    | str                  |                                            | URL to a PDF for OCR processing.                                    |
| `pages`                      | str                  |                                            | Page range for PDF processing (e.g., `"1-3,5"`).                    |
| `apply_chat_template`        | bool                 | False                                      | Apply the model's chat template to `prompt`.                        |
| `sampling_params`            | dict                 | {}                                         | Sampling parameters (temperature, top_p, etc.).                     |
| `stream`                     | bool                 | False                                      | Enable streaming output.                                            |
| `max_batch_size`             | int                  | `DEFAULT_BATCH_SIZE`                       | Maximum tokens to stream per HTTP POST call.                        |
| `min_batch_size`             | int                  | `DEFAULT_MIN_BATCH_SIZE`                   | Minimum tokens to stream per HTTP POST call.                        |
| `batch_size_growth_factor`   | int                  | `DEFAULT_BATCH_SIZE_GROWTH_FACTOR`         | Growth factor for dynamic batch sizing.                             |

</details>

### Sampling Parameters

<details>
  <summary>Click to expand table</summary>

| Argument                          | Type                        | Default | Description                                                                       |
| --------------------------------- | --------------------------- | ------- | --------------------------------------------------------------------------------- |
| `n`                               | int                         | 1       | Number of output sequences to return.                                             |
| `best_of`                         | Optional[int]               | `n`     | Number of sequences to generate, returning the top `n`.                           |
| `presence_penalty`                | float                       | 0.0     | Penalizes tokens based on presence in generated text.                             |
| `frequency_penalty`               | float                       | 0.0     | Penalizes tokens based on frequency in generated text.                            |
| `repetition_penalty`              | float                       | 1.0     | Penalizes tokens based on appearance in prompt and generated text.                |
| `temperature`                     | float                       | 1.0     | Controls randomness. Lower = more deterministic. Zero = greedy.                   |
| `top_p`                           | float                       | 1.0     | Cumulative probability of top tokens. Must be in (0, 1].                          |
| `top_k`                           | int                         | -1      | Number of top tokens to consider. -1 = all tokens.                                |
| `min_p`                           | float                       | 0.0     | Minimum probability relative to the most likely token.                            |
| `use_beam_search`                 | bool                        | False   | Use beam search instead of sampling.                                              |
| `length_penalty`                  | float                       | 1.0     | Penalizes sequences by length. Used in beam search.                               |
| `early_stopping`                  | Union[bool, str]            | False   | Stopping condition in beam search. `True`, `False`, or `"never"`.                 |
| `stop`                            | Union[None, str, List[str]] | None    | Strings that stop generation when produced.                                       |
| `stop_token_ids`                  | Optional[List[int]]         | None    | Token IDs that stop generation.                                                   |
| `ignore_eos`                      | bool                        | False   | Continue generating after EOS token.                                              |
| `max_tokens`                      | int                         | 16      | Maximum tokens to generate per output sequence.                                   |
| `skip_special_tokens`             | bool                        | True    | Skip special tokens in output.                                                    |
| `spaces_between_special_tokens`   | bool                        | True    | Add spaces between special tokens.                                                |

</details>

### Text Input Formats

You may use a `prompt` or a list of `messages`:

1. **`prompt`** — A raw string. The model's chat template is not applied unless `apply_chat_template` is `true`.

    ```json
    {
      "input": {
        "prompt": "Extract text from this document.",
        "sampling_params": {
          "temperature": 0.7,
          "max_tokens": 100
        }
      }
    }
    ```

2. **`messages`** — A list of role/content dicts. The model's chat template is applied automatically.

    ```json
    {
      "input": {
        "messages": [
          {"role": "system", "content": "You are a helpful OCR assistant."},
          {"role": "user", "content": "Extract all text from this image as markdown."}
        ],
        "sampling_params": {
          "temperature": 0.7,
          "max_tokens": 100
        }
      }
    }
    ```

## Architecture

```
RunPod Request → handler.py → Input Detection
  ├─ ChandraOCR input? → chandra.py → PDF/Image preprocessing → vLLM
  ├─ OpenAI route?      → OpenAIvLLMEngine → vLLM
  └─ Native route?      → vLLMEngine → vLLM
```

### Key Source Files (`src/`)

| File               | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `handler.py`       | RunPod serverless entry point; routes requests to the appropriate engine |
| `engine.py`        | `vLLMEngine` and `OpenAIvLLMEngine` classes wrapping `AsyncLLMEngine`   |
| `engine_args.py`   | Configuration from env vars with auto-discovery of vLLM engine args     |
| `chandra.py`       | ChandraOCR pipeline: PDF/image decoding, preprocessing, OCR handling    |
| `tokenizer.py`     | Chat template application via `TokenizerWrapper`                        |
| `utils.py`         | `JobInput` parsing, `BatchSize` dynamic batching, error formatting      |
| `download_model.py`| Model download during Docker build for baked images                     |
| `constants.py`     | Default values for batch size, concurrency, growth factor               |

### Compatible Model Architectures

You can deploy **any model on Hugging Face** supported by vLLM. For the complete list, see the [vLLM Supported Models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html).
