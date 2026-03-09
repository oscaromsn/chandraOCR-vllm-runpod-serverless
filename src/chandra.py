import asyncio
import base64
import io
import logging
import os
import time
from typing import AsyncGenerator

log = logging.getLogger(__name__)

# Qwen3-VL patch factor: patch_size=14 × merge_size=2 = 28
_PATCH_FACTOR = 28
# Default max_pixels: 1024 visual tokens × 784 (28²) = 802,816
_DEFAULT_MAX_PIXELS = int(os.environ.get("CHANDRA_MAX_PIXELS", 802816))
# Max pages to infer concurrently (prevents KV cache exhaustion)
_DEFAULT_MAX_CONCURRENT_PAGES = int(os.environ.get("CHANDRA_MAX_CONCURRENT_PAGES", 4))

CHANDRA_INPUT_KEYS = {"pdf_base64", "image_base64", "images_base64", "pdf_url"}

DEFAULT_OCR_PROMPT = (
    "OCR this image. Extract all text content preserving the original layout, "
    "formatting, and structure. Output the result as clean markdown. "
    "For tables, use markdown table syntax. For mathematical expressions, use LaTeX notation. "
    "Preserve headings, lists, and paragraph structure."
)


def detect_chandra_input(job_input: dict) -> bool:
    return bool(CHANDRA_INPUT_KEYS & set(job_input.keys()))


def _parse_page_range(page_range_str: str, total_pages: int) -> list:
    """Parse page range like '1-3,5,7-9' into sorted list of 0-indexed page numbers."""
    pages = []
    for part in page_range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start) - 1, min(int(end), total_pages)))
        else:
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                pages.append(idx)
    return sorted(set(pages))


def _decode_images(job_input: dict) -> list:
    """Decode images from various input formats. Returns list of PIL Images."""
    from PIL import Image

    images = []

    if "pdf_base64" in job_input:
        from pdf2image import convert_from_bytes
        pdf_bytes = base64.b64decode(job_input["pdf_base64"])
        images = convert_from_bytes(pdf_bytes, dpi=job_input.get("dpi", 200))

    elif "pdf_url" in job_input:
        import requests as req
        from pdf2image import convert_from_bytes
        resp = req.get(job_input["pdf_url"], timeout=120)
        resp.raise_for_status()
        images = convert_from_bytes(resp.content, dpi=job_input.get("dpi", 200))

    elif "image_base64" in job_input:
        img_bytes = base64.b64decode(job_input["image_base64"])
        images = [Image.open(io.BytesIO(img_bytes))]

    elif "images_base64" in job_input:
        for b64 in job_input["images_base64"]:
            img_bytes = base64.b64decode(b64)
            images.append(Image.open(io.BytesIO(img_bytes)))

    if "page_range" in job_input and len(images) > 1:
        selected = _parse_page_range(job_input["page_range"], len(images))
        images = [images[i] for i in selected]

    return images


def _preprocess_image(image, max_pixels: int = _DEFAULT_MAX_PIXELS):
    """Resize image so H*W <= max_pixels, with dimensions divisible by _PATCH_FACTOR."""
    w, h = image.size
    pixels = w * h
    if pixels <= max_pixels:
        # Still ensure dimensions are divisible by patch factor
        new_w = max(_PATCH_FACTOR, (w // _PATCH_FACTOR) * _PATCH_FACTOR)
        new_h = max(_PATCH_FACTOR, (h // _PATCH_FACTOR) * _PATCH_FACTOR)
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h))
        return image

    scale = (max_pixels / pixels) ** 0.5
    new_w = max(_PATCH_FACTOR, int(w * scale) // _PATCH_FACTOR * _PATCH_FACTOR)
    new_h = max(_PATCH_FACTOR, int(h * scale) // _PATCH_FACTOR * _PATCH_FACTOR)

    # Ensure we don't exceed max_pixels after rounding up
    while new_w * new_h > max_pixels:
        if new_w >= new_h:
            new_w -= _PATCH_FACTOR
        else:
            new_h -= _PATCH_FACTOR

    log.debug(f"Resized image {w}x{h} -> {new_w}x{new_h} ({new_w*new_h} pixels, ~{new_w*new_h//784} visual tokens)")
    return image.resize((new_w, new_h))


def _image_to_base64(image, fmt: str = "JPEG", quality: int = 85) -> tuple[str, str]:
    """Convert PIL Image to base64 string. Returns (base64_str, mime_type)."""
    buf = io.BytesIO()
    if fmt.upper() == "JPEG" and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buf, format=fmt.upper(), quality=quality)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return base64.b64encode(buf.getvalue()).decode("utf-8"), mime


def _build_ocr_prompt(job_input: dict) -> str:
    if "custom_prompt" in job_input:
        return job_input["custom_prompt"]
    try:
        from chandra_ocr.prompts import get_ocr_prompt
        return get_ocr_prompt()
    except (ImportError, AttributeError):
        pass
    return DEFAULT_OCR_PROMPT


async def _generate_single_page(openai_engine, openai_input: dict, page_num: int, semaphore: asyncio.Semaphore) -> dict:
    """Run a single page through the OpenAI engine and collect the response."""
    from utils import JobInput

    async with semaphore:
        log.info(f"Page {page_num}: starting inference")
        t0 = time.monotonic()

        synthetic = {
            "openai_route": "/v1/chat/completions",
            "openai_input": openai_input,
        }
        job_input = JobInput(synthetic)

        result = None
        async for batch in openai_engine.generate(job_input):
            result = batch

        elapsed = time.monotonic() - t0
        log.info(f"Page {page_num}: inference completed in {elapsed:.1f}s")
        return result


async def process_chandra_request(job_input: dict, openai_engine) -> AsyncGenerator:
    """Full ChandraOCR pipeline: decode input -> build vision prompts -> run concurrently -> assemble output."""
    pipeline_t0 = time.monotonic()
    log.info("Processing ChandraOCR request")

    decode_t0 = time.monotonic()
    images = _decode_images(job_input)
    if not images:
        yield {"error": "No images could be decoded from input"}
        return

    max_pixels = job_input.get("max_pixels", _DEFAULT_MAX_PIXELS)
    image_format = job_input.get("image_format", "JPEG")
    image_quality = job_input.get("image_quality", 85)
    decode_elapsed = time.monotonic() - decode_t0

    max_concurrent = job_input.get("max_concurrent_pages", _DEFAULT_MAX_CONCURRENT_PAGES)
    log.info(
        f"Processing {len(images)} page(s) (max_pixels={max_pixels}, format={image_format}, "
        f"max_concurrent={max_concurrent}, decode={decode_elapsed:.1f}s)"
    )

    ocr_prompt = _build_ocr_prompt(job_input)
    max_tokens = job_input.get("max_tokens")  # None = let vLLM auto-calculate
    temperature = job_input.get("temperature", 0)
    served_model_name = openai_engine.served_model_name

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    preprocess_t0 = time.monotonic()
    for i, image in enumerate(images):
        image = _preprocess_image(image, max_pixels=max_pixels)
        b64, mime = _image_to_base64(image, fmt=image_format, quality=image_quality)
        openai_input = {
            "model": served_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": ocr_prompt},
                ]
            }],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            openai_input["max_tokens"] = max_tokens
        tasks.append(_generate_single_page(openai_engine, openai_input, page_num=i + 1, semaphore=semaphore))
    preprocess_elapsed = time.monotonic() - preprocess_t0

    inference_t0 = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    inference_elapsed = time.monotonic() - inference_t0

    pages = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            pages.append({"page": i + 1, "markdown": f"Error processing page: {result}", "error": str(result)})
        elif isinstance(result, dict) and "error" in result:
            pages.append({"page": i + 1, "markdown": "", "error": result["error"]})
        else:
            try:
                text = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = str(result)
            pages.append({"page": i + 1, "markdown": text})

    full_markdown = "\n\n---\n\n".join(p["markdown"] for p in pages if p.get("markdown"))
    pipeline_elapsed = time.monotonic() - pipeline_t0

    timing = {
        "decode_s": round(decode_elapsed, 2),
        "preprocess_s": round(preprocess_elapsed, 2),
        "inference_s": round(inference_elapsed, 2),
        "total_pipeline_s": round(pipeline_elapsed, 2),
    }
    log.info(
        f"ChandraOCR complete: {len(pages)} pages in {pipeline_elapsed:.1f}s "
        f"(decode={decode_elapsed:.1f}s, preprocess={preprocess_elapsed:.1f}s, inference={inference_elapsed:.1f}s)"
    )

    yield {
        "pages": pages,
        "full_markdown": full_markdown,
        "total_pages": len(pages),
        "timing": timing,
    }
