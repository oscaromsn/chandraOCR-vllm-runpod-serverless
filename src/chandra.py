import asyncio
import base64
import io
import logging
from typing import AsyncGenerator

log = logging.getLogger(__name__)

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
        images = convert_from_bytes(pdf_bytes, dpi=job_input.get("dpi", 300))

    elif "pdf_url" in job_input:
        import requests as req
        from pdf2image import convert_from_bytes
        resp = req.get(job_input["pdf_url"], timeout=120)
        resp.raise_for_status()
        images = convert_from_bytes(resp.content, dpi=job_input.get("dpi", 300))

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


def _image_to_base64(image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_ocr_prompt(job_input: dict) -> str:
    if "custom_prompt" in job_input:
        return job_input["custom_prompt"]
    try:
        from chandra_ocr.prompts import get_ocr_prompt
        return get_ocr_prompt()
    except (ImportError, AttributeError):
        pass
    return DEFAULT_OCR_PROMPT


async def _generate_single_page(openai_engine, openai_input: dict) -> dict:
    """Run a single page through the OpenAI engine and collect the response."""
    from utils import JobInput

    synthetic = {
        "openai_route": "/v1/chat/completions",
        "openai_input": openai_input,
    }
    job_input = JobInput(synthetic)

    result = None
    async for batch in openai_engine.generate(job_input):
        result = batch
    return result


async def process_chandra_request(job_input: dict, openai_engine) -> AsyncGenerator:
    """Full ChandraOCR pipeline: decode input -> build vision prompts -> run concurrently -> assemble output."""
    log.info("Processing ChandraOCR request")

    images = _decode_images(job_input)
    if not images:
        yield {"error": "No images could be decoded from input"}
        return

    log.info(f"Processing {len(images)} page(s)")

    ocr_prompt = _build_ocr_prompt(job_input)
    max_tokens = job_input.get("max_tokens")  # None = let vLLM auto-calculate
    temperature = job_input.get("temperature", 0)
    served_model_name = openai_engine.served_model_name

    tasks = []
    for image in images:
        b64 = _image_to_base64(image)
        openai_input = {
            "model": served_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": ocr_prompt},
                ]
            }],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            openai_input["max_tokens"] = max_tokens
        tasks.append(_generate_single_page(openai_engine, openai_input))

    results = await asyncio.gather(*tasks, return_exceptions=True)

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

    yield {
        "pages": pages,
        "full_markdown": full_markdown,
        "total_pages": len(pages),
    }
