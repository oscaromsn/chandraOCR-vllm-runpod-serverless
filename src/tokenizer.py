from transformers import AutoTokenizer
import os
from pathlib import Path
from typing import Union


def _load_fallback_chat_template() -> str | None:
    """Load the bundled Qwen3-VL chat template as a fallback."""
    template_path = Path(__file__).parent / "chat_templates" / "qwen3vl.jinja"
    try:
        return template_path.read_text()
    except FileNotFoundError:
        return None


class TokenizerWrapper:
    def __init__(self, tokenizer_name_or_path, tokenizer_revision, trust_remote_code):
        print(f"tokenizer_name_or_path: {tokenizer_name_or_path}, tokenizer_revision: {tokenizer_revision}, trust_remote_code: {trust_remote_code}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, revision=tokenizer_revision or "main", trust_remote_code=trust_remote_code)

        # Priority: CUSTOM_CHAT_TEMPLATE env var > model's built-in > bundled fallback
        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
        if self.custom_chat_template and isinstance(self.custom_chat_template, str):
            self.tokenizer.chat_template = self.custom_chat_template
        elif not self.tokenizer.chat_template:
            fallback = _load_fallback_chat_template()
            if fallback:
                print("No chat template found in model tokenizer, using bundled Qwen3-VL fallback")
                self.tokenizer.chat_template = fallback
        self.has_chat_template = bool(self.tokenizer.chat_template)

    def apply_chat_template(self, input: Union[str, list[dict[str, str]]]) -> str:
        if isinstance(input, list):
            if not self.has_chat_template:
                raise ValueError(
                    "Chat template does not exist for this model, you must provide a single string input instead of a list of messages"
                )
        elif isinstance(input, str):
            input = [{"role": "user", "content": input}]
        else:
            raise ValueError("Input must be a string or a list of messages")
        
        return self.tokenizer.apply_chat_template(
            input, tokenize=False, add_generation_prompt=True
        )
