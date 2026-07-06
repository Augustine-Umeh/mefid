"""Step 5: Qwen2-VL inference for a single frame window."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import torch  # pyright: ignore[reportMissingImports]

from PIL import Image  # pyright: ignore[reportMissingImports]

from exports.schema.constants import (
    CAPTION_MAX_NEW_TOKENS,
    CAPTION_MAX_PIXELS_DEFAULT,
    CAPTION_MAX_PIXELS_FLOOR,
    CAPTION_MODEL,
    CAPTION_PROMPT,
    CAPTION_CACHE,
)
from exports.utils.logger import get_logger

logger = get_logger()


def _resolve_caption_cache_dir() -> str | None:
    raw = CAPTION_CACHE
    if raw is None or not str(raw).strip():
        return None
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("CAPTION_CACHE=%r unusable (%s); using default HF cache", raw, exc)
        return None
    if not os.access(path, os.W_OK):
        logger.warning("CAPTION_CACHE=%r not writable; using default HF cache", raw)
        return None
    return str(path.resolve())


def build_messages(frames: list[Image.Image], prompt: str) -> list[dict[str, Any]]:
    """Build a Qwen2-VL chat message with one image block per frame."""
    content: list[dict[str, Any]] = [
        {"type": "image", "image": frame} for frame in frames
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def decode_caption_text(processor: Any, output_ids: Any, input_ids: Any) -> str:
    """Strip the prompt prefix and return the generated caption text."""
    trimmed = [
        out[len(inp) :]
        for inp, out in zip(input_ids, output_ids, strict=True)
    ]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    
    return text


class CaptionGenerator:
    """Loads Qwen2-VL and generates text for multi-frame windows."""

    def __init__(self, model: Any, processor: Any) -> None:
        self._model = model
        self._processor = processor

    @classmethod
    def load(cls) -> CaptionGenerator:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # pyright: ignore[reportMissingImports]

        cache_dir = _resolve_caption_cache_dir()
        model_id = CAPTION_MODEL
        logger.info("Loading caption model %s", model_id)

        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=CAPTION_MAX_PIXELS_FLOOR,
            max_pixels=CAPTION_MAX_PIXELS_DEFAULT,
            cache_dir=cache_dir,
        )

        if torch.cuda.is_available():
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir,
            )
        else:
            logger.warning("CUDA unavailable; loading caption model on CPU (slow)")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                cache_dir=cache_dir,
            )

        model.eval()
        return cls(model=model, processor=processor)

    def caption_window(self, frames: list[Image.Image], *, max_pixels: int) -> str:
        """Describe the chronological frame sequence in one concise paragraph."""
        if not frames:
            return ""

        from qwen_vl_utils import process_vision_info  # pyright: ignore[reportMissingImports]

        messages = build_messages(frames, CAPTION_PROMPT)
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            max_pixels=max_pixels,
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=CAPTION_MAX_NEW_TOKENS,
            )

        caption = decode_caption_text(
            self._processor,
            output_ids,
            inputs.input_ids,
        )

        if torch.cuda.is_available():
            del inputs, output_ids, image_inputs, video_inputs
            torch.cuda.empty_cache()

        return caption
