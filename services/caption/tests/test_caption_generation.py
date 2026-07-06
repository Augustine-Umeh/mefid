"""Unit tests for Qwen2-VL caption generation helpers and window inference."""

import sys
from unittest.mock import MagicMock, patch

from PIL import Image  # pyright: ignore[reportMissingImports]

from src.pipeline.caption_generation import (
    CaptionGenerator,
    build_messages,
    decode_caption_text,
)


def _fake_frames(count: int = 2) -> list[Image.Image]:
    return [Image.new("RGB", (8, 8), color=(i, i, i)) for i in range(count)]


def test_build_messages_one_image_block_per_frame():
    frames = _fake_frames(3)
    messages = build_messages(frames, "Describe the clip.")

    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[-1]["type"] == "text"
    assert content[-1]["text"] == "Describe the clip."
    assert sum(item["type"] == "image" for item in content) == 3


def test_decode_caption_text_trims_prompt_prefix():
    processor = MagicMock()
    processor.batch_decode.return_value = ["assistant A player kicks the ball."]

    text = decode_caption_text(
        processor,
        output_ids=[[1, 2, 3]],
        input_ids=[[1]],
    )

    assert text == "A player kicks the ball."


def test_caption_window_returns_empty_for_no_frames():
    generator = CaptionGenerator(model=MagicMock(), processor=MagicMock())
    assert generator.caption_window([], max_pixels=1280) == ""


def test_caption_window_generates_caption():
    torch_mock = MagicMock()
    torch_mock.inference_mode.return_value.__enter__ = MagicMock(return_value=None)
    torch_mock.inference_mode.return_value.__exit__ = MagicMock(return_value=False)
    torch_mock.cuda.is_available.return_value = False

    qwen_mock = MagicMock()
    qwen_mock.process_vision_info.return_value = ([_fake_frames()[0]], None)

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = [[10, 11, 12]]

    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    bound_inputs = MagicMock(input_ids=[[1, 2]])
    processor.return_value.to.return_value = bound_inputs
    processor.batch_decode.return_value = ["Generated caption text."]

    with patch.dict(
        sys.modules,
        {"torch": torch_mock, "qwen_vl_utils": qwen_mock},
    ):
        generator = CaptionGenerator(model=model, processor=processor)
        text = generator.caption_window(_fake_frames(2), max_pixels=512 * 28 * 28)

    assert text == "Generated caption text."
    model.generate.assert_called_once()
    _, kwargs = processor.call_args
    assert kwargs["max_pixels"] == 512 * 28 * 28


def test_caption_generator_load_uses_4bit_on_cuda():
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True
    torch_mock.float16 = "float16"
    torch_mock.float32 = "float32"

    transformers_mock = MagicMock()
    auto_processor = MagicMock()
    auto_processor.from_pretrained.return_value = MagicMock()
    transformers_mock.AutoProcessor = auto_processor
    transformers_mock.BitsAndBytesConfig = MagicMock()
    model_cls = MagicMock()
    model = MagicMock()
    model.eval.return_value = model
    model_cls.from_pretrained.return_value = model
    transformers_mock.Qwen2VLForConditionalGeneration = model_cls

    with patch.dict(
        sys.modules,
        {"torch": torch_mock, "transformers": transformers_mock},
    ):
        generator = CaptionGenerator.load()

    assert isinstance(generator, CaptionGenerator)
    model_cls.from_pretrained.assert_called_once()
    assert "quantization_config" in model_cls.from_pretrained.call_args.kwargs
    transformers_mock.BitsAndBytesConfig.assert_called_once()
