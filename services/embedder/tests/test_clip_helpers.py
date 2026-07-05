"""Fast unit tests (no CLIP model download)."""

import base64
from io import BytesIO

import pytest
import torch
from PIL import Image  # pyright: ignore[reportMissingImports]

from src.clip_service import _l2_normalize, decode_base64_jpeg


def test_l2_normalize_unit_vectors() -> None:
    x = torch.tensor([[3.0, 4.0], [0.0, 12.0]])
    y = _l2_normalize(x)
    assert torch.allclose(y[0].norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(y[1].norm(), torch.tensor(1.0), atol=1e-5)


def test_decode_base64_jpeg_round_trip() -> None:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img = decode_base64_jpeg(b64)
    assert img.mode == "RGB"
    assert img.size == (4, 4)


def test_decode_invalid_base64() -> None:
    with pytest.raises(ValueError, match="invalid base64"):
        decode_base64_jpeg("@@@not-valid-base64@@@")


def test_decode_invalid_image_bytes() -> None:
    b64 = base64.b64encode(b"not a jpeg").decode()
    with pytest.raises(ValueError, match="invalid image"):
        decode_base64_jpeg(b64)
