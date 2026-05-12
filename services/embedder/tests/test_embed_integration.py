"""Optional full-stack tests against a real CLIP checkpoint (slow, needs disk + network once)."""

import base64
import os
import uuid
from io import BytesIO

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_CLIP_INTEGRATION"),
    reason="Set RUN_CLIP_INTEGRATION=1 to download CLIP and run integration tests",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.embedder_app import app

    with TestClient(app) as c:
        yield c


def _tiny_jpeg_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(128, 64, 32)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_embed_images_returns_768_unit_norm(client) -> None:
    fid = str(uuid.uuid4())
    r = client.post(
        "/embed/images/",
        json={
            "frames": [{"frame_id": fid, "frame_data": _tiny_jpeg_b64()}],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    emb = data["embeddings"][0]["embedding"]
    assert len(emb) == 768
    norm = sum(x * x for x in emb) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_embed_text_returns_768_unit_norm(client) -> None:
    r = client.post(
        "/embed/text/",
        json={"text": "a red bicycle near a lake"},
    )
    assert r.status_code == 200, r.text
    emb = r.json()["embedding"]
    assert len(emb) == 768
    norm = sum(x * x for x in emb) ** 0.5
    assert abs(norm - 1.0) < 1e-3
