"""HTTP validation on embed routes without loading CLIP."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routes.embed_image import router as embed_image_router
from src.routes.embed_text import router as embed_text_router


def _image_app() -> TestClient:
    app = FastAPI()
    app.include_router(embed_image_router, prefix="/embed/images")
    return TestClient(app)


def _text_app() -> TestClient:
    app = FastAPI()
    app.include_router(embed_text_router, prefix="/embed/text")
    return TestClient(app)


def test_embed_images_empty_frames_422() -> None:
    client = _image_app()
    r = client.post("/embed/images/", json={"frames": []})
    assert r.status_code == 422


def test_embed_text_whitespace_422() -> None:
    client = _text_app()
    r = client.post("/embed/text/", json={"text": "   "})
    assert r.status_code == 422
