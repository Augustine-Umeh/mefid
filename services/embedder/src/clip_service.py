"""CLIP image/text embedding (ViT-L/14 @ 336px via Hugging Face)."""

from __future__ import annotations

import base64
import binascii
import os
from io import BytesIO
from pathlib import Path
from typing import List, Sequence, Tuple
from uuid import UUID

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from exports.schema.constants import (
    CLIP_MODEL,
    TRANSFORMERS_CACHE,
    EMBED_IMAGE_BATCH_SIZE,
)

from exports.utils.logger import get_logger

logger = get_logger()

CLIP_MAX_LENGTH = 77


def _resolve_hf_cache_dir() -> str | None:
    """Use TRANSFORMERS_CACHE only if the path is creatable and writable (Docker-friendly).

    We fall back to Hugging Face's default cache if host is normal macOS/Linux host.
    """
    raw = TRANSFORMERS_CACHE
    if raw is None or not str(raw).strip():
        return None
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "TRANSFORMERS_CACHE=%r is not usable (%s); using default HF cache",
            raw,
            exc,
        )
        return None
    if not os.access(path, os.W_OK):
        logger.warning(
            "TRANSFORMERS_CACHE=%r is not writable; using default HF cache",
            raw,
        )
        return None
    return str(path.resolve())


def _l2_normalize(vectors: torch.Tensor) -> torch.Tensor:
    norms = vectors.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    return vectors / norms


def _clip_features_to_tensor(feats: torch.Tensor | object) -> torch.Tensor:
    """Transformers 4.x returns a tensor; 5.x returns ModelOutput (projected CLIP vec in ``pooler_output``)."""
    if isinstance(feats, torch.Tensor):
        return feats
    if isinstance(feats, tuple):
        for item in feats:
            if isinstance(item, torch.Tensor):
                return item
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        t = getattr(feats, attr, None)
        if isinstance(t, torch.Tensor):
            if attr == "last_hidden_state":
                return t[:, 0, :]
            return t
    raise TypeError(f"Cannot get embedding tensor from {type(feats).__name__}")


def decode_base64_jpeg(data: str) -> Image.Image:
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as e:
        raise ValueError("invalid base64") from e
    try:
        img = Image.open(BytesIO(raw))
        return img.convert("RGB")
    except OSError as e:
        raise ValueError("invalid image data") from e


class ClipEmbeddingEngine:
    """Loads CLIP once; synchronous encode helpers (call from thread pool in async routes)."""

    def __init__(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        device: torch.device,
        image_batch_size: int,
    ) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._image_batch_size = max(1, image_batch_size)

    @classmethod
    def load(cls) -> "ClipEmbeddingEngine":
        model_id = CLIP_MODEL
        cache_dir = _resolve_hf_cache_dir()
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        logger.info("Loading CLIP model=%s cache_dir=%s", model_id, cache_dir or "(HF default)")
        model = CLIPModel.from_pretrained(model_id, **kwargs)
        processor = CLIPProcessor.from_pretrained(model_id, **kwargs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        engine = cls(model, processor, device, EMBED_IMAGE_BATCH_SIZE)
        dim = model.config.projection_dim
        logger.info(
            "CLIP ready device=%s projection_dim=%s image_batch_size=%s",
            device,
            dim,
            engine._image_batch_size,
        )
        return engine

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self._processor(images=list(images), return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            raw = self._model.get_image_features(**inputs)
        feats = _clip_features_to_tensor(raw)
        return _l2_normalize(feats)

    def embed_images(
        self, frame_ids: Sequence[UUID], frame_b64: Sequence[str]
    ) -> List[Tuple[UUID, List[float]]]:
        if len(frame_ids) != len(frame_b64):
            raise ValueError("frame_ids and frame_b64 length mismatch")

        out: List[Tuple[UUID, List[float]]] = []
        bs = self._image_batch_size
        for start in range(0, len(frame_ids), bs):
            chunk_ids = frame_ids[start : start + bs]
            chunk_b64 = frame_b64[start : start + bs]
            chunk_imgs: List[Image.Image] = []
            for i, raw in enumerate(chunk_b64):
                try:
                    chunk_imgs.append(decode_base64_jpeg(raw))
                except ValueError as e:
                    raise ValueError(f"frame at index {start + i}: {e}") from e
            vecs = self._encode_image_batch(chunk_imgs)
            for fid, row in zip(chunk_ids, vecs, strict=True):
                out.append((fid, row.detach().cpu().float().tolist()))
        return out

    def _text_would_truncate(self, text: str) -> bool:
        enc = self._processor.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        return len(enc["input_ids"]) > CLIP_MAX_LENGTH

    def embed_text(self, text: str) -> List[float]:
        truncate = False
        if self._text_would_truncate(text):
            logger.warning(
                "CLIP text input exceeds %s tokens; truncation will apply (chars=%s)",
                CLIP_MAX_LENGTH,
                len(text),
            )
            truncate = True
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=CLIP_MAX_LENGTH,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            raw = self._model.get_text_features(**inputs)
        feats = _l2_normalize(_clip_features_to_tensor(raw))
        return feats[0].detach().cpu().float().tolist()

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        cleaned = [(t or "").strip() for t in texts]
        if not any(cleaned):
            return [[] for _ in texts]

        truncate_flags: List[bool] = []
        for text in cleaned:
            if not text:
                truncate_flags.append(False)
                continue
            would_truncate = self._text_would_truncate(text)
            if would_truncate:
                logger.warning(
                    "CLIP text input exceeds %s tokens; truncation will apply (chars=%s)",
                    CLIP_MAX_LENGTH,
                    len(text),
                )
            truncate_flags.append(would_truncate)

        inputs = self._processor(
            text=cleaned,
            return_tensors="pt",
            padding=True,
            truncation=any(truncate_flags),
            max_length=CLIP_MAX_LENGTH,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            raw = self._model.get_text_features(**inputs)
        feats = _l2_normalize(_clip_features_to_tensor(raw))
        return [row.detach().cpu().float().tolist() for row in feats]
