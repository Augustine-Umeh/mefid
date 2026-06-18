"""faster-whisper wrapper for Mefid transcript ingestion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

from exports.schema.constants import (
    WHISPER_CACHE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)
from exports.utils.logger import get_logger
from exports.utils.transcript_chunking import (
    TimedWord,
    TranscriptChunkDraft,
    chunk_segment_text_for_clip,
    chunk_words_for_clip,
    merge_segment_drafts,
)

logger = get_logger()


def _resolve_whisper_cache_dir() -> str | None:
    raw = WHISPER_CACHE
    if raw is None or not str(raw).strip():
        return None
    # Expand the path to the user's home directory.
    path = Path(raw).expanduser() 
    try:
        # Create the directory and all parent directories if they don't exist.
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("WHISPER_CACHE=%r unusable (%s); using default cache", raw, exc)
        return None
    if not os.access(path, os.W_OK):
        logger.warning("WHISPER_CACHE=%r not writable; using default cache", raw)
        return None
    return str(path.resolve())


class WhisperEngine:
    """Loads faster-whisper once; synchronous transcribe helpers."""

    def __init__(self, model: WhisperModel) -> None:
        self._model = model

    @classmethod
    def load(cls) -> "WhisperEngine":
        cache_dir = _resolve_whisper_cache_dir()
        kwargs = {}
        if cache_dir:
            kwargs["download_root"] = cache_dir

        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s cache=%s",
            WHISPER_MODEL,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
            cache_dir or "(default)",
        )
        model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            **kwargs,
        )
        logger.info("Whisper ready model=%s", WHISPER_MODEL)
        return cls(model)

    def transcribe_to_chunks(self, video_path: str) -> List[TranscriptChunkDraft]:
        """Run ASR and return CLIP-safe timed transcript chunks."""
        segments, _info = self._model.transcribe(
            video_path,
            word_timestamps=True,
            vad_filter=True,
        )

        drafts: List[TranscriptChunkDraft] = []
        for segment in segments:
            words: List[TimedWord] = []
            if segment.words:
                for word in segment.words:
                    token = (word.word or "").strip()
                    if not token:
                        continue
                    words.append(
                        TimedWord(
                            word=token,
                            start=float(word.start),
                            end=float(word.end),
                        )
                    )

            if words:
                drafts.extend(chunk_words_for_clip(words))
            else:
                drafts.extend(
                    chunk_segment_text_for_clip(
                        segment.text or "",
                        float(segment.start),
                        float(segment.end),
                    )
                )

        return merge_segment_drafts(drafts)
