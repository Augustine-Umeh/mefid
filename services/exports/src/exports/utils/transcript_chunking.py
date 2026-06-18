"""Split Whisper output into CLIP-safe transcript chunks with timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from exports.schema.constants import CLIP_TEXT_CHUNK_MAX_TOKENS

# Rough English heuristic when a tokenizer is unavailable (~4 chars/token).
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class TimedWord:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptChunkDraft:
    start_time: float
    end_time: float
    text: str


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // _CHARS_PER_TOKEN)

"""
Flushes a list of TimedWords into a list of TranscriptChunkDrafts.
"""
def _flush_word_buffer(
    buffer: List[TimedWord],
    out: List[TranscriptChunkDraft],
) -> None:
    if not buffer:
        return
    text = " ".join(w.word.strip() for w in buffer if w.word.strip()).strip()
    if not text:
        buffer.clear()
        return
    out.append(
        TranscriptChunkDraft(
            start_time=buffer[0].start,
            end_time=buffer[-1].end,
            text=text,
        )
    )
    buffer.clear()

"""
Groups word-level timestamps into embeddable chunks under the token budget.
"""
def chunk_words_for_clip(
    words: Sequence[TimedWord],
    *,
    max_tokens: int = CLIP_TEXT_CHUNK_MAX_TOKENS,
) -> List[TranscriptChunkDraft]:
    """Group word-level timestamps into embeddable chunks under the token budget."""
    chunks: List[TranscriptChunkDraft] = []
    buffer: List[TimedWord] = []
    buffer_tokens = 0

    for word in words:
        token_est = _estimate_tokens(word.word)
        if buffer and buffer_tokens + token_est > max_tokens:
            _flush_word_buffer(buffer, chunks)
            buffer_tokens = 0
        buffer.append(word)
        buffer_tokens += token_est

    _flush_word_buffer(buffer, chunks)
    return chunks


def chunk_segment_text_for_clip(
    text: str,
    start_time: float,
    end_time: float,
    *,
    max_tokens: int = CLIP_TEXT_CHUNK_MAX_TOKENS,
) -> List[TranscriptChunkDraft]:
    """Fallback when word timestamps are unavailable: split on sentence boundaries."""
    stripped = text.strip()
    if not stripped:
        return []
    if _estimate_tokens(stripped) <= max_tokens:
        return [
            TranscriptChunkDraft(
                start_time=start_time,
                end_time=end_time,
                text=stripped,
            )
        ]

    sentences = _split_sentences(stripped)
    if len(sentences) == 1:
        return _split_by_char_budget(stripped, start_time, end_time, max_tokens)

    chunks: List[TranscriptChunkDraft] = []
    duration = max(end_time - start_time, 0.0)
    cursor = start_time
    buffer: List[str] = []
    buffer_tokens = 0
    total_chars = sum(len(s) for s in sentences) or 1

    def flush(sentences_in_buffer: List[str]) -> None:
        nonlocal cursor
        if not sentences_in_buffer:
            return
        chunk_text = " ".join(sentences_in_buffer).strip()
        char_share = sum(len(s) for s in sentences_in_buffer) / total_chars
        chunk_end = end_time if cursor + duration * char_share >= end_time else cursor + duration * char_share
        chunks.append(
            TranscriptChunkDraft(
                start_time=cursor,
                end_time=chunk_end,
                text=chunk_text,
            )
        )
        cursor = chunk_end

    for sentence in sentences:
        est = _estimate_tokens(sentence)
        if buffer and buffer_tokens + est > max_tokens:
            flush(buffer)
            buffer = []
            buffer_tokens = 0
        buffer.append(sentence)
        buffer_tokens += est

    flush(buffer)
    if chunks:
        last = chunks[-1]
        chunks[-1] = TranscriptChunkDraft(
            start_time=last.start_time,
            end_time=end_time,
            text=last.text,
        )
    return chunks


def _split_sentences(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    for ch in text:
        current.append(ch)
        if ch in ".!?":
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts or [text]


def _split_by_char_budget(
    text: str,
    start_time: float,
    end_time: float,
    max_tokens: int,
) -> List[TranscriptChunkDraft]:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    words = text.split()
    chunks: List[TranscriptChunkDraft] = []
    buffer: List[str] = []
    char_count = 0
    duration = max(end_time - start_time, 0.0)
    total_chars = len(text) or 1
    cursor = start_time

    def flush(buf: List[str]) -> None:
        nonlocal cursor
        if not buf:
            return
        chunk_text = " ".join(buf).strip()
        char_share = len(chunk_text) / total_chars
        chunk_end = min(end_time, cursor + duration * char_share)
        chunks.append(
            TranscriptChunkDraft(
                start_time=cursor,
                end_time=chunk_end,
                text=chunk_text,
            )
        )
        cursor = chunk_end

    for word in words:
        extra = len(word) + (1 if buffer else 0)
        if buffer and char_count + extra > max_chars:
            flush(buffer)
            buffer = []
            char_count = 0
        buffer.append(word)
        char_count += extra

    flush(buffer)
    if chunks:
        last = chunks[-1]
        chunks[-1] = TranscriptChunkDraft(
            start_time=last.start_time,
            end_time=end_time,
            text=last.text,
        )
    return chunks


def merge_segment_drafts(
    drafts: Iterable[TranscriptChunkDraft],
) -> List[TranscriptChunkDraft]:
    """Drop empty chunks and preserve order."""
    return [d for d in drafts if d.text.strip()]
