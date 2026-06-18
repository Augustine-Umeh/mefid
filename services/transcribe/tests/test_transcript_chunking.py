"""Unit tests for CLIP-safe transcript chunking."""

from exports.utils.transcript_chunking import (
    TimedWord,
    chunk_segment_text_for_clip,
    chunk_words_for_clip,
)


def test_chunk_words_splits_long_segments() -> None:
    words = [
        TimedWord(word=f"word{i}", start=float(i), end=float(i) + 0.5)
        for i in range(120)
    ]
    chunks = chunk_words_for_clip(words, max_tokens=10)
    assert len(chunks) > 1
    assert chunks[0].start_time == 0.0
    assert chunks[-1].end_time == words[-1].end
    for chunk in chunks:
        assert chunk.text
        assert chunk.start_time <= chunk.end_time


def test_chunk_segment_short_text_unchanged() -> None:
    chunks = chunk_segment_text_for_clip("hello there", 1.0, 3.0, max_tokens=70)
    assert len(chunks) == 1
    assert chunks[0].text == "hello there"
    assert chunks[0].start_time == 1.0
    assert chunks[0].end_time == 3.0


def test_chunk_segment_long_text_splits() -> None:
    long_text = " ".join(["sentence"] * 200)
    chunks = chunk_segment_text_for_clip(long_text, 0.0, 100.0, max_tokens=20)
    assert len(chunks) > 1
    assert chunks[0].start_time == 0.0
    assert chunks[-1].end_time == 100.0
