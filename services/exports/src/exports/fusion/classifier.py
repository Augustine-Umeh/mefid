"""Regex-based query classification for weighted candidate retrieval."""

from __future__ import annotations

import re
from enum import Enum, auto

VISUAL_PATTERNS: tuple[str, ...] = (
    r"\b(red|blue|green|yellow|black|white|orange|purple|pink|brown|grey|gray|golden|silver)\b",
    r"\b(striped|spotted|leather|metallic|shiny|dark|bright|colorful)\b",
    r"\b(wearing|holding|dressed|outfit|suit|dress|shirt|jacket|tie|hat|shoes|uniform|jersey|coat|pants)\b",
    r"\b(background|foreground|next to|inside|under|on top of|beside|behind)\b",
    r"\b(scene|setting|room|outdoor|indoor|crowd|empty|wide shot|close up)\b",
    r"what does .+ look like",
    r"show me .+",
)

SPEECH_PATTERNS: tuple[str, ...] = (
    r"\b(said|says|saying|talked|discussed|mentioned|argued|stated|told|announced|explained|asked)\b",
    r"\b(dialogue|conversation|speech|quote|statement|topic|theme|opinion|thoughts on)\b",
    r"\b(what did .+ say|where did .+ mention|when did .+ talk)\b",
    r'"[^"]+"',
    r"'[^']+'",
)


class QueryClass(Enum):
    VISUAL = auto()
    SPEECH = auto()
    AMBIGUOUS = auto()


def classify_query(query: str) -> QueryClass:
    """Classify a text query as visual-, speech-, or mixed-intent."""
    q = query.lower()
    visual_score = sum(1 for pattern in VISUAL_PATTERNS if re.search(pattern, q))
    speech_score = sum(1 for pattern in SPEECH_PATTERNS if re.search(pattern, q))

    if visual_score > 0 and speech_score == 0:
        return QueryClass.VISUAL
    if speech_score > 0 and visual_score == 0:
        return QueryClass.SPEECH
    return QueryClass.AMBIGUOUS
