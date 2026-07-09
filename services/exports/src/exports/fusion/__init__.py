"""Temporal signal fusion for multi-index text search (Step 5)."""

from exports.fusion.classifier import QueryClass, classify_query
from exports.fusion.pipeline import (
    candidate_counts_for,
    fuse_and_rank,
    group_by_timestamp,
    intersection_multiplier,
    normalize_scores_by_type,
)

__all__ = [
    "QueryClass",
    "classify_query",
    "candidate_counts_for",
    "fuse_and_rank",
    "group_by_timestamp",
    "intersection_multiplier",
    "normalize_scores_by_type",
]
