"""Captioning pipeline stages (scene detect → budget → extract → window → caption → merge)."""

from .types import CaptionDraft, Scene, Window

__all__ = ["CaptionDraft", "Scene", "Window"]
