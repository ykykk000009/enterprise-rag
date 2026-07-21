"""Pluggable local or remote vision descriptions for extracted PDF images."""

from __future__ import annotations

from typing import Protocol


class ImageDescriptionProvider(Protocol):
    def describe_image(
        self,
        *,
        image_bytes: bytes,
        page_no: int,
        bbox: tuple[float, float, float, float],
        nearby_text: str,
        ocr_text: str,
    ) -> str:
        """Return a grounded image description suitable for retrieval."""
