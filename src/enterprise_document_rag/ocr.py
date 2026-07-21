"""Optional local OCR used only for low-text PDF pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pymupdf


@dataclass(frozen=True)
class OcrTextBlock:
    text: str
    bbox: tuple[float, float, float, float] | None
    confidence: float | None


class OcrProvider(Protocol):
    def extract_page(self, *, page: pymupdf.Page, dpi: int) -> tuple[OcrTextBlock, ...]: ...

    def extract_region(
        self,
        *,
        page: pymupdf.Page,
        bbox: tuple[float, float, float, float],
        dpi: int,
    ) -> tuple[OcrTextBlock, ...]: ...

    def extract_image(self, *, image_bytes: bytes) -> tuple[OcrTextBlock, ...]: ...


class RapidOcrProvider:
    """CPU-only OCR provider backed by RapidOCR and ONNX Runtime."""

    def __init__(self) -> None:
        self._engine = None

    def extract_page(self, *, page: pymupdf.Page, dpi: int) -> tuple[OcrTextBlock, ...]:
        import numpy as np

        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )
        return self._extract(image=image, coordinate_scale=scale, origin=(0.0, 0.0))

    def extract_region(
        self,
        *,
        page: pymupdf.Page,
        bbox: tuple[float, float, float, float],
        dpi: int,
    ) -> tuple[OcrTextBlock, ...]:
        import numpy as np

        scale = dpi / 72
        clip = pymupdf.Rect(bbox) & page.rect
        if clip.is_empty:
            return ()
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            clip=clip,
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )
        return self._extract(
            image=image,
            coordinate_scale=scale,
            origin=(float(clip.x0), float(clip.y0)),
        )

    def extract_image(self, *, image_bytes: bytes) -> tuple[OcrTextBlock, ...]:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return ()
        # Image-local coordinates let document parsers reconstruct raster tables.
        return self._extract(image=image, coordinate_scale=1.0, origin=(0.0, 0.0))

    def _extract(
        self,
        *,
        image,
        coordinate_scale: float | None,
        origin: tuple[float, float] | None,
    ) -> tuple[OcrTextBlock, ...]:
        from rapidocr_onnxruntime import RapidOCR

        if self._engine is None:
            self._engine = RapidOCR()
        result, _ = self._engine(image)
        if not result:
            return ()
        blocks: list[OcrTextBlock] = []
        for polygon, text, confidence in result:
            normalized = " ".join(str(text).split())
            if not normalized:
                continue
            coordinates = [(float(point[0]), float(point[1])) for point in polygon]
            if coordinate_scale is None:
                bbox = None
            else:
                origin_x, origin_y = origin or (0.0, 0.0)
                xs = [point[0] / coordinate_scale + origin_x for point in coordinates]
                ys = [point[1] / coordinate_scale + origin_y for point in coordinates]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            blocks.append(
                OcrTextBlock(
                    text=normalized,
                    bbox=bbox,
                    confidence=float(confidence),
                )
            )
        return tuple(blocks)
