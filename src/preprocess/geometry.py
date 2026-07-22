from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import fitz

BBox = Tuple[float, float, float, float]


def bbox_to_list(bbox: Optional[BBox]) -> Optional[List[float]]:
    if bbox is None:
        return None
    return [round(float(v), 2) for v in bbox]


def bbox_intersects(a: Optional[BBox], b: Optional[BBox]) -> bool:
    if a is None or b is None:
        return False
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def rect_area(rect: fitz.Rect) -> float:
    return max(0.0, rect.width) * max(0.0, rect.height)


def clamp_bbox(bbox: BBox, page_rect: fitz.Rect, margin: float = 4.0) -> fitz.Rect:
    x0, y0, x1, y1 = bbox
    return fitz.Rect(
        max(page_rect.x0, x0 - margin),
        max(page_rect.y0, y0 - margin),
        min(page_rect.x1, x1 + margin),
        min(page_rect.y1, y1 + margin),
    )


def render_clip(page: fitz.Page, bbox: BBox, output_path: Path, *, dpi: int = 144) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = clamp_bbox(bbox, page.rect)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    pix.save(str(output_path))
    del pix
    return output_path.as_posix()
