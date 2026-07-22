from __future__ import annotations

from typing import Any, Dict, List, Sequence

import fitz

try:
    from .geometry import BBox, bbox_intersects, rect_area
except ImportError:
    from geometry import BBox, bbox_intersects, rect_area


def detect_visual_regions(page: fitz.Page, table_bboxes: Sequence[BBox]) -> List[Dict[str, Any]]:
    visual_regions: List[Dict[str, Any]] = []
    page_area = rect_area(page.rect)
    image_blocks = [block for block in page.get_text("dict").get("blocks", []) if block.get("type") == 1]

    for block in image_blocks:
        bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))  # type: ignore[assignment]
        if rect_area(fitz.Rect(bbox)) < page_area * 0.01:
            continue
        visual_regions.append(
            {
                "visual_index": len(visual_regions) + 1,
                "kind": "image",
                "bbox": bbox,
                "reason": "embedded_image",
            }
        )

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []

    drawing_rects = [
        fitz.Rect(drawing.get("rect"))
        for drawing in drawings
        if drawing.get("rect") is not None and rect_area(fitz.Rect(drawing.get("rect"))) > 4
    ]
    drawing_rects = [
        rect
        for rect in drawing_rects
        if not any(bbox_intersects((rect.x0, rect.y0, rect.x1, rect.y1), table_bbox) for table_bbox in table_bboxes)
    ]

    if len(drawing_rects) >= 5:
        union = fitz.Rect(drawing_rects[0])
        for rect in drawing_rects[1:]:
            union.include_rect(rect)
        if rect_area(union) >= page_area * 0.02:
            visual_regions.append(
                {
                    "visual_index": len(visual_regions) + 1,
                    "kind": "diagram",
                    "bbox": (union.x0, union.y0, union.x1, union.y1),
                    "reason": f"{len(drawing_rects)} vector drawing elements",
                }
            )

    return visual_regions
