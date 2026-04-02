from __future__ import annotations

from ._ndarray_backend import xp


def _clip(val, lo, hi):
    return max(lo, min(int(val), hi))


def normalize_boxes(boxes, padding: int = 0, shape: tuple[int, int] | None = None):
    """Normalize user-provided ROI boxes to clipped integer tuples."""
    norm = []
    if boxes is None:
        return norm
    for box in boxes:
        if isinstance(box, (list, tuple)) and len(box) == 4:
            y0, y1, x0, x1 = box
        elif isinstance(box, tuple) and len(box) == 2:
            ys, xs = box
            y0, y1 = ys.start or 0, ys.stop
            x0, x1 = xs.start or 0, xs.stop
        else:
            raise ValueError("ROI boxes must be (y0, y1, x0, x1) tuples or "
                             "(slice_y, slice_x) tuples.")

        if padding:
            y0 -= padding
            x0 -= padding
            y1 += padding
            x1 += padding

        if shape is not None:
            y0 = _clip(y0, 0, shape[0])
            y1 = _clip(y1, 0, shape[0])
            x0 = _clip(x0, 0, shape[1])
            x1 = _clip(x1, 0, shape[1])

        if y1 > y0 and x1 > x0:
            norm.append((y0, y1, x0, x1))
    return norm


def boxes_from_mask(mask, padding: int = 0, shape: tuple[int, int] | None = None):
    """Infer a single bounding box from a boolean mask."""
    mask_arr = xp.asarray(mask)
    if mask_arr.ndim > 2:
        mask_arr = mask_arr.any(axis=0)
    coords = xp.argwhere(mask_arr)
    if coords.size == 0:
        return []
    y0 = int(coords[:, 0].min())
    y1 = int(coords[:, 0].max()) + 1
    x0 = int(coords[:, 1].min())
    x1 = int(coords[:, 1].max()) + 1
    return normalize_boxes([(y0, y1, x0, x1)], padding=padding, shape=shape)


def merge_boxes(boxes):
    """Merge overlapping boxes to reduce duplicate processing."""
    if not boxes:
        return []
    merged = []
    for box in boxes:
        added = False
        for i, m in enumerate(merged):
            if _overlap(box, m):
                merged[i] = _merge_two(box, m)
                added = True
                break
        if not added:
            merged.append(box)
    return merged


def _overlap(b1, b2):
    return not (b1[1] <= b2[0] or b2[1] <= b1[0]
                or b1[3] <= b2[2] or b2[3] <= b1[2])


def _merge_two(b1, b2):
    return (min(b1[0], b2[0]),
            max(b1[1], b2[1]),
            min(b1[2], b2[2]),
            max(b1[3], b2[3]))

