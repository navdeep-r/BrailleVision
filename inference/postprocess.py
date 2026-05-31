"""
inference/postprocess.py

Converts YOLO's unordered list of bounding boxes into a structured reading order
suitable for the Braille decoder.

Pipeline:
  filter_by_size
  → filter_by_edge
  → split_merged_cells
  → cluster_into_rows
  → [detect_spaces_in_row for each row]
  → reconstruct_reading_order (chains all above)

Each detection dict has keys:
  x1, y1, x2, y2, cx, cy, w, h, conf
  (pattern_int, cnn_conf added later by the inference script)

WHY running mean for row clustering:
  Using the running mean Y of the current row (not just the last element's Y)
  is more stable under camera tilt of ~10°.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# 1. Size filter
# ---------------------------------------------------------------------------

def filter_by_size(
    detections: List[Dict],
    tolerance: float = 0.30,
) -> List[Dict]:
    """
    Remove detections whose width or height differ >30% from the median.

    WHY: All Braille cells on one page are the same physical size.
    False positives (staples, logos, text) tend to be much smaller or larger.
    """
    if not detections:
        return []

    widths  = np.array([d["w"] for d in detections])
    heights = np.array([d["h"] for d in detections])
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))

    filtered = [
        d for d in detections
        if abs(d["w"] - med_w) / max(med_w, 1e-6) <= tolerance
        and abs(d["h"] - med_h) / max(med_h, 1e-6) <= tolerance
    ]
    return filtered


# ---------------------------------------------------------------------------
# 2. Edge filter
# ---------------------------------------------------------------------------

def filter_by_edge(
    detections: List[Dict],
    image_shape: Tuple[int, int, int],
    margin_px: int = 5,
) -> List[Dict]:
    """
    Remove detections whose box is within margin_px of any image edge.

    WHY: Edge cells have incomplete dot patterns → wrong classification.
    Better to discard than to output corrupt characters.
    """
    img_h, img_w = image_shape[:2]

    filtered = [
        d for d in detections
        if d["x1"] > margin_px
        and d["y1"] > margin_px
        and d["x2"] < img_w - margin_px
        and d["y2"] < img_h - margin_px
    ]
    return filtered


# ---------------------------------------------------------------------------
# 3. Split merged cells
# ---------------------------------------------------------------------------

def split_merged_cells(
    detections: List[Dict],
    max_width_ratio: float = 1.5,
) -> List[Dict]:
    """
    Detect boxes 1.5× wider than median and split into N equal-width slices.

    WHY: When two adjacent Braille cells touch, YOLO may merge them into one
    wide box. We split by the expected cell width.
    """
    if not detections:
        return []

    widths  = np.array([d["w"] for d in detections])
    med_w   = float(np.median(widths))
    threshold = max_width_ratio * med_w

    result = []
    for d in detections:
        if d["w"] > threshold:
            n = max(2, round(d["w"] / med_w))
            slice_w = d["w"] / n
            for i in range(n):
                new_x1 = d["x1"] + i * slice_w
                new_x2 = new_x1 + slice_w
                new_cx  = (new_x1 + new_x2) / 2.0
                slice_d = dict(d)
                slice_d.update({
                    "x1": new_x1,
                    "x2": new_x2,
                    "cx": new_cx,
                    "w":  slice_w,
                })
                result.append(slice_d)
        else:
            result.append(d)

    return result


# ---------------------------------------------------------------------------
# 4. Cluster into rows
# ---------------------------------------------------------------------------

def cluster_into_rows(
    detections: List[Dict],
    gap_factor: float = 0.6,
) -> List[List[Dict]]:
    """
    Group detections into rows using running-mean Y clustering.

    WHY running mean: More stable than using the last element's Y
    when the page is slightly tilted (~10°).

    WHY gap_factor=0.6: A gap > 60% of cell height separates rows.
    """
    if not detections:
        return []

    # Sort top-to-bottom
    sorted_dets = sorted(detections, key=lambda d: d["cy"])
    avg_h = float(np.mean([d["h"] for d in sorted_dets]))
    threshold = gap_factor * avg_h

    rows: List[List[Dict]] = []
    current_row: List[Dict] = [sorted_dets[0]]
    current_mean_cy = sorted_dets[0]["cy"]

    for d in sorted_dets[1:]:
        if abs(d["cy"] - current_mean_cy) <= threshold:
            current_row.append(d)
            # Update running mean
            current_mean_cy = float(np.mean([c["cy"] for c in current_row]))
        else:
            # Sort current row left-to-right
            rows.append(sorted(current_row, key=lambda c: c["cx"]))
            current_row = [d]
            current_mean_cy = d["cy"]

    rows.append(sorted(current_row, key=lambda c: c["cx"]))
    return rows


# ---------------------------------------------------------------------------
# 5. Detect spaces within a row
# ---------------------------------------------------------------------------

def detect_spaces_in_row(
    row: List[Dict],
    space_multiplier: float = 1.8,
) -> List[Optional[Dict]]:
    """
    Insert None sentinels between cells where the gap is large.

    Gap[i] = row[i+1].x1 - row[i].x2  (clamp to ≥ 0)
    Threshold = 1.8 × median gap

    Returns a list mixing dicts and Nones.
    """
    if len(row) <= 1:
        return list(row)

    gaps = []
    for i in range(len(row) - 1):
        gap = max(0.0, row[i + 1]["x1"] - row[i]["x2"])
        gaps.append(gap)

    median_gap = float(np.median(gaps))
    space_threshold = space_multiplier * median_gap

    result: List[Optional[Dict]] = [row[0]]
    for i, gap in enumerate(gaps):
        if gap > space_threshold:
            result.append(None)  # space sentinel
        result.append(row[i + 1])

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def reconstruct_reading_order(
    raw_detections: List[Dict],
    image_shape: Tuple[int, int, int],
) -> List[List[Optional[Dict]]]:
    """
    Convert an unordered list of YOLO detections into structured reading order.

    Returns list of rows; each row is a list of dicts-or-None (None = word space).
    """
    dets = filter_by_size(raw_detections)
    dets = filter_by_edge(dets, image_shape)
    dets = split_merged_cells(dets)
    rows = cluster_into_rows(dets)
    structured = [detect_spaces_in_row(row) for row in rows]
    return structured
