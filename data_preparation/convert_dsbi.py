"""
data_preparation/convert_dsbi.py

Converts DSBI Braille dataset annotations to YOLO format.

DSBI stores individual dot positions in +recto.txt / +verso.txt files.
Each non-blank, non-comment line has the x,y pixel coordinates as the
last two whitespace-separated tokens.

We group individual dots into Braille cells using flood-fill clustering,
then compute YOLO bounding boxes (class 0 = BrailleCell).
"""

import os
import shutil
import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Parse raw dot annotation file
# ---------------------------------------------------------------------------

def parse_dot_annotation_file(annotation_path: str) -> List[Tuple[float, float]]:
    """
    Read every non-blank, non-comment line from a DSBI dot file.
    Parse the last two whitespace-separated tokens as x, y floats.
    Returns list of (x, y) tuples.
    """
    dots = []
    first_lines_printed = False

    with open(annotation_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Print first 5 lines to verify format
    print(f"  [DEBUG] First 5 lines of {os.path.basename(annotation_path)}:")
    for line in lines[:5]:
        print(f"    {repr(line.rstrip())}")
    print()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[-2])
            y = float(parts[-1])
            dots.append((x, y))
        except (ValueError, IndexError):
            continue

    return dots


# ---------------------------------------------------------------------------
# 2. Estimate dot spacing
# ---------------------------------------------------------------------------

def estimate_dot_spacing(dots: List[Tuple[float, float]]) -> float:
    """
    Sample up to 200 dots, find nearest-neighbor distances, return median.
    Fallback: 20.0 if fewer than 2 dots.
    """
    if len(dots) < 2:
        return 20.0

    sample = dots[:200]
    pts = np.array(sample, dtype=np.float32)  # (N, 2)
    nn_dists = []

    for i, pt in enumerate(pts):
        diffs = pts - pt
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        dists[i] = np.inf  # exclude self
        nn_dists.append(float(np.min(dists)))

    return float(np.median(nn_dists))


# ---------------------------------------------------------------------------
# 3. Group dots into cells via flood-fill
# ---------------------------------------------------------------------------

def group_dots_into_cells(
    dots: List[Tuple[float, float]],
    dot_spacing: float,
    grouping_radius: float = 2.5,
) -> List[List[Tuple[float, float]]]:
    """
    Greedy flood-fill: each group is a Braille cell (1–6 dots).
    threshold = grouping_radius * dot_spacing
    """
    threshold = grouping_radius * dot_spacing
    assigned = [False] * len(dots)
    pts = np.array(dots, dtype=np.float32)
    groups = []

    for seed_idx in range(len(dots)):
        if assigned[seed_idx]:
            continue
        group_indices = [seed_idx]
        assigned[seed_idx] = True
        queue = [seed_idx]

        while queue:
            current = queue.pop(0)
            current_pt = pts[current]
            for j in range(len(dots)):
                if assigned[j]:
                    continue
                dist = float(np.hypot(pts[j, 0] - current_pt[0], pts[j, 1] - current_pt[1]))
                if dist <= threshold:
                    group_indices.append(j)
                    assigned[j] = True
                    queue.append(j)

        # Valid Braille cell: 1–6 dots
        if 1 <= len(group_indices) <= 6:
            groups.append([dots[i] for i in group_indices])

    return groups


# ---------------------------------------------------------------------------
# 4. Compute cell bounding box (YOLO normalised format)
# ---------------------------------------------------------------------------

def compute_cell_bbox(
    dot_group: List[Tuple[float, float]],
    img_w: int,
    img_h: int,
    padding: float = 0.15,
) -> Tuple[float, float, float, float]:
    """
    Returns (cx_norm, cy_norm, w_norm, h_norm) suitable for YOLO label.
    """
    xs = [p[0] for p in dot_group]
    ys = [p[1] for p in dot_group]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Ensure minimum 1px tight extent
    if max_x - min_x < 1:
        max_x = min_x + 1
    if max_y - min_y < 1:
        max_y = min_y + 1

    tight_w = max_x - min_x
    tight_h = max_y - min_y

    # Expand by padding
    min_x -= padding * tight_w
    max_x += padding * tight_w
    min_y -= padding * tight_h
    max_y += padding * tight_h

    # Clip to image bounds
    min_x = max(0.0, min_x)
    max_x = min(float(img_w), max_x)
    min_y = max(0.0, min_y)
    max_y = min(float(img_h), max_y)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    w  = max_x - min_x
    h  = max_y - min_y

    return (cx / img_w, cy / img_h, w / img_w, h / img_h)


# ---------------------------------------------------------------------------
# 5. Convert a single DSBI image + annotation pair
# ---------------------------------------------------------------------------

def convert_dsbi_image(
    image_path: str,
    annotation_path: str,
    output_image_dir: str,
    output_label_dir: str,
) -> int:
    """
    Process one image. Writes YOLO label file, copies image.
    Returns number of cells written.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Could not load image: {image_path}")
        return 0

    img_h, img_w = img.shape[:2]
    stem = Path(image_path).stem
    label_path = os.path.join(output_label_dir, f"{stem}.txt")
    out_img_path = os.path.join(output_image_dir, Path(image_path).name)

    dots = parse_dot_annotation_file(annotation_path)

    # Hard negative: fewer than 3 dots → empty label
    if len(dots) < 3:
        open(label_path, "w").close()
        shutil.copy2(image_path, out_img_path)
        return 0

    dot_spacing = estimate_dot_spacing(dots)
    groups = group_dots_into_cells(dots, dot_spacing)

    lines = []
    for group in groups:
        cx, cy, w, h = compute_cell_bbox(group, img_w, img_h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    shutil.copy2(image_path, out_img_path)
    return len(lines)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    DSBI_ROOT = "datasets/raw_sources/dsbi"
    YOLO_IMAGES = "datasets/processed_yolo/images"
    YOLO_LABELS = "datasets/processed_yolo/labels"

    if not os.path.isdir(DSBI_ROOT):
        print(f"[ERROR] DSBI root not found: {DSBI_ROOT}")
        print("  Place DSBI data under datasets/raw_sources/dsbi/ and re-run.")
        return

    # Read DSBI split assignment files if present
    stem_to_split = {}
    for split_file, split_name in [("train.txt", "train"), ("test.txt", "test")]:
        fpath = os.path.join(DSBI_ROOT, split_file)
        if os.path.isfile(fpath):
            with open(fpath) as f:
                for line in f:
                    stem = line.strip()
                    if stem:
                        stem_to_split[stem] = split_name

    IMAGE_EXTS = [".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG"]

    total_cells = 0
    test_counter = 0

    for root, _dirs, files in os.walk(DSBI_ROOT):
        for fname in files:
            if not fname.endswith("+recto.txt") and not fname.endswith("+verso.txt"):
                continue

            ann_path = os.path.join(root, fname)
            # Derive image stem: remove +recto.txt or +verso.txt suffix
            if fname.endswith("+recto.txt"):
                img_stem = fname[: -len("+recto.txt")]
            else:
                img_stem = fname[: -len("+verso.txt")]

            # Find matching image
            img_path = None
            for ext in IMAGE_EXTS:
                candidate = os.path.join(root, img_stem + ext)
                if os.path.isfile(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                print(f"  [WARN] No image found for annotation: {ann_path}")
                continue

            # Determine split
            if img_stem in stem_to_split:
                split = stem_to_split[img_stem]
                # Route every 5th test image to val
                if split == "test":
                    test_counter += 1
                    if test_counter % 5 == 0:
                        split = "val"
            else:
                split = "train"

            out_img_dir = os.path.join(YOLO_IMAGES, split)
            out_lbl_dir = os.path.join(YOLO_LABELS, split)
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)

            n = convert_dsbi_image(img_path, ann_path, out_img_dir, out_lbl_dir)
            total_cells += n
            print(f"  [OK] {img_stem} → {split} ({n} cells)")

    print(f"\n[DONE] Total cells written: {total_cells}")


if __name__ == "__main__":
    main()
