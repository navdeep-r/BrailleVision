"""
data_preparation/convert_angelina.py

Converts the Angelina Braille dataset to:
  1. YOLO detection labels  (datasets/processed_yolo/)
  2. CNN cell crops + CSVs  (datasets/cell_crops/)

Angelina stores one JSON file per image. Each JSON has a "marks" key
containing a list of {"x","y","w","h","label"} objects where label is
the Braille character (a-z, punctuation, space).

Bit convention:
  bit 0 = dot 1 (top-left)    bit 3 = dot 4 (top-right)
  bit 1 = dot 2 (mid-left)    bit 4 = dot 5 (mid-right)
  bit 2 = dot 3 (bot-left)    bit 5 = dot 6 (bot-right)
"""

import os
import json
import shutil
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Character → 6-bit pattern mapping
# ---------------------------------------------------------------------------

CHAR_TO_PATTERN: Dict[str, int] = {
    "a": 0b000001, "b": 0b000011, "c": 0b001001, "d": 0b011001, "e": 0b010001,
    "f": 0b001011, "g": 0b011011, "h": 0b010011, "i": 0b001010, "j": 0b011010,
    "k": 0b000101, "l": 0b000111, "m": 0b001101, "n": 0b011101, "o": 0b010101,
    "p": 0b001111, "q": 0b011111, "r": 0b010111, "s": 0b001110, "t": 0b011110,
    "u": 0b100101, "v": 0b100111, "w": 0b111010, "x": 0b101101, "y": 0b111101,
    "z": 0b110101,
    ",": 0b000010, ";": 0b000110, ":": 0b010010, ".": 0b110010, "!": 0b010110,
    "?": 0b100110, "-": 0b100100, "'": 0b000100, " ": 0b000000,
    # Capital indicator
    "#": 0b111100,  # number indicator
    "@": 0b100000,  # capital indicator
}


# Hardcoded Angelina custom Cyrillic mapping
ANGELINA_MAPPING = {
    "а": "a", "б": "b", "ц": "c", "д": "d", "е": "e",
    "ф": "f", "г": "g", "х": "h", "и": "i", "ж": "j",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "ч": "q", "р": "r", "с": "s", "т": "t",
    "у": "u", "ѳ": "v", "в": "w", "щ": "x", "э": "y",
    "з": "z"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_image_for_json(json_path: str) -> Optional[str]:
    """Find the image file that matches a JSON annotation (same stem)."""
    stem = Path(json_path).stem
    parent = Path(json_path).parent
    for ext in [".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG", ".bmp", ".BMP"]:
        candidate = parent / (stem + ext)
        if candidate.is_file():
            return str(candidate)
    return None


def bbox_to_yolo(
    x: float, y: float, w: float, h: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert absolute top-left (x,y,w,h) to YOLO normalised (cx,cy,w,h)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return (cx, cy, w / img_w, h / img_h)


def extract_cell_crop(
    image: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    padding: float = 0.10,
    output_size: Tuple[int, int] = (64, 64),
) -> Optional[np.ndarray]:
    """
    Crop a single cell from the image with CLAHE normalisation.
    Returns a grayscale (H, W) uint8 array, or None if invalid.
    """
    img_h, img_w = image.shape[:2]

    pad_x = padding * w
    pad_y = padding * h
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(img_w, int(x + w + pad_x))
    y2 = min(img_h, int(y + h + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Per-crop CLAHE — mandatory for lighting normalisation
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Resize
    result = cv2.resize(gray, output_size, interpolation=cv2.INTER_LINEAR)
    return result


# ---------------------------------------------------------------------------
# Load Angelina annotations
# ---------------------------------------------------------------------------

def load_angelina_annotations(image_dir: str) -> List[Dict]:
    """
    Walk the directory recursively for .json files.
    Returns list of {"image_path", "marks", "source_dir"} dicts.
    """
    records = []
    for root, _dirs, files in os.walk(image_dir):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            json_path = os.path.join(root, fname)
            img_path = _find_image_for_json(json_path)
            if img_path is None:
                continue

            try:
                with open(json_path, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [WARN] Could not parse {json_path}: {e}")
                continue

            # Adapt key names — Angelina may use 'marks', 'labeled_cells', 'cells', or 'shapes'
            marks = data.get("marks", data.get("labeled_cells", data.get("cells", data.get("shapes", []))))
            if not isinstance(marks, list):
                marks = []

            records.append({
                "image_path": img_path,
                "marks": marks,
                "source_dir": root,
                "json_path": json_path,
            })

    return records


# ---------------------------------------------------------------------------
# Convert one image record
# ---------------------------------------------------------------------------

def convert_angelina_image(
    record: Dict,
    output_image_dir: str,
    output_label_dir: str,
    output_crop_dir: str,
    split: str,
    crop_csv_rows: List[List],
) -> int:
    """
    Process one image from the Angelina dataset.
    Returns number of cells written.
    """
    img_path = record["image_path"]
    marks = record["marks"]

    img = cv2.imread(img_path)
    if img is None:
        return 0

    img_h, img_w = img.shape[:2]
    stem = Path(img_path).stem

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_crop_dir, exist_ok=True)

    label_lines = []
    cell_count = 0

    for i, mark in enumerate(marks):
        # Extract bounding box (handle different key conventions)
        try:
            if "points" in mark and len(mark["points"]) == 2:
                x1, y1 = mark["points"][0]
                x2, y2 = mark["points"][1]
                x = float(min(x1, x2))
                y = float(min(y1, y2))
                w = float(abs(x2 - x1))
                h = float(abs(y2 - y1))
                char = str(mark.get("label", "?")).lower()
            else:
                x = float(mark.get("x", mark.get("left", 0)))
                y = float(mark.get("y", mark.get("top", 0)))
                w = float(mark.get("w", mark.get("width", 0)))
                h = float(mark.get("h", mark.get("height", 0)))
                char = str(mark.get("label", mark.get("char", mark.get("character", "?")))).lower()
        except (TypeError, ValueError):
            continue

        if w <= 0 or h <= 0:
            continue

        # YOLO detection line (always write, even for unknown chars)
        cx, cy, wn, hn = bbox_to_yolo(x, y, w, h, img_w, img_h)
        label_lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

        # CNN crop: Map using angelina_charset.json if available
        if char in ANGELINA_MAPPING:
            char = ANGELINA_MAPPING[char]

        # Only accept known characters; skip unknown/unmapped cells for CNN training
        if char not in CHAR_TO_PATTERN:
            continue

        pattern_int = CHAR_TO_PATTERN[char]
        crop = extract_cell_crop(img, x, y, w, h)
        if crop is None:
            continue

        crop_fname = f"{stem}_cell{i:04d}.png"
        crop_path = os.path.join(output_crop_dir, crop_fname)
        cv2.imwrite(crop_path, crop)

        crop_csv_rows.append([
            crop_fname,
            pattern_int,
            char,
            record["source_dir"],
        ])
        cell_count += 1

    # Write YOLO label file
    label_path = os.path.join(output_label_dir, f"{stem}.txt")
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))
        if label_lines:
            f.write("\n")

    # Copy image
    dest_img = os.path.join(output_image_dir, Path(img_path).name)
    if not os.path.exists(dest_img):
        shutil.copy2(img_path, dest_img)

    return cell_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANGELINA_ROOT = "datasets/raw_sources/angelina"
    YOLO_IMAGES   = "datasets/processed_yolo/images"
    YOLO_LABELS   = "datasets/processed_yolo/labels"
    CROPS_ROOT    = "datasets/cell_crops"

    # Split ratios per Angelina subset
    SUBSET_SPLITS = {
        "books":       {"train": 0.70, "val": 0.15, "test": 0.15},
        "handwritten": {"train": 0.60, "val": 0.20, "test": 0.20},
        "pics":        {"train": 0.70, "val": 0.15, "test": 0.15},
        "uploaded":    {"train": 0.00, "val": 0.00, "test": 1.00},
        "not_braille": {"train": 1.00, "val": 0.00, "test": 0.00},
    }

    if not os.path.isdir(ANGELINA_ROOT):
        print(f"[ERROR] Angelina root not found: {ANGELINA_ROOT}")
        print("  Place Angelina data under datasets/raw_sources/angelina/ and re-run.")
        return

    # CSV accumulators per split
    crop_csv: Dict[str, List] = {"train": [], "val": [], "test": []}

    total_cells = 0

    for subset_name, ratios in SUBSET_SPLITS.items():
        subset_dir = os.path.join(ANGELINA_ROOT, subset_name)
        if not os.path.isdir(subset_dir):
            print(f"  [SKIP] Subset not found: {subset_dir}")
            continue

        is_hard_negative = subset_name == "not_braille"
        records = load_angelina_annotations(subset_dir)

        if not records:
            print(f"  [WARN] No valid records in subset: {subset_name}")
            continue

        # Assign splits by consecutive index (keep pages together)
        n = len(records)
        n_train = int(n * ratios["train"])
        n_val   = int(n * ratios["val"])

        split_assignments = (
            ["train"] * n_train +
            ["val"]   * n_val +
            ["test"]  * (n - n_train - n_val)
        )

        print(f"\n[Subset: {subset_name}]  {n} images -> "
              f"train={n_train}, val={n_val}, test={n - n_train - n_val}")

        for record, split in zip(records, split_assignments):
            if ratios[split] == 0.0:
                # Safety: skip if ratio is 0 (shouldn't happen but guard)
                continue

            out_img_dir  = os.path.join(YOLO_IMAGES, split)
            out_lbl_dir  = os.path.join(YOLO_LABELS, split)
            out_crop_dir = os.path.join(CROPS_ROOT, split)

            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)
            os.makedirs(out_crop_dir, exist_ok=True)

            if is_hard_negative:
                # Copy image + empty label — teaches YOLO zero cells exist here
                img_path = record["image_path"]
                img = cv2.imread(img_path)
                if img is None:
                    continue
                stem = Path(img_path).stem
                label_path = os.path.join(out_lbl_dir, f"{stem}.txt")
                open(label_path, "w").close()
                dest = os.path.join(out_img_dir, Path(img_path).name)
                if not os.path.exists(dest):
                    shutil.copy2(img_path, dest)
                continue

            n_cells = convert_angelina_image(
                record,
                out_img_dir,
                out_lbl_dir,
                out_crop_dir,
                split,
                crop_csv[split],
            )
            total_cells += n_cells

    # Write per-split CSV files
    csv_columns = ["filename", "pattern_int", "character", "source"]
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(CROPS_ROOT, f"{split}_labels.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_columns)
            writer.writerows(crop_csv[split])
        print(f"\n[CSV] {csv_path}  ({len(crop_csv[split])} rows)")

    print(f"\n[DONE] Total CNN crop cells written: {total_cells}")


if __name__ == "__main__":
    main()
