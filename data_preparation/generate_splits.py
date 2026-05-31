"""
data_preparation/generate_splits.py

Verifies label-image pairing, writes data.yaml for YOLO training,
and generates metadata CSVs (split manifest + class distribution).

Run AFTER convert_dsbi.py and convert_angelina.py.
"""

import os
import csv
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List

YOLO_ROOT  = "datasets/processed_yolo"
CROPS_ROOT = "datasets/cell_crops"
META_DIR   = "datasets/metadata"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"}


# ---------------------------------------------------------------------------
# 1. Verify label-image pairing
# ---------------------------------------------------------------------------

def verify_label_image_pairing(yolo_root: str) -> bool:
    """
    For each split, find all image stems and label stems.
    Print mismatches. Return True if all clean.
    """
    all_clean = True
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(yolo_root, "images", split)
        lbl_dir = os.path.join(yolo_root, "labels", split)

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            print(f"  [WARN] Missing directory for split '{split}'")
            continue

        img_stems = {
            Path(f).stem
            for f in os.listdir(img_dir)
            if Path(f).suffix in IMAGE_EXTS
        }
        lbl_stems = {
            Path(f).stem
            for f in os.listdir(lbl_dir)
            if f.endswith(".txt")
        }

        imgs_without_labels = img_stems - lbl_stems
        labels_without_imgs = lbl_stems - img_stems

        if imgs_without_labels:
            print(f"  [MISMATCH {split}] Images without labels ({len(imgs_without_labels)}):")
            for s in sorted(imgs_without_labels)[:10]:
                print(f"    {s}")
            all_clean = False

        if labels_without_imgs:
            print(f"  [MISMATCH {split}] Labels without images ({len(labels_without_imgs)}):")
            for s in sorted(labels_without_imgs)[:10]:
                print(f"    {s}")
            all_clean = False

        if not imgs_without_labels and not labels_without_imgs:
            print(f"  [OK] {split}: {len(img_stems)} images, {len(lbl_stems)} labels — all paired.")

    return all_clean


# ---------------------------------------------------------------------------
# 2. Write data.yaml for YOLO
# ---------------------------------------------------------------------------

def write_data_yaml(yolo_root: str, output_path: str) -> None:
    """Write the YOLO data.yaml with absolute paths."""
    abs_root = os.path.abspath(yolo_root)
    config = {
        "path":  abs_root,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": ["BrailleCell"],
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  [OK] Wrote {output_path}")


# ---------------------------------------------------------------------------
# 3. Count annotations per split
# ---------------------------------------------------------------------------

def count_annotations_per_split(yolo_root: str) -> pd.DataFrame:
    """
    Count images and total cell bounding boxes per split.
    Writes to datasets/metadata/split_manifest.csv.
    """
    rows = []
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(yolo_root, "images", split)
        lbl_dir = os.path.join(yolo_root, "labels", split)

        if not os.path.isdir(img_dir):
            rows.append({"split": split, "images": 0, "cells": 0})
            continue

        n_images = sum(
            1 for f in os.listdir(img_dir)
            if Path(f).suffix in IMAGE_EXTS
        )
        n_cells = 0
        if os.path.isdir(lbl_dir):
            for f in os.listdir(lbl_dir):
                if not f.endswith(".txt"):
                    continue
                with open(os.path.join(lbl_dir, f)) as fp:
                    n_cells += sum(1 for line in fp if line.strip())

        rows.append({"split": split, "images": n_images, "cells": n_cells})

    df = pd.DataFrame(rows)
    out = os.path.join(META_DIR, "split_manifest.csv")
    os.makedirs(META_DIR, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  [OK] Wrote {out}")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 4. Compute CNN class distribution
# ---------------------------------------------------------------------------

def compute_class_distribution(crops_root: str) -> pd.DataFrame:
    """
    Count examples per pattern_int per split.
    Flags any class with < 50 training examples.
    Writes to datasets/metadata/class_distribution.csv.
    """
    dfs = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(crops_root, f"{split}_labels.csv")
        if not os.path.isfile(csv_path):
            print(f"  [WARN] CSV not found: {csv_path}")
            dfs[split] = pd.DataFrame(columns=["pattern_int", "character", "count"])
            continue
        df = pd.read_csv(csv_path)
        counts = (
            df.groupby(["pattern_int", "character"])
            .size()
            .reset_index(name="count")
        )
        counts = counts.rename(columns={"count": f"count_{split}"})
        dfs[split] = counts

    # Merge all splits
    merged = dfs["train"]
    for split in ["val", "test"]:
        merged = merged.merge(dfs[split], on=["pattern_int", "character"], how="outer")

    merged = merged.fillna(0)
    for col in ["count_train", "count_val", "count_test"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)

    # Flag rare classes
    if "count_train" in merged.columns:
        rare = merged[merged["count_train"] < 50]
        if not rare.empty:
            print(f"\n  [WARNING] {len(rare)} classes with < 50 training examples:")
            print(rare[["pattern_int", "character", "count_train"]].to_string(index=False))

    out = os.path.join(META_DIR, "class_distribution.csv")
    os.makedirs(META_DIR, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"  [OK] Wrote {out}")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Step 1: Verify label-image pairing")
    print("=" * 60)
    clean = verify_label_image_pairing(YOLO_ROOT)
    if not clean:
        print("\n[WARNING] Mismatches found above. Fix before training.")

    print("\n" + "=" * 60)
    print("Step 2: Write data.yaml")
    print("=" * 60)
    data_yaml_path = os.path.join(YOLO_ROOT, "data.yaml")
    write_data_yaml(YOLO_ROOT, data_yaml_path)

    # Write classes.txt
    classes_path = os.path.join(YOLO_ROOT, "classes.txt")
    with open(classes_path, "w") as f:
        f.write("BrailleCell\n")
    print(f"  [OK] Wrote {classes_path}")

    print("\n" + "=" * 60)
    print("Step 3: Count annotations per split")
    print("=" * 60)
    count_annotations_per_split(YOLO_ROOT)

    print("\n" + "=" * 60)
    print("Step 4: Compute CNN class distribution")
    print("=" * 60)
    compute_class_distribution(CROPS_ROOT)

    print("\n[DONE] generate_splits.py complete.")


if __name__ == "__main__":
    main()
