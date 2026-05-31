"""
training/train_yolo.py

YOLOv8s training script for Braille cell detection.

Target metrics:
  mAP@0.5 > 0.85
  Recall  > Precision  (missed cell = missing character — recall is paramount)

Key augmentation notes:
  - flipud=0.0  → NEVER flip vertically (Braille is directional)
  - hsv_v=0.4   → Aggressive brightness aug for lighting robustness
  - mosaic=1.0  → Mosaic augmentation for dense-detection robustness
  - NMS IoU=0.35 → Tightly packed cells need lower IoU than default 0.7

After training, run tune_thresholds() to find the optimal conf/iou pair.
"""

import os
import sys
import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_YAML  = "datasets/processed_yolo/data.yaml"
BEST_PT    = "model/best.pt"
RUN_NAME   = "braille_cell_detector"
PROJECT    = "runs/detect"


# ---------------------------------------------------------------------------
# 1. Dataset verification
# ---------------------------------------------------------------------------

def verify_dataset(data_yaml_path: str) -> bool:
    """
    Check that train/val image dirs exist and have enough files.
    Returns True if all checks pass.
    """
    import yaml

    if not os.path.isfile(data_yaml_path):
        print(f"[ERROR] data.yaml not found: {data_yaml_path}")
        print("  Run data_preparation/generate_splits.py first.")
        return False

    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("path", ".")
    ok = True
    for split in ["train", "val"]:
        img_dir = os.path.join(root, cfg.get(split, f"images/{split}"))
        if not os.path.isdir(img_dir):
            print(f"[ERROR] Directory not found: {img_dir}")
            ok = False
            continue
        n_files = sum(
            1 for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        if n_files < 5:
            print(f"[ERROR] Too few images in {img_dir}: found {n_files}, need ≥ 5")
            ok = False
        else:
            print(f"  [OK] {split}: {n_files} images in {img_dir}")

    return ok


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def run_training(data_yaml_path: str, device: str = "0") -> str:
    """
    Train YOLOv8s on the Braille cell detection task.
    Returns path to best.pt weights.
    """
    model = YOLO("yolov8s.pt")

    results = model.train(
        data=data_yaml_path,
        epochs=150,
        imgsz=640,
        batch=16,
        patience=30,
        device=device,
        workers=4,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        warmup_momentum=0.8,
        optimizer="AdamW",
        # Colour/brightness augmentation — KEY for lighting robustness
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        perspective=0.0005,
        flipud=0.0,     # NEVER flip vertically — Braille is directional
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        project=PROJECT,
        name=RUN_NAME,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        iou=0.7,
        conf=0.001,
    )

    # Path to best weights produced by Ultralytics
    run_dir = results.save_dir
    best_path = os.path.join(str(run_dir), "weights", "best.pt")
    print(f"\n[INFO] Best weights: {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_path: str, data_yaml_path: str) -> None:
    """
    Run validation on the test split and print key metrics.
    """
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml_path,
        split="test",
        conf=0.35,
        iou=0.35,
        plots=True,
        save_json=True,
    )

    map50   = metrics.box.map50
    map5095 = metrics.box.map
    precision = float(metrics.box.mp)
    recall    = float(metrics.box.mr)

    print("\n" + "=" * 50)
    print("YOLO Evaluation Results (test split)")
    print("=" * 50)
    print(f"  mAP@0.5       : {map50:.4f}  (target > 0.85)")
    print(f"  mAP@0.5:0.95  : {map5095:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}  (should be > Precision)")
    print("=" * 50)

    if map50 < 0.85:
        print("[WARNING] mAP@0.5 below target. Consider more data or longer training.")
    if recall < precision:
        print("[WARNING] Recall < Precision — tune NMS thresholds to favour recall.")


# ---------------------------------------------------------------------------
# 4. NMS threshold sweep
# ---------------------------------------------------------------------------

def tune_thresholds(model_path: str, data_yaml_path: str) -> None:
    """
    Sweep conf × iou to find the combination with highest F1 on val split.
    """
    model = YOLO(model_path)

    conf_vals = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    iou_vals  = [0.25, 0.30, 0.35, 0.40]

    best_f1   = -1.0
    best_conf = 0.35
    best_iou  = 0.35

    print("\n[NMS Threshold Sweep on val split]")
    print(f"{'conf':>6}  {'iou':>5}  {'P':>7}  {'R':>7}  {'F1':>7}")

    for conf in conf_vals:
        for iou in iou_vals:
            metrics = model.val(
                data=data_yaml_path,
                split="val",
                conf=conf,
                iou=iou,
                verbose=False,
            )
            p = float(metrics.box.mp)
            r = float(metrics.box.mr)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            print(f"  {conf:.2f}   {iou:.2f}   {p:.4f}   {r:.4f}   {f1:.4f}")

            if f1 > best_f1:
                best_f1   = f1
                best_conf = conf
                best_iou  = iou

    print(f"\n[BEST] conf={best_conf}, iou={best_iou}, F1={best_f1:.4f}")
    print(f"  Use these values for --conf {best_conf} --iou {best_iou} in inference.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Step 1: Verify dataset")
    print("=" * 60)
    if not verify_dataset(DATA_YAML):
        print("\n[ABORT] Fix dataset issues before training.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Step 2: Train YOLO")
    print("=" * 60)
    # Use GPU if available, otherwise CPU
    device = "0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    best_path = run_training(DATA_YAML, device=device)

    print("\n" + "=" * 60)
    print("Step 3: Evaluate on test split")
    print("=" * 60)
    evaluate_model(best_path, DATA_YAML)

    print("\n" + "=" * 60)
    print("Step 4: Copy best.pt to model/")
    print("=" * 60)
    os.makedirs("model", exist_ok=True)
    shutil.copy2(best_path, BEST_PT)
    print(f"  [OK] Copied to {BEST_PT}")

    print("\n" + "=" * 60)
    print("Step 5: NMS threshold sweep")
    print("=" * 60)
    tune_thresholds(BEST_PT, DATA_YAML)

    print("\n[DONE] train_yolo.py complete.")


if __name__ == "__main__":
    main()
