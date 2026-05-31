"""
scaffold.py — Create the full BrailleVision directory tree.
Run once before doing anything else.
"""
import os

dirs = [
    "datasets/raw_sources/dsbi",
    "datasets/raw_sources/angelina",
    "datasets/processed_yolo/images/train",
    "datasets/processed_yolo/images/val",
    "datasets/processed_yolo/images/test",
    "datasets/processed_yolo/labels/train",
    "datasets/processed_yolo/labels/val",
    "datasets/processed_yolo/labels/test",
    "datasets/cell_crops/train",
    "datasets/cell_crops/val",
    "datasets/cell_crops/test",
    "datasets/metadata",
    "data_preparation",
    "training",
    "inference",
    "backend",
    "frontend",
    "model",
    "runs/detect",
    "runs/classify",
    "sample_inputs",
    "sample_outputs",
    "demo/screenshots",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    gk = os.path.join(d, ".gitkeep")
    if not os.listdir(d):
        open(gk, "w").close()

print("Scaffold created.")
