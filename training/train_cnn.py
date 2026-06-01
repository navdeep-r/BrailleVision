"""
training/train_cnn.py

Training script for the BrailleCell CNN classifier (MobileNetV3Small).

Targets:
  - Top-1 accuracy > 95% on clean test crops
  - Top-1 accuracy > 88% on augmented test crops

Key design decisions:
  - WeightedRandomSampler oversamples rare patterns (z, x, q)
  - label_smoothing=0.1 prevents overconfidence on ambiguous patterns
  - CosineAnnealingLR scheduler for smooth convergence
  - Gradient clipping (max_norm=1.0) stabilises early training
  - Early stopping (patience=20) prevents overfitting
"""

import os
import sys
import time
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# Allow running from root of project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.cell_dataset import build_dataloaders, get_transforms, BrailleCellDataset
from training.cell_model import BrailleCellClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CROPS_DIR   = "datasets/cell_crops"
MODEL_OUT   = "model/cell_classifier_best.pth"
EPOCHS      = 15
BATCH_SIZE  = 4096
LR          = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE    = 20
DEVICE_NAME = "cuda"


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    Returns (avg_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return (total_loss / total, correct / total)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate on a DataLoader.
    Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return (total_loss / total, correct / total)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str,
) -> None:
    """Save full training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy":        val_acc,
    }, path)


# ---------------------------------------------------------------------------
# Per-class accuracy breakdown
# ---------------------------------------------------------------------------

def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 65,
) -> Dict[int, float]:
    """
    Compute per-class accuracy.
    Returns dict {class_id: accuracy}.
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_cpu = labels.numpy()
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for p, t in zip(preds, labels_cpu):
                class_total[t] += 1
                if p == t:
                    class_correct[t] += 1

    result = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            result[c] = class_correct[c] / class_total[c]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device(DEVICE_NAME)
    print(f"[INFO] Using device: {device}")

    # Build dataloaders
    loaders = build_dataloaders(CROPS_DIR, batch_size=BATCH_SIZE)

    if loaders["train"] is None:
        print("[ERROR] No training data found. Run data preparation scripts first.")
        sys.exit(1)

    # Model
    model = BrailleCellClassifier(num_classes=BrailleCellClassifier.NUM_CLASSES)
    model.to(device)

    # Optimiser + scheduler + loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)

        val_loss, val_acc = (0.0, 0.0)
        if loaders["val"] is not None:
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"{epoch:>6d}  {train_loss:>11.4f}  {train_acc:>9.4f}  "
            f"{val_loss:>10.4f}  {val_acc:>8.4f}  {current_lr:>10.6f}  "
            f"[{elapsed:.1f}s]"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, MODEL_OUT)
            print(f"  [SAVED] Saved best model (val_acc={val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n[EARLY STOP] Patience {PATIENCE} exceeded at epoch {epoch}.")
            break

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.4f}")
    print(f"       Model saved to: {MODEL_OUT}")

    # --- Final test evaluation ---
    if loaders["test"] is not None:
        print("\n--- Loading best checkpoint for test evaluation ---")
        model_final = BrailleCellClassifier.load_for_inference(MODEL_OUT, device=DEVICE_NAME)
        test_loss, test_acc = evaluate(model_final, loaders["test"], criterion, device)
        print(f"[TEST] Loss={test_loss:.4f}  Accuracy={test_acc:.4f}")

        # Per-class breakdown
        per_class = evaluate_per_class(model_final, loaders["test"], device)
        print("\n[Per-class accuracy on test set]")
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1])

        print("  Worst 5 classes:")
        for cls_id, acc in sorted_classes[:5]:
            print(f"    pattern_int={cls_id:3d} (0b{cls_id:06b})  acc={acc:.3f}")

        print("  Best 5 classes:")
        for cls_id, acc in sorted_classes[-5:]:
            print(f"    pattern_int={cls_id:3d} (0b{cls_id:06b})  acc={acc:.3f}")

        if test_acc < 0.95:
            print(f"\n[WARNING] Test accuracy {test_acc:.3f} below target 0.95.")
            print("  Consider: more data, longer training, or stronger augmentation.")
        else:
            print(f"\n[TARGET MET] Test accuracy {test_acc:.3f} ≥ 0.95 ✓")


if __name__ == "__main__":
    main()
