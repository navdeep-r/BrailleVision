"""
inference/evaluate.py

End-to-end evaluation script.

Computes Character Error Rate (CER) and Word Error Rate (WER) on a test set
with ground-truth transcriptions, then saves per-image results to CSV.

Usage:
  python inference/evaluate.py \
    --test_dir  sample_inputs/ \
    --gt_csv    sample_inputs/ground_truth.csv \
    --output    sample_outputs/eval_results.csv

Ground truth CSV format:
  image_path, expected_text
"""

import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference import load_models, process_image
from inference.braille_decoder import BrailleDecoder


# ---------------------------------------------------------------------------
# Edit distance helpers
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(a), len(b)
    # Try fast path with python-Levenshtein if available
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    CER = Levenshtein(ref_chars, hyp_chars) / len(ref_chars).
    Returns 0.0 for empty reference.
    """
    if not reference:
        return 0.0
    ref = list(reference)
    hyp = list(hypothesis)
    dist = _edit_distance(ref, hyp)
    return dist / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    WER = Levenshtein(ref_words, hyp_words) / len(ref_words).
    Returns 0.0 for empty reference.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    dist = _edit_distance(ref_words, hyp_words)
    return dist / len(ref_words)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test_set(
    test_images_dir: str,
    ground_truth_csv: str,
    yolo_path: str,
    cnn_path: str,
    output_csv: str,
    grade: int = 2,
    device: str = "cpu",
    conf: float = 0.35,
    iou: float = 0.35,
) -> None:
    """
    Run the full BrailleVision pipeline on each image in the ground-truth CSV
    and compute CER and WER.
    """
    if not os.path.isfile(ground_truth_csv):
        print(f"[ERROR] Ground truth CSV not found: {ground_truth_csv}")
        print("  Expected columns: image_path, expected_text")
        return

    # Load models
    yolo_model, cnn_model = load_models(yolo_path, cnn_path, device)
    decoder = BrailleDecoder(grade=grade)

    # Fake args namespace for process_image
    class Args:
        use_dog = False
        visualize = False
        speak = False
        output_dir = os.path.dirname(output_csv) or "sample_outputs"
        pass
    args_fake = Args()
    args_fake.conf   = conf
    args_fake.iou    = iou
    args_fake.device = device
    args_fake.grade  = grade
    args_fake.use_dog = False
    args_fake.visualize = False
    args_fake.speak = False

    # Read ground truth
    with open(ground_truth_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        gt_rows = list(reader)

    print(f"[EVAL] {len(gt_rows)} test images")
    print(f"{'Image':<40} {'CER':>6}  {'WER':>6}")
    print("-" * 55)

    results = []
    cers, wers = [], []

    for row in gt_rows:
        img_path = row.get("image_path", row.get("image", "")).strip()
        expected = row.get("expected_text", row.get("text", "")).strip()

        if not os.path.isfile(img_path):
            full_path = os.path.join(test_images_dir, img_path)
            if os.path.isfile(full_path):
                img_path = full_path
            else:
                print(f"  [SKIP] Not found: {img_path}")
                continue

        hypothesis, _annot = process_image(img_path, yolo_model, cnn_model, decoder, args_fake)

        cer = character_error_rate(expected, hypothesis)
        wer = word_error_rate(expected, hypothesis)
        cers.append(cer)
        wers.append(wer)

        basename = os.path.basename(img_path)
        print(f"  {basename:<38} {cer:>6.3f}  {wer:>6.3f}")

        results.append({
            "image_path":     img_path,
            "expected_text":  expected,
            "hypothesis":     hypothesis,
            "cer":            round(cer, 4),
            "wer":            round(wer, 4),
        })

    # Save per-image results
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "expected_text", "hypothesis", "cer", "wer"])
        writer.writeheader()
        writer.writerows(results)

    if cers:
        print("\n" + "=" * 55)
        print(f"  Mean CER : {np.mean(cers):.4f}  ± {np.std(cers):.4f}")
        print(f"  Mean WER : {np.mean(wers):.4f}  ± {np.std(wers):.4f}")
        print("=" * 55)
        print(f"  Results saved to: {output_csv}")
    else:
        print("[WARN] No images evaluated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="BrailleVision evaluation")
    parser.add_argument("--test_dir",   type=str, default="sample_inputs")
    parser.add_argument("--gt_csv",     type=str, default="sample_inputs/ground_truth.csv")
    parser.add_argument("--output",     type=str, default="sample_outputs/eval_results.csv")
    parser.add_argument("--yolo",       type=str, default="model/best.pt")
    parser.add_argument("--cnn",        type=str, default="model/cell_classifier_best.pth")
    parser.add_argument("--grade",      type=int, default=2)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--conf",       type=float, default=0.35)
    parser.add_argument("--iou",        type=float, default=0.35)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_on_test_set(
        test_images_dir=args.test_dir,
        ground_truth_csv=args.gt_csv,
        yolo_path=args.yolo,
        cnn_path=args.cnn,
        output_csv=args.output,
        grade=args.grade,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )
