"""
inference/inference.py

Main inference script — ties every module together into the complete pipeline.

Pipeline order:
  1. preprocess_full       (blur check + perspective + CLAHE)
  2. run_yolo_detection    (find WHERE cells are)
  3. classify_cell         (identify WHAT dot pattern each cell has)
  4. reconstruct_reading_order  (sort into rows, detect spaces)
  5. decoder.decode_page   (convert patterns to text)
  6. draw_annotations      (visualise result)

Usage examples:
  python inference/inference.py --source sample_inputs/test.jpg --visualize --speak
  python inference/inference.py --source 0  # webcam
"""

import os
import sys
import cv2
import numpy as np
import argparse
import torch
from torchvision import transforms
from typing import Optional, Tuple, List, Dict

# Allow running from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from inference.preprocess import preprocess_full
from inference.postprocess import reconstruct_reading_order
from inference.braille_decoder import BrailleDecoder, build_grade1_table
from inference.tts_engine import TTSEngine
from training.cell_model import BrailleCellClassifier


# ---------------------------------------------------------------------------
# Module-level CNN transform (must match training exactly)
# ---------------------------------------------------------------------------

CNN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Grade 1 table for annotation overlay
_G1_TABLE = build_grade1_table()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BrailleVision Inference")
    parser.add_argument("--source",       type=str,   default="sample_inputs/test.jpg",
                        help="Image path, or '0' for webcam")
    parser.add_argument("--yolo_weights", type=str,   default="model/best.pt")
    parser.add_argument("--cnn_weights",  type=str,   default="model/cell_classifier_best.pth")
    parser.add_argument("--conf",         type=float, default=0.35,
                        help="YOLO detection confidence threshold")
    parser.add_argument("--iou",          type=float, default=0.35,
                        help="YOLO NMS IoU threshold")
    parser.add_argument("--grade",        type=int,   default=2,
                        help="Braille grade (1 or 2)")
    parser.add_argument("--visualize",    action="store_true",
                        help="Display annotated result")
    parser.add_argument("--speak",        action="store_true",
                        help="Speak decoded text via TTS")
    parser.add_argument("--output_dir",   type=str,   default="sample_outputs",
                        help="Directory to save annotated images")
    parser.add_argument("--use_dog",      action="store_true",
                        help="Apply Difference-of-Gaussians filter (for flat lighting)")
    parser.add_argument("--device",       type=str,   default="cpu",
                        help="PyTorch device for CNN (cpu or cuda)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    yolo_path: str,
    cnn_path:  str,
    device:    str,
) -> Tuple[Optional[YOLO], Optional[BrailleCellClassifier]]:
    """
    Load YOLO and CNN models once at startup.
    Returns (yolo_model, cnn_model). Either may be None if weights are missing.
    """
    yolo_model = None
    cnn_model  = None

    if os.path.isfile(yolo_path):
        yolo_model = YOLO(yolo_path)
        print(f"[INFO] YOLO loaded from {yolo_path}")
    else:
        print(f"[WARN] YOLO weights not found: {yolo_path}")

    if os.path.isfile(cnn_path):
        cnn_model = BrailleCellClassifier.load_for_inference(cnn_path, device=device)
        print(f"[INFO] CNN loaded from {cnn_path}")
    else:
        print(f"[WARN] CNN weights not found: {cnn_path}")

    return (yolo_model, cnn_model)


# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------

def run_yolo_detection(
    yolo_model: YOLO,
    image:      np.ndarray,
    conf:       float,
    iou:        float,
) -> List[Dict]:
    """
    Run YOLO on the preprocessed image.
    Returns a list of detection dicts with keys:
      x1, y1, x2, y2, cx, cy, w, h, conf
    """
    results = yolo_model.predict(image, conf=conf, iou=iou, verbose=False)
    detections = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            w  = float(x2 - x1)
            h  = float(y2 - y1)
            detections.append({
                "x1":   float(x1),
                "y1":   float(y1),
                "x2":   float(x2),
                "y2":   float(y2),
                "cx":   cx,
                "cy":   cy,
                "w":    w,
                "h":    h,
                "conf": float(box.conf[0].cpu().numpy()),
            })

    return detections


# ---------------------------------------------------------------------------
# CNN cell classification
# ---------------------------------------------------------------------------

def classify_cell(
    image:     np.ndarray,
    detection: Dict,
    cnn_model: BrailleCellClassifier,
    device:    str,
    padding:   float = 0.10,
) -> Tuple[int, float]:
    """
    Classify the dot pattern in a single detected cell.
    Returns (pattern_int, confidence).
    """
    img_h, img_w = image.shape[:2]

    x1 = max(0, int(detection["x1"] - padding * detection["w"]))
    y1 = max(0, int(detection["y1"] - padding * detection["h"]))
    x2 = min(img_w, int(detection["x2"] + padding * detection["w"]))
    y2 = min(img_h, int(detection["y2"] + padding * detection["h"]))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return (0, 0.0)

    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Per-crop CLAHE — normalise each cell independently
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Apply CNN transform and run inference
    tensor = CNN_TRANSFORM(gray).unsqueeze(0).to(device)  # (1, 3, 64, 64)

    with torch.no_grad():
        logits = cnn_model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, cls = probs.max(dim=1)

    return (int(cls.item()), float(conf.item()))


# ---------------------------------------------------------------------------
# Annotation drawing
# ---------------------------------------------------------------------------

def draw_annotations(
    image:        np.ndarray,
    detections:   List[Dict],
    decoded_text: str,
) -> np.ndarray:
    """
    Draw bounding boxes coloured by CNN confidence, decoded chars above boxes,
    and a text banner at the bottom.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for d in detections:
        cnn_conf = d.get("cnn_conf", 0.0)
        pattern  = d.get("pattern_int", 0)
        char     = _G1_TABLE.get(pattern, "?")

        # Color by confidence
        if cnn_conf >= 0.70:
            color = (0, 200, 0)    # green — high confidence
        elif cnn_conf >= 0.40:
            color = (0, 200, 200)  # yellow — medium confidence
        else:
            color = (0, 0, 200)    # red — low/uncertain

        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Decoded character above the box
        label_y = max(y1 - 5, 12)
        cv2.putText(vis, char, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    # Black banner at bottom with decoded text
    banner_h = 40
    banner   = np.zeros((banner_h, vis.shape[1], 3), dtype=np.uint8)
    display_text = decoded_text[:80] + ("..." if len(decoded_text) > 80 else "")
    cv2.putText(banner, display_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    annotated = np.vstack([vis, banner])
    return annotated


# ---------------------------------------------------------------------------
# Full per-image pipeline
# ---------------------------------------------------------------------------

def process_image(
    image_or_path,
    yolo_model: Optional[YOLO],
    cnn_model:  Optional[BrailleCellClassifier],
    decoder:    BrailleDecoder,
    args:       argparse.Namespace,
) -> Tuple[str, np.ndarray]:
    """
    Run the complete BrailleVision pipeline on one image.
    Returns (decoded_text, annotated_image).
    """
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Step 1: Preprocess
    processed, meta = preprocess_full(
        image_or_path,
        use_dog=args.use_dog,
    )
    if processed is None:
        msg = "(Image too blurry — hold camera steady)" if not meta["sharp"] else "(Failed to load image)"
        return (msg, blank_img)

    if yolo_model is None:
        return ("(YOLO model not loaded)", processed)

    # Step 2: YOLO detection
    raw_detections = run_yolo_detection(yolo_model, processed, args.conf, args.iou)
    if not raw_detections:
        return ("(No Braille detected)", processed)

    # Step 3: CNN classification per cell
    device = args.device
    for det in raw_detections:
        if cnn_model is not None:
            pattern_int, cnn_conf = classify_cell(processed, det, cnn_model, device)
        else:
            pattern_int, cnn_conf = (0, 0.0)
        det["pattern_int"] = pattern_int
        det["cnn_conf"]    = cnn_conf

    # Step 4: Reconstruct reading order
    structured_rows = reconstruct_reading_order(raw_detections, processed.shape)

    # Step 5: Extract pattern sequences (replace dicts with int, keep None)
    pattern_rows = []
    flat_dets    = []
    for row in structured_rows:
        pat_row = []
        for cell in row:
            if cell is None:
                pat_row.append(None)
            else:
                pat_row.append(cell.get("pattern_int", 0))
                flat_dets.append(cell)
        pattern_rows.append(pat_row)

    # Step 6: Decode
    decoded_text = decoder.decode_page(pattern_rows)
    if not decoded_text.strip():
        decoded_text = "(No readable Braille found)"

    # Step 7: Annotate
    annotated = draw_annotations(processed, flat_dets, decoded_text)

    return (decoded_text, annotated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("BrailleVision — Initialising")
    print("=" * 60)

    yolo_model, cnn_model = load_models(args.yolo_weights, args.cnn_weights, args.device)
    decoder = BrailleDecoder(grade=args.grade)
    tts     = TTSEngine() if args.speak else None

    os.makedirs(args.output_dir, exist_ok=True)

    # Webcam mode
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {args.source}")
            return

        print("\n[WEBCAM] Press 'c' to capture, 'q' to quit.")
        cap_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("BrailleVision — Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                print("\n[CAPTURE] Processing...")
                text, annotated = process_image(frame, yolo_model, cnn_model, decoder, args)
                print(f"\n[DECODED]\n{text}\n")
                cv2.imshow("BrailleVision — Result", annotated)

                out_path = os.path.join(args.output_dir, f"capture_{cap_idx:04d}.jpg")
                cv2.imwrite(out_path, annotated)
                txt_path = os.path.join(args.output_dir, f"capture_{cap_idx:04d}.txt")
                with open(txt_path, "w") as f:
                    f.write(text)
                cap_idx += 1

                if tts and text.strip() and not text.startswith("("):
                    tts.speak(text)

            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Single image mode
        print(f"\n[PROCESSING] {args.source}")
        text, annotated = process_image(args.source, yolo_model, cnn_model, decoder, args)

        print(f"\n[DECODED TEXT]\n{'=' * 40}\n{text}\n{'=' * 40}")

        # Save outputs
        base = os.path.splitext(os.path.basename(args.source))[0]
        out_img = os.path.join(args.output_dir, f"{base}_annotated.jpg")
        out_txt = os.path.join(args.output_dir, f"{base}_decoded.txt")
        cv2.imwrite(out_img, annotated)
        with open(out_txt, "w") as f:
            f.write(text)
        print(f"[SAVED] {out_img}  |  {out_txt}")

        if args.visualize:
            cv2.imshow("BrailleVision — Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if tts and text.strip() and not text.startswith("("):
            tts.speak(text)


if __name__ == "__main__":
    main()
