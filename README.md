# BrailleVision 2026

> Real physical Braille → English text → Speech, using a two-stage deep learning pipeline.

## What Is BrailleVision?

BrailleVision is an accessibility tool that reads physical handwritten or embossed Braille from a camera or image upload and converts it to English text and speech. It uses a two-stage ML pipeline: a YOLOv8s detector finds WHERE each Braille cell is, and a MobileNetV3Small CNN identifies WHAT 6-bit dot pattern each cell contains. A stateful UEB decoder then converts the dot patterns to English with Grade 2 contraction support.

This is a real accessibility tool — every accuracy decision matters.

---

## System Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU supported; CUDA strongly recommended for training)
- CUDA 11.8+ (optional, for GPU training)
- OS: Windows 10/11, macOS 12+, Ubuntu 20.04+
- RAM: 8 GB minimum (16 GB recommended)
- Disk: ~4 GB for models + datasets

---

## Quick Setup (5 commands)

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd braille

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create directory structure
python scaffold.py

# 5. Run inference (demo with sample image)
python inference/inference.py --source sample_inputs/test_braille.jpg --visualize --speak
```

---

## Dataset Setup

### DSBI Dataset
```
datasets/raw_sources/dsbi/
├── train.txt
├── test.txt
├── image1.jpg
├── image1+recto.txt
└── ...
```

### Angelina Dataset
```
datasets/raw_sources/angelina/
├── books/
├── handwritten/
├── pics/
├── uploaded/
└── not_braille/
```

Each Angelina subdirectory contains `image.jpg` + `image.json` pairs.

---

## Data Preparation

```bash
python data_preparation/convert_dsbi.py
python data_preparation/convert_angelina.py
python data_preparation/generate_splits.py
```

Verify: check `datasets/metadata/split_manifest.csv` and `datasets/metadata/class_distribution.csv` before training.

---

## Training

### Stage 1: YOLO Cell Detector (~2–4 hours on RTX 3060)
```bash
python training/train_yolo.py
```
Output: `model/best.pt`

Check: `runs/detect/braille_cell_detector/results.png`

### Stage 2: CNN Cell Classifier (~30–60 minutes on GPU)
```bash
python training/train_cnn.py
```
Output: `model/cell_classifier_best.pth`

---

## Inference

### Single image
```bash
python inference/inference.py \
  --source sample_inputs/test_braille.jpg \
  --visualize \
  --speak
```

### Webcam (press 'c' to capture, 'q' to quit)
```bash
python inference/inference.py --source 0 --speak
```

### All options
```
--source         Image path or camera index (0, 1, …)
--yolo_weights   Path to YOLO weights (default: model/best.pt)
--cnn_weights    Path to CNN weights  (default: model/cell_classifier_best.pth)
--conf           Detection confidence threshold (default: 0.35)
--iou            NMS IoU threshold (default: 0.35)
--grade          Braille grade: 1 or 2 (default: 2)
--visualize      Display annotated image
--speak          Read decoded text aloud
--use_dog        Apply Difference-of-Gaussians filter (flat/flash illumination)
--device         pytorch device: cpu or cuda (default: cpu)
--output_dir     Where to save outputs (default: sample_outputs/)
```

---

## Web App

```bash
# Start backend (in one terminal)
python backend/app.py

# Open frontend in browser
# Open frontend/index.html in Chrome or Firefox
```

API endpoints:
- `GET  http://localhost:5000/api/health` — status check
- `POST http://localhost:5000/api/process-image` — `{"image": "<base64>"}` → result

---

## Evaluation

```bash
python inference/evaluate.py \
  --gt_csv sample_inputs/ground_truth.csv \
  --output sample_outputs/eval_results.csv
```

Ground truth CSV format: `image_path,expected_text`

---

## Evaluation Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| YOLO mAP@0.5 | TBD after training | Target > 0.85 |
| YOLO mAP@0.5:0.95 | TBD | — |
| YOLO Precision | TBD | — |
| YOLO Recall | TBD | Target > Precision |
| CNN Top-1 Accuracy (clean) | TBD | Target > 95% |
| CNN Top-1 Accuracy (augmented) | TBD | Target > 88% |
| Mean CER (real images) | TBD | — |
| Mean WER (real images) | TBD | — |

_Lighting conditions for real-world test images: TBD_

---

## Architecture Overview

The pipeline has 8 steps:
1. **Preprocessing** — blur check, perspective correction, CLAHE, optional DoG filter
2. **YOLO Detection** — YOLOv8s finds bounding boxes for every Braille cell (1 class)
3. **CNN Classification** — MobileNetV3Small classifies the 6-bit dot pattern per cell
4. **Post-Processing** — size/edge filters, merged-cell splitting, row clustering with running-mean Y
5. **Space Detection** — gap-based word space insertion within rows
6. **Braille Decoding** — stateful UEB Grade 1/2 decoder with capital/number indicators
7. **Annotation** — colour-coded confidence overlay on detected cells
8. **TTS** — pyttsx3 (offline) or gTTS (online) speech output

```
Image → [Preprocess] → [YOLO] → [CNN per cell] → [Postprocess]
      → [Decoder] → Text → [TTS] → Audio
```

---

## Edge Cases Handled

- **Blur rejection**: Laplacian variance check; returns "Hold steady" if too blurry
- **Perspective correction**: Detects page quad, applies homographic warp before grayscale
- **Merged cells**: Boxes 1.5× wider than median are split into equal slices
- **Size outliers**: Detections >30% from median cell size are discarded
- **Edge cells**: Detections within 5px of any image edge are discarded
- **Number indicator**: State machine switches to numeric lookup for digits
- **Capital indicator**: Single cap → one uppercase letter; double cap → WORD CAPS
- **Space resets**: Word spaces reset number mode and capital-word mode correctly
- **Grade 2 contractions**: Checked only when pattern is a standalone word
- **DoG filter**: `--use_dog` for frontal-flash / flat illumination enhancement
- **Hard negatives**: `not_braille` images with empty labels teach YOLO zero cells
- **Per-crop CLAHE**: Each cell normalised independently regardless of page position

---

## Limitations

- **Book spine curvature**: Pages photographed from a bound book curve; no radial distortion correction is applied
- **Non-paper Braille**: Metal, plastic, and slate-and-stylus Braille surfaces have very different contrast characteristics — the model may generalise poorly
- **Grade 2 contractions**: Only the 23 most common whole-word contractions are implemented; letter-based contractions (e.g., "st" = dots 3,4) are not
- **Low-resolution cameras**: Cells must be at least ~15px wide in the final image for reliable detection
- **Extreme tilt**: Perspective correction handles up to ~30° tilt reliably; beyond that accuracy degrades

---

## AI Tools Disclosure

This project was built with the assistance of an AI coding assistant (Antigravity / Google DeepMind) for code scaffolding, architecture design, and implementation. All algorithmic decisions, accuracy constraints, and edge case specifications were defined by the project team.

---

## Project Structure

```
braillevision/
├── requirements.txt       # Python dependencies
├── scaffold.py            # Creates directory tree
├── data_preparation/      # Dataset conversion scripts
├── training/              # YOLO + CNN training
├── inference/             # Preprocessing, postprocessing, decoder, TTS, inference
├── backend/               # Flask REST API
├── frontend/              # HTML/CSS/JS web UI
├── model/                 # Trained weights (not committed)
├── datasets/              # Raw + processed data (not committed)
├── sample_inputs/         # Test images
└── sample_outputs/        # Inference results
```

---

*BrailleVision 2026 — Build honestly. Document clearly. Focus on accessibility.*
