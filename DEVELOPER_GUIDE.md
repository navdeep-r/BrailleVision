# BrailleVision Developer Guide

This document serves as a detailed technical index for the BrailleVision project. It outlines the architecture, setup instructions, and critical components required to run, train, and maintain the system.

## 1. System Architecture

BrailleVision uses a two-stage computer vision pipeline, strictly separated into training pipelines, inference pipelines, and a web application interface.

* **Stage 1 (Object Detection):** A YOLOv8s model detects bounding boxes for every individual Braille cell in an image.
* **Stage 2 (Classification):** A MobileNetV3Small Convolutional Neural Network (CNN) classifies the 6-bit dot pattern inside each detected cell bounding box.
* **Decoder:** A custom Braille decoder translates the sequence of dot patterns into English text, supporting Grade 1 and Grade 2 Braille contractions.

## 2. Setup Instructions

To get started with developing and running BrailleVision locally:

### Prerequisites
* **Python:** 3.10 or higher
* **Node.js:** v18 or higher (for the frontend)
* **Git:** For cloning the repository

### Backend (Python & AI Models)
1. **Create and Activate Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Start the Flask API Server:**
   ```powershell
   python backend/app.py
   ```
   *The backend loads the required models on startup and listens on port 5000.*

### Frontend (Next.js)
1. **Navigate to Frontend Directory:**
   ```powershell
   cd frontend-next
   ```
2. **Install Dependencies:**
   ```powershell
   npm install
   ```
3. **Start the Development Server:**
   ```powershell
   npm run dev
   ```
   *The frontend UI will be available at http://localhost:3000.*

## 3. Training & Inference Code

The project components are organized by their roles in the ML lifecycle.

### Training Scripts
* **YOLO Training:** [`training/train_yolo.py`](./training/train_yolo.py)
  * Trains the YOLOv8s detector to identify Braille cell locations.
* **CNN Training:** [`training/train_cnn.py`](./training/train_cnn.py)
  * Trains the MobileNetV3Small classifier on cropped cell images.

### Inference Pipeline
* **Core Inference Engine:** [`inference/inference.py`](./inference/inference.py)
  * *Handles the full pipeline: Preprocessing -> YOLO -> CNN -> Postprocessing -> Decoded Text.*
* **Image Preprocessing:** [`inference/preprocess.py`](./inference/preprocess.py)
  * Includes functions for blurring checks, perspective correction, and contrast enhancement (CLAHE).
* **Braille Decoder (Grade 2):** [`inference/braille_decoder.py`](./inference/braille_decoder.py)
  * Stateful UEB Grade 1/2 decoder with support for capital and number indicators.

### Application Stack
* **Flask Backend API:** [`backend/app.py`](./backend/app.py)
  * Exposes the inference pipeline via REST API endpoints.
* **Next.js Frontend Client:** [`frontend-next/src/app/page.tsx`](./frontend-next/src/app/page.tsx)
  * React-based UI for uploading images and viewing translated Braille.

## 4. Model Weights

The trained weights are generated after running the training scripts or can be downloaded manually. They must be placed in the `model/` directory for inference to work.

* **YOLOv8s Cell Detector:** [`model/best.pt`](./model/best.pt)
* **CNN Dot Classifier:** [`model/cell_classifier_best.pth`](./model/cell_classifier_best.pth)

## 5. Dataset Information

The models are trained using two primary data sources: the DSBI (Double-Sided Braille Image) dataset and the Angelina Braille Reader charset. The dataset pipeline processes these sources into crops suitable for training.

* **Raw DSBI Source Data:** [`datasets/raw_sources/dsbi/`](./datasets/raw_sources/dsbi/)
* **Dataset Notes & Metadata:** [`datasets/metadata/dataset_notes.md`](./datasets/metadata/dataset_notes.md)
* **YOLO Formatted Crops:** [`datasets/cell_crops/`](./datasets/cell_crops/)

*Note: Datasets and generated crops are not included in the version control.*

## 6. Sample Outputs

Inference results, including images drawn with green bounding boxes and decoded text annotations, are saved automatically when testing the Python inference script locally.

* **Output Directory:** [`sample_outputs/`](./sample_outputs/)
  * Run `python inference/inference.py --source <image_path>` to generate new inference outputs here.
