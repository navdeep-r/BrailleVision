"""
backend/app.py

Flask REST API for BrailleVision.

Models are loaded ONCE at module startup — never per request.
Per-request loading takes 3-10 seconds and would break the user experience.

Environment variables:
  YOLO_PATH  (default: model/best.pt)
  CNN_PATH   (default: model/cell_classifier_best.pth)
  DEVICE     (default: cpu)

Routes:
  GET  /api/health           → status + loaded model paths
  POST /api/process-image    → {"image": "<base64>"} → pipeline result
"""

import os
import sys
import base64
import traceback
from typing import Optional

# Globally disable GPU because the RTX 5060 sm_120 is unsupported by the current PyTorch binaries.
# This forces both YOLO and the CNN to run on CPU without crashing.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference import load_models, process_image
from inference.braille_decoder import BrailleDecoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YOLO_PATH = os.getenv("YOLO_PATH", "model/best.pt")
CNN_PATH  = os.getenv("CNN_PATH",  "model/cell_classifier_best.pth")
DEVICE    = os.getenv("DEVICE",    "cpu")
GRADE     = int(os.getenv("BRAILLE_GRADE", "2"))

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------

# Resolve project root for serving frontend files
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend")

app = Flask(__name__, static_folder=_FRONTEND_DIR, static_url_path="/static")
CORS(app)  # Allow cross-origin requests

# ---------------------------------------------------------------------------
# Load models ONCE at startup
# ---------------------------------------------------------------------------

print("[STARTUP] Loading models...")
_yolo_model, _cnn_model = load_models(YOLO_PATH, CNN_PATH, DEVICE)
_decoder = BrailleDecoder(grade=GRADE)
print("[STARTUP] Models ready.")


# ---------------------------------------------------------------------------
# Image encode/decode helpers
# ---------------------------------------------------------------------------

def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded image string to a BGR numpy array.
    Strips the data URI prefix if present.
    """
    try:
        # Strip data URI prefix: "data:image/jpeg;base64,..."
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        raw_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] decode_base64_image: {e}")
        return None


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a BGR numpy array as a JPEG base64 data URI."""
    success, buffer = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    if not success:
        return ""
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return "data:image/jpeg;base64," + b64


# ---------------------------------------------------------------------------
# Core pipeline wrapper
# ---------------------------------------------------------------------------

class _InferenceArgs:
    """Minimal args namespace compatible with process_image()."""
    conf    = float(os.getenv("CONF",  "0.35"))
    iou     = float(os.getenv("IOU",   "0.35"))
    device  = DEVICE
    grade   = GRADE
    use_dog = False
    visualize = False
    speak   = False
    output_dir = "sample_outputs"


_INFERENCE_ARGS = _InferenceArgs()


def run_full_pipeline(image: np.ndarray) -> dict:
    """
    Run the full BrailleVision pipeline on a numpy image.
    Returns a result dict with keys:
      text, annotated_image_b64, cell_count, error
    """
    try:
        text, annotated = process_image(
            image,
            _yolo_model,
            _cnn_model,
            _decoder,
            _INFERENCE_ARGS,
        )

        # Handle blur rejection
        if text and "blurry" in text.lower():
            return {
                "text": "",
                "annotated_image_b64": "",
                "cell_count": 0,
                "error": "Image too blurry. Hold camera steady.",
            }

        # Encode annotated image
        b64 = encode_image_to_base64(annotated) if annotated is not None else ""

        # Count detected cells (rough: count non-None items in annotated)
        cell_count = 0
        if text and not text.startswith("("):
            # Approximate from text length
            cell_count = len([c for c in text if c != " " and c != "\n"])

        return {
            "text": text if text else "(No Braille detected)",
            "annotated_image_b64": b64,
            "cell_count": cell_count,
            "error": "",
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "text": "",
            "annotated_image_b64": "",
            "cell_count": 0,
            "error": f"Pipeline error: {str(e)}",
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def serve_frontend():
    """Serve the frontend index.html at the root URL."""
    from flask import send_from_directory
    return send_from_directory(_FRONTEND_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_frontend_files(filename):
    """Serve other frontend assets (CSS, JS)."""
    filepath = os.path.join(_FRONTEND_DIR, filename)
    if os.path.isfile(filepath):
        from flask import send_from_directory
        return send_from_directory(_FRONTEND_DIR, filename)
    # Fall through to other routes / 404
    from flask import abort
    abort(404)

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "yolo":   YOLO_PATH,
        "cnn":    CNN_PATH,
        "device": DEVICE,
        "grade":  GRADE,
        "yolo_loaded": _yolo_model is not None,
        "cnn_loaded":  _cnn_model  is not None,
    })


@app.route("/api/process-image", methods=["POST"])
def process_image_endpoint():
    """
    Process a Braille image.

    Request body (JSON):
      {"image": "<base64 string with optional data URI prefix>"}

    Response (JSON):
      {
        "text": "decoded braille text",
        "annotated_image_b64": "data:image/jpeg;base64,...",
        "cell_count": 42,
        "error": ""
      }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    b64  = data.get("image", "")
    if not b64:
        return jsonify({"error": "Missing 'image' field in request body"}), 400

    img = decode_base64_image(b64)
    if img is None:
        return jsonify({"error": "Could not decode image. Check base64 encoding."}), 400

    result = run_full_pipeline(img)

    # Return 200 even for soft errors (blurry, no detection) — let frontend handle
    return jsonify(result), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("sample_outputs", exist_ok=True)
    print("\n[BrailleVision API] Running on http://0.0.0.0:5000")
    print("  Open http://localhost:5000 in your browser to use the UI.")
    app.run(host="0.0.0.0", port=5000, debug=False)
