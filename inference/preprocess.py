"""
inference/preprocess.py

Full preprocessing pipeline for BrailleVision.
All functions are pure stateless operations on numpy arrays.

Pipeline order (MUST NOT be rearranged):
  1. load_image
  2. check_blur
  3. correct_perspective  ← on BGR (needs colour for reliable edge detection)
  4. convert to grayscale
  5. apply_gaussian_denoise  ← BEFORE CLAHE
  6. apply_clahe
  7. apply_dog_filter  (optional)
  8. convert back to 3-channel for YOLO
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Load image
# ---------------------------------------------------------------------------

def load_image(source) -> Optional[np.ndarray]:
    """
    Accept a file path (str) or a numpy array (BGR).
    Returns BGR numpy array, or None on failure.
    """
    if isinstance(source, np.ndarray):
        return source.copy()
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            print(f"  [WARN] preprocess: Could not load '{source}'")
        return img
    return None


# ---------------------------------------------------------------------------
# 2. Blur detection
# ---------------------------------------------------------------------------

def check_blur(image: np.ndarray, threshold: float = 80.0) -> Tuple[bool, float]:
    """
    Compute Laplacian variance as a sharpness metric.
    Returns (is_sharp: bool, variance: float).

    WHY: Blurry images have smoothed edges → low Laplacian variance.
    Threshold 80 is tuned for phone cameras at arm's length.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return (variance >= threshold, variance)


# ---------------------------------------------------------------------------
# 3. Perspective correction
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points as [top-left, top-right, bottom-right, bottom-left].
    Input: (4, 2) array of unordered corners.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # top-left:     min(x+y)
    ordered[2] = pts[np.argmax(s)]   # bottom-right: max(x+y)

    d = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(d)]   # top-right:    min(x-y)
    ordered[3] = pts[np.argmax(d)]   # bottom-left:  max(x-y)

    return ordered


def detect_page_boundary(
    image: np.ndarray,
    min_area_fraction: float = 0.20,
) -> Optional[np.ndarray]:
    """
    Detect the four corners of a Braille page in the image.
    Returns (4, 2) float32 corners, or None if not found.
    """
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to close small gaps in page boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < min_area_fraction * img_area:
            break  # sorted descending, no point continuing
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def correct_perspective(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Detect the page boundary and apply a perspective warp to straighten it.
    Returns (warped_image, was_corrected).
    Falls back to (original_image, False) if no page quad found.

    WHY: Off-angle cameras make cells appear as trapezoids and rows converge.
    Correcting this before YOLO gives a large accuracy gain.
    """
    corners = detect_page_boundary(image)
    if corners is None:
        return (image, False)

    ordered = order_points(corners)
    tl, tr, br, bl = ordered

    # Compute natural output dimensions
    width_top    = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    out_w = int(max(width_top, width_bottom))

    height_left  = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    out_h = int(max(height_left, height_right))

    if out_w < 50 or out_h < 50:
        return (image, False)

    dst = np.array([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return (warped, True)


# ---------------------------------------------------------------------------
# 4. CLAHE
# ---------------------------------------------------------------------------

def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    WHY: Standard histogram equalization is global. CLAHE works on local tiles,
    boosting dark corners without blowing out bright centre regions.
    Single most impactful preprocessing step for Braille.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# 5. Gaussian denoise
# ---------------------------------------------------------------------------

def apply_gaussian_denoise(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Mild Gaussian blur to suppress sensor noise.

    CRITICAL ORDER: call BEFORE CLAHE — denoising after CLAHE undoes enhancement.
    """
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


# ---------------------------------------------------------------------------
# 6. Difference-of-Gaussians filter
# ---------------------------------------------------------------------------

def apply_dog_filter(
    gray: np.ndarray,
    sigma1: float = 1.0,
    sigma2: float = 3.0,
) -> np.ndarray:
    """
    Difference-of-Gaussians: enhances blob-like structures (Braille dots).
    Recommended for frontal-flash / flat illumination.

    WHY: DoG enhances blob-like structures regardless of lighting direction.
    """
    g1 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma2)
    dog = g1 - g2

    # Normalise to 0-255
    dog = dog - dog.min()
    denom = dog.max()
    if denom > 0:
        dog = dog / denom * 255.0
    return dog.astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_full(
    image_input,
    blur_threshold: float = 80.0,
    use_dog: bool = False,
    clahe_clip: float = 2.0,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Complete preprocessing pipeline. Execute steps in EXACTLY this order.

    Returns (processed_3ch, meta_dict).
    Returns (None, meta) if image is None or too blurry.

    meta keys: blur_score, sharp, perspective_corrected, dog_applied
    """
    meta = {
        "blur_score": 0.0,
        "sharp": False,
        "perspective_corrected": False,
        "dog_applied": False,
    }

    # Step 1: Load
    image = load_image(image_input)
    if image is None:
        return (None, meta)

    # Step 2: Blur check
    is_sharp, blur_score = check_blur(image, threshold=blur_threshold)
    meta["blur_score"] = blur_score
    meta["sharp"] = is_sharp
    if not is_sharp:
        return (None, meta)

    # Step 3: Perspective correction (BEFORE grayscale — needs colour)
    image, corrected = correct_perspective(image)
    meta["perspective_corrected"] = corrected

    # Step 4: Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 5: Denoise BEFORE CLAHE
    gray = apply_gaussian_denoise(gray)

    # Step 6: CLAHE
    gray = apply_clahe(gray, clip_limit=clahe_clip)

    # Step 7: DoG (optional)
    if use_dog:
        gray = apply_dog_filter(gray)
        meta["dog_applied"] = True

    # Step 8: Convert back to 3-channel (YOLO needs 3-channel input)
    processed_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return (processed_3ch, meta)
