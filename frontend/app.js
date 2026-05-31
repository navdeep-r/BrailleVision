/**
 * BrailleVision 2026 — app.js
 *
 * Frontend logic for camera capture, file upload, API calls, and TTS.
 * All communication is with the Flask backend at localhost:5000.
 */

"use strict";

const API_BASE = "http://localhost:5000";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _stream       = null;   // MediaStream from getUserMedia
let _decodedText  = "";     // Current decoded output

// ---------------------------------------------------------------------------
// DOM references (resolved lazily on first use)
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/**
 * Start the device camera (environment-facing / rear camera preferred).
 * Enables the Capture button once the stream is active.
 */
async function startCamera() {
  const video     = $("cameraFeed");
  const btnStart  = $("btnStartCamera");
  const btnCap    = $("btnCapture");
  const idleDiv   = $("cameraIdle");

  try {
    btnStart.disabled = true;
    btnStart.textContent = "Starting…";

    const constraints = {
      video: {
        facingMode: { ideal: "environment" },
        width:      { ideal: 1280 },
        height:     { ideal: 960 },
      },
      audio: false,
    };

    _stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = _stream;
    await video.play();

    // Hide idle overlay
    idleDiv.classList.add("hidden");
    idleDiv.setAttribute("aria-hidden", "true");

    btnCap.disabled = false;
    btnStart.textContent = "Camera On";
    btnStart.style.opacity = "0.6";

  } catch (err) {
    console.error("Camera error:", err);
    btnStart.disabled  = false;
    btnStart.textContent = "Start Camera";

    const msg = err.name === "NotAllowedError"
      ? "Camera access denied. Please allow camera permissions."
      : `Camera error: ${err.message}`;
    showAlert(msg);
  }
}


// ---------------------------------------------------------------------------
// Capture from camera
// ---------------------------------------------------------------------------

/**
 * Capture the current video frame and send it to the backend for decoding.
 */
function captureAndProcess() {
  const video  = $("cameraFeed");
  const canvas = $("captureCanvas");

  if (!video.srcObject) {
    showAlert("Camera is not started.");
    return;
  }

  // Draw current frame at full video resolution
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const base64 = canvas.toDataURL("image/jpeg", 0.92);
  processImage(base64);
}


// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------

/**
 * Handle image file selection via the file input.
 */
function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => processImage(e.target.result);
  reader.onerror = () => showAlert("Failed to read the selected file.");
  reader.readAsDataURL(file);

  // Reset input so the same file can be re-uploaded if needed
  event.target.value = "";
}


// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------

/**
 * Send a base64 image to the BrailleVision backend and populate results.
 * @param {string} base64Image - Data URI or raw base64 string.
 */
async function processImage(base64Image) {
  showLoading(true);
  hideResults();

  try {
    const response = await fetch(`${API_BASE}/api/process-image`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: base64Image }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    if (data.error) {
      // Soft error (blurry image, no detection, etc.) — show blur indicator if relevant
      if (data.error.toLowerCase().includes("blurry")) {
        showBlurIndicator(true);
        setTimeout(() => showBlurIndicator(false), 3000);
      } else {
        showAlert(data.error);
      }
      return;
    }

    // Populate results
    _decodedText = data.text || "(No Braille detected)";

    const decodedEl = $("decodedText");
    decodedEl.textContent = _decodedText;

    if (data.annotated_image_b64) {
      const img = $("annotatedImage");
      img.src = data.annotated_image_b64;
      img.alt = `Braille image with ${data.cell_count} detected cells annotated`;
    }

    const metaEl = $("resultMeta");
    const count = data.cell_count || 0;
    metaEl.textContent = `${count} Braille cell${count !== 1 ? "s" : ""} detected`;

    showResults(true);

  } catch (err) {
    console.error("Pipeline error:", err);
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      showAlert(
        "Cannot reach backend. Make sure BrailleVision API is running:\n" +
        "  python backend/app.py"
      );
    } else {
      showAlert(`Error: ${err.message}`);
    }
  } finally {
    showLoading(false);
  }
}


// ---------------------------------------------------------------------------
// Text-to-Speech
// ---------------------------------------------------------------------------

/**
 * Speak the decoded text using the Web Speech API.
 */
function speakText() {
  if (!_decodedText || _decodedText.startsWith("(")) return;

  // Cancel any ongoing speech first
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(_decodedText);
  utterance.rate  = 0.9;
  utterance.pitch = 1.0;
  utterance.lang  = "en-US";

  const btnSpeak = $("btnSpeak");
  utterance.onstart = () => {
    btnSpeak.style.background = "var(--accent-hover)";
    btnSpeak.setAttribute("aria-label", "Speaking — click to stop");
  };
  utterance.onend = utterance.onerror = () => {
    btnSpeak.style.background = "";
    btnSpeak.setAttribute("aria-label", "Speak decoded text");
  };

  window.speechSynthesis.speak(utterance);
}


// ---------------------------------------------------------------------------
// Copy to clipboard
// ---------------------------------------------------------------------------

async function copyText() {
  if (!_decodedText) return;

  try {
    await navigator.clipboard.writeText(_decodedText);
    const btn = $("btnCopy");
    const original = btn.innerHTML;
    btn.textContent = "✓ Copied";
    btn.style.color = "var(--green)";
    setTimeout(() => {
      btn.innerHTML = original;
      btn.style.color = "";
    }, 2000);
  } catch (err) {
    showAlert("Could not copy to clipboard. Select and copy text manually.");
  }
}


// ---------------------------------------------------------------------------
// Clear results
// ---------------------------------------------------------------------------

function clearResults() {
  _decodedText = "";
  $("decodedText").textContent = "";
  $("annotatedImage").src = "";
  $("resultMeta").textContent = "";
  hideResults();

  // Cancel any ongoing speech
  window.speechSynthesis.cancel();
}


// ---------------------------------------------------------------------------
// UI state helpers
// ---------------------------------------------------------------------------

function showLoading(show) {
  const el = $("loadingState");
  if (show) {
    el.removeAttribute("hidden");
  } else {
    el.setAttribute("hidden", "");
  }
}

function showResults(show) {
  const el = $("resultsSection");
  if (show) {
    el.removeAttribute("hidden");
    // Smooth scroll to results
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  } else {
    el.setAttribute("hidden", "");
  }
}

function hideResults() {
  $("resultsSection").setAttribute("hidden", "");
}

function showBlurIndicator(show) {
  const el = $("blurIndicator");
  if (show) {
    el.removeAttribute("hidden");
  } else {
    el.setAttribute("hidden", "");
  }
}

function showAlert(message) {
  // Use a non-blocking approach — add a toast or console fallback
  console.warn("[BrailleVision]", message);
  // Simple inline alert styled with CSS (better than window.alert for accessibility)
  const existing = document.querySelector(".bv-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "bv-toast";
  toast.setAttribute("role", "alert");
  toast.setAttribute("aria-live", "assertive");
  toast.style.cssText = `
    position: fixed;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    background: #1e1b2e;
    border: 1px solid var(--red, #ef4444);
    color: #f0f0f8;
    padding: 0.85rem 1.5rem;
    border-radius: 10px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.875rem;
    z-index: 1000;
    max-width: 480px;
    text-align: center;
    line-height: 1.5;
    white-space: pre-line;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    animation: toastIn 0.25s ease;
  `;
  toast.textContent = message;

  // Inject keyframe if not already present
  if (!document.querySelector("#bv-toast-style")) {
    const style = document.createElement("style");
    style.id = "bv-toast-style";
    style.textContent = `
      @keyframes toastIn {
        from { opacity: 0; transform: translateX(-50%) translateY(12px); }
        to   { opacity: 1; transform: translateX(-50%) translateY(0); }
      }
    `;
    document.head.appendChild(style);
  }

  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}


// ---------------------------------------------------------------------------
// Health check on load
// ---------------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", async () => {
  try {
    const resp = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (!resp.ok) throw new Error("API not healthy");
    const data = await resp.json();
    console.info("[BrailleVision] API connected.", data);
  } catch (_) {
    console.warn("[BrailleVision] Backend not reachable — run `python backend/app.py` first.");
    showAlert(
      "BrailleVision backend not running.\n" +
      "Start it with:  python backend/app.py"
    );
  }
});
