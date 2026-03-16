"""
Smart Dustbin — Real-Time Webcam Waste Classifier
==================================================
Uses the trained EfficientNetV2B0 model to classify waste
from your webcam feed in real time.

Place waste items inside the scan zone (center box) and
the model will classify what it sees.

Controls:
    Q / ESC  — Quit
    S        — Save a screenshot of the current frame
    SPACE    — Pause / Resume

Requirements:
    pip install opencv-python tensorflow numpy
"""

import os
import sys
import time
import math
import numpy as np
import cv2

# ── Suppress TF info logs before import ──────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Fix WSL ↔ Windows file locking
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "smart_dustbin_model.keras")
WEIGHTS_PATH    = os.path.join(BASE_DIR, "smart_dustbin_model.weights.h5")
CLASSES_PATH    = os.path.join(BASE_DIR, "class_names.txt")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

IMG_SIZE           = (240, 240)
CONFIDENCE_THRESH  = 0.50
SMOOTHING_FRAMES   = 7
SCAN_ZONE_RATIO    = 0.45

# Default class names (alphabetical — 8 classes)
# Will be overridden by class_names.txt if it exists
CLASS_NAMES = [
    "E-waste",
    "battery waste",
    "glass waste",
    "light bulbs",
    "metal waste",
    "organic waste",
    "paper waste",
    "plastic waste",
]

# Load class names from file if available (ensures correct order)
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r") as f:
        loaded = [line.strip() for line in f if line.strip()]
    if loaded:
        CLASS_NAMES = loaded

# Bin info: (bin_name, color_BGR, recyclable?, disposal_note)
WASTE_INFO = {
    "E-waste": (
        "E-Waste Bin", (60, 60, 220), False,
        "Take to e-waste collection center. Remove batteries first."
    ),
    "battery waste": (
        "Hazardous Bin", (0, 130, 255), False,
        "HAZARDOUS — Never throw in regular trash. Drop at collection point."
    ),
    "glass waste": (
        "Recyclable Bin", (0, 200, 100), True,
        "RECYCLABLE — Rinse clean. Remove caps/lids before recycling."
    ),
    "light bulbs": (
        "Hazardous Bin", (0, 130, 255), False,
        "HAZARDOUS — Do not break. Take to special waste facility."
    ),
    "metal waste": (
        "Recyclable Bin", (0, 200, 100), True,
        "RECYCLABLE — Crush cans to save space. Rinse food containers."
    ),
    "organic waste": (
        "Organic / Wet Bin", (50, 130, 180), False,
        "COMPOSTABLE — Can be composted. Keep separate from dry waste."
    ),
    "paper waste": (
        "Paper Bin", (220, 160, 50), True,
        "RECYCLABLE — Keep dry. Remove any plastic coating or tape."
    ),
    "plastic waste": (
        "Plastic Bin", (0, 220, 255), True,
        "RECYCLABLE — Check resin code. Rinse containers before recycling."
    ),
}

# Colors (BGR)
COLOR_BG         = (30, 30, 30)
COLOR_TEXT        = (255, 255, 255)
COLOR_ACCENT     = (0, 220, 120)
COLOR_DIM        = (120, 120, 120)
COLOR_ZONE_IDLE  = (160, 160, 160)
COLOR_RECYCLE    = (0, 220, 80)       # Bright green for recyclable badge

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────────────────────────────────────
def build_model_architecture(num_classes):
    from tensorflow.keras.applications import EfficientNetV2B0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
    from tensorflow.keras.models import Model

    base_model = EfficientNetV2B0(
        weights=None, include_top=False, input_shape=(*IMG_SIZE, 3),
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs=base_model.input, outputs=predictions)


# ──────────────────────────────────────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────────────────────────────────────
def load_model():
    # Priority 1: Use exported .weights.h5 (most reliable)
    if os.path.exists(WEIGHTS_PATH):
        print(f"📦  Loading weights from: {WEIGHTS_PATH}")
        model = build_model_architecture(len(CLASS_NAMES))
        model.load_weights(WEIGHTS_PATH)
        print("✅  Model loaded from exported weights")
    elif os.path.exists(MODEL_PATH):
        print(f"📦  Loading model from: {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅  Model loaded (standard)")
        except Exception:
            print("⚠️  Standard load failed. Rebuilding architecture...")
            model = build_model_architecture(len(CLASS_NAMES))
            model.load_weights(MODEL_PATH)
            print("✅  Model loaded (architecture rebuild)")
    else:
        print(f"❌  No model found!")
        print(f"   Expected: {WEIGHTS_PATH}")
        print(f"        or : {MODEL_PATH}")
        print(f"\n   Run 'python export_weights.py' in WSL first.")
        sys.exit(1)

    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("🔥  Ready!\n")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Scan Zone
# ──────────────────────────────────────────────────────────────────────────────
def get_scan_zone(frame_h, frame_w):
    size = int(min(frame_h, frame_w) * SCAN_ZONE_RATIO)
    cx, cy = frame_w // 2, frame_h // 2
    x1, y1 = cx - size // 2, cy - size // 2
    return x1, y1, x1 + size, y1 + size, size


def preprocess_zone(frame, x1, y1, x2, y2):
    region = frame[y1:y2, x1:x2]
    resized = cv2.resize(region, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return preprocess_input(np.expand_dims(rgb.astype(np.float32), axis=0))


# ──────────────────────────────────────────────────────────────────────────────
# Animated Scan Zone
# ──────────────────────────────────────────────────────────────────────────────
def draw_scan_zone_animated(frame, x1, y1, x2, y2, color, t):
    """
    Draw a scan zone with:
      - Pulsing corner brackets (breathing effect)
      - A horizontal scan line sweeping up and down
    t = current time in seconds (used for animation phase)
    """
    w = x2 - x1
    h = y2 - y1

    # ── Breathing corners ────────────────────────────────────────────────
    # Corner length oscillates between 20 and 40
    pulse = math.sin(t * 3.0) * 0.5 + 0.5          # 0..1 pulsing
    corner_len = int(20 + pulse * 20)
    corner_len = min(corner_len, w // 4, h // 4)

    # Corner opacity also pulses slightly
    alpha = 0.6 + pulse * 0.4  # 0.6..1.0

    # Blend a brighter version of the color for the pulse
    bright = tuple(min(255, int(c * (0.7 + pulse * 0.5))) for c in color)

    # Thin full rectangle (always subtle)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    ct = 3  # corner thickness
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), bright, ct)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), bright, ct)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), bright, ct)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), bright, ct)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), bright, ct)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), bright, ct)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), bright, ct)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), bright, ct)

    # ── Scanning line (sweeps up and down) ───────────────────────────────
    scan_phase = (math.sin(t * 2.0) * 0.5 + 0.5)   # 0..1 bounce
    scan_y = int(y1 + scan_phase * h)
    scan_y = max(y1 + 2, min(scan_y, y2 - 2))

    # Draw a semi-transparent horizontal line with gradient fade
    overlay = frame.copy()
    cv2.line(overlay, (x1 + 5, scan_y), (x2 - 5, scan_y), color, 2)

    # Small glow around the scan line
    for offset in range(1, 8):
        fade = max(0, 255 - offset * 40)
        fade_color = tuple(int(c * fade / 255) for c in color)
        if y1 < scan_y - offset < y2:
            cv2.line(overlay, (x1 + 5, scan_y - offset),
                     (x2 - 5, scan_y - offset), fade_color, 1)
        if y1 < scan_y + offset < y2:
            cv2.line(overlay, (x1 + 5, scan_y + offset),
                     (x2 - 5, scan_y + offset), fade_color, 1)

    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)


# ──────────────────────────────────────────────────────────────────────────────
# Result Panel
# ──────────────────────────────────────────────────────────────────────────────
def draw_result_panel(frame, prediction, confidence, bin_name, bin_color,
                      recyclable, note, x1, y1, x2, y2):
    """Draw the classification result with recycling note below the scan zone."""
    fh, fw = frame.shape[:2]

    card_w = x2 - x1
    card_h = 120  # Taller to fit the note
    card_x = x1
    card_y = y2 + 12

    if card_y + card_h > fh:
        card_y = y1 - card_h - 12

    # Semi-transparent card background
    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y),
                  (card_x + card_w, card_y + card_h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Colored left stripe
    cv2.rectangle(frame, (card_x, card_y),
                  (card_x + 5, card_y + card_h), bin_color, -1)

    # Category name + recyclable badge
    label_x = card_x + 15
    cv2.putText(frame, prediction.upper(),
                (label_x, card_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, bin_color, 2)

    if recyclable:
        # Recyclable badge next to the name
        label_text = prediction.upper()
        (tw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.65, 2)
        badge_x = label_x + tw + 10
        badge_text = "RECYCLABLE"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.35, 1)
        # Badge background
        cv2.rectangle(frame, (badge_x - 3, card_y + 10),
                      (badge_x + bw + 5, card_y + 10 + bh + 6),
                      COLOR_RECYCLE, -1)
        cv2.putText(frame, badge_text,
                    (badge_x, card_y + 10 + bh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Confidence bar
    bar_x = label_x
    bar_y_pos = card_y + 38
    bar_w = card_w - 80
    bar_h = 12
    cv2.rectangle(frame, (bar_x, bar_y_pos),
                  (bar_x + bar_w, bar_y_pos + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y_pos),
                  (bar_x + filled, bar_y_pos + bar_h), bin_color, -1)
    cv2.putText(frame, f"{confidence * 100:.0f}%",
                (bar_x + bar_w + 8, bar_y_pos + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)

    # Bin recommendation
    cv2.putText(frame, f"Bin: {bin_name}",
                (label_x, card_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    # Disposal / recycling note
    # Wrap long notes to fit the card
    max_chars = card_w // 8  # Approximate chars that fit
    if len(note) > max_chars:
        note_line1 = note[:max_chars]
        # Try to break at a space
        last_space = note_line1.rfind(' ')
        if last_space > 0:
            note_line1 = note[:last_space]
            note_line2 = note[last_space + 1:]
        else:
            note_line2 = note[max_chars:]

        cv2.putText(frame, note_line1,
                    (label_x, card_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_DIM, 1)
        cv2.putText(frame, note_line2[:max_chars],
                    (label_x, card_y + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_DIM, 1)
    else:
        cv2.putText(frame, note,
                    (label_x, card_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_DIM, 1)


# ──────────────────────────────────────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, paused):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Smart Dustbin Classifier",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ACCENT, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    # Bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), COLOR_BG, -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "Q: Quit | S: Screenshot | SPACE: Pause",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DIM, 1)

    # Paused overlay
    if paused:
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (w // 2 - 80, h // 2 - 25),
                      (w // 2 + 80, h // 2 + 25), COLOR_BG, -1)
        cv2.addWeighted(overlay3, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "PAUSED",
                    (w // 2 - 55, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("🗑️   Smart Dustbin — Real-Time Webcam Classifier")
    print("=" * 55)

    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌  Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷  Webcam: {actual_w}x{actual_h}")
    print("   Place waste items inside the center scan zone.")
    print("   Press Q or ESC to quit.\n")

    pred_buffer = []
    paused = False
    prev_time = time.time()
    start_time = time.time()

    zx1, zy1, zx2, zy2, zone_size = get_scan_zone(actual_h, actual_w)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌  Failed to read from webcam")
                    break

                display = frame.copy()
                t = time.time() - start_time  # Animation timer

                # ── Classify ─────────────────────────────────────────────
                processed = preprocess_zone(frame, zx1, zy1, zx2, zy2)
                preds = model.predict(processed, verbose=0)[0]

                pred_buffer.append(preds)
                if len(pred_buffer) > SMOOTHING_FRAMES:
                    pred_buffer.pop(0)

                avg_preds = np.mean(pred_buffer, axis=0)
                pred_idx = np.argmax(avg_preds)
                confidence = float(avg_preds[pred_idx])
                prediction = CLASS_NAMES[pred_idx]

                # ── Draw ─────────────────────────────────────────────────
                if confidence >= CONFIDENCE_THRESH:
                    info = WASTE_INFO.get(prediction)
                    if info:
                        bin_name, bin_color, recyclable, note = info
                    else:
                        bin_name, bin_color, recyclable, note = (
                            "Unknown", COLOR_DIM, False, "")

                    # Animated scan zone in category color
                    draw_scan_zone_animated(display, zx1, zy1, zx2, zy2,
                                            bin_color, t)

                    # Result card with recycling note
                    draw_result_panel(display, prediction, confidence,
                                      bin_name, bin_color, recyclable, note,
                                      zx1, zy1, zx2, zy2)
                else:
                    # Idle — gray animated scan zone
                    draw_scan_zone_animated(display, zx1, zy1, zx2, zy2,
                                            COLOR_ZONE_IDLE, t)

                    # "Place item here" hint
                    hint = "Place waste item here"
                    (tw, _), _ = cv2.getTextSize(hint,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.6, 1)
                    hx = zx1 + (zone_size - tw) // 2
                    cv2.putText(display, hint, (hx, zy2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                COLOR_ZONE_IDLE, 1)

                # FPS
                current_time = time.time()
                fps = 1.0 / max(current_time - prev_time, 1e-6)
                prev_time = current_time

                draw_hud(display, fps, paused)

            else:
                display = frame.copy()
                t = time.time() - start_time
                draw_scan_zone_animated(display, zx1, zy1, zx2, zy2,
                                        COLOR_ZONE_IDLE, t)
                draw_hud(display, fps, paused=True)

            cv2.imshow("Smart Dustbin Classifier", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('s'), ord('S')):
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SCREENSHOTS_DIR, f"capture_{ts}.png")
                cv2.imwrite(path, frame)
                print(f"📸  Screenshot saved: {path}")
            elif key == 32:
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋  Webcam released. Goodbye!")


if __name__ == "__main__":
    main()
