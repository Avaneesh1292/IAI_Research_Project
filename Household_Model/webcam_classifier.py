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
# UI Helpers
# ──────────────────────────────────────────────────────────────────────────────
def draw_text_with_shadow(img, text, position, font, scale, color, thickness=1, shadow_color=(0,0,0), shadow_offset=(2,2)):
    """Draws text with a subtle drop shadow for better readability."""
    x, y = position
    sx, sy = shadow_offset
    cv2.putText(img, text, (x + sx, y + sy), font, scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, position, font, scale, color, thickness, cv2.LINE_AA)

def draw_rounded_rect(img, top_left, bottom_right, color, thickness=1, radius=10, fill=False):
    """Draws a rectangle with rounded corners."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw corners
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

    if fill:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Result Panel
# ──────────────────────────────────────────────────────────────────────────────
def draw_result_panel(frame, prediction, confidence, bin_name, bin_color,
                      recyclable, note, x1, y1, x2, y2):
    """Draw the classification result with recycling note below the scan zone."""
    fh, fw = frame.shape[:2]

    card_w = x2 - x1
    card_h = 130  # Taller to fit the note with better padding
    card_x = x1
    card_y = y2 + 15

    if card_y + card_h > fh - 40: # Avoid HUD overlap
        card_y = y1 - card_h - 15

    # ── Modern Glassmorphic-style Panel ──
    overlay = frame.copy()
    draw_rounded_rect(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 25), radius=12, fill=True)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Border
    draw_rounded_rect(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 60, 65), thickness=1, radius=12)

    # Thick colored left accent stripe (simulated by a filled rounded rect shifted left and clipped)
    stripe_w = 8
    cv2.rectangle(frame, (card_x, card_y + 12), (card_x + stripe_w, card_y + card_h - 12), bin_color, -1)

    # Category name
    label_x = card_x + 20
    draw_text_with_shadow(frame, prediction.upper(), (label_x, card_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, bin_color, 2)

    if recyclable:
        # Pilled Recyclable badge
        label_text = prediction.upper()
        (tw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        badge_x = label_x + tw + 15
        badge_text = "RECYCLABLE"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        draw_rounded_rect(frame, (badge_x - 6, card_y + 14), (badge_x + bw + 6, card_y + 14 + bh + 8), COLOR_RECYCLE, radius=8, fill=True)
        cv2.putText(frame, badge_text, (badge_x, card_y + 14 + bh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 30, 0), 1, cv2.LINE_AA)

    # Custom segmented confidence bar
    bar_x = label_x
    bar_y_pos = card_y + 45
    bar_w = card_w - 90
    bar_h = 8
    
    # Track
    draw_rounded_rect(frame, (bar_x, bar_y_pos), (bar_x + bar_w, bar_y_pos + bar_h), (50, 50, 50), radius=4, fill=True)
    # Fill
    filled = int(bar_w * confidence)
    if filled > 8: # Avoid drawing weird artifacts if confidence is super low
        draw_rounded_rect(frame, (bar_x, bar_y_pos), (bar_x + filled, bar_y_pos + bar_h), bin_color, radius=4, fill=True)
        
    draw_text_with_shadow(frame, f"{confidence * 100:.1f}%", (bar_x + bar_w + 12, bar_y_pos + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240,240,240), 1)

    # Bin recommendation (Bold)
    draw_text_with_shadow(frame, f"Drop in:", (label_x, card_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM, 1)
    (drop_w, _), _ = cv2.getTextSize("Drop in: ", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    draw_text_with_shadow(frame, bin_name, (label_x + drop_w, card_y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_TEXT, 1)

    # Disposal / recycling note
    max_chars = card_w // 7  # Approximate chars that fit
    if len(note) > max_chars:
        # Smart text wrapping
        words = note.split(' ')
        lines = []
        current_line = []
        for word in words:
            if len(' '.join(current_line + [word])) <= max_chars:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
            
        y_offset = card_y + 105
        for i, line in enumerate(lines[:2]): # Max 2 lines
            draw_text_with_shadow(frame, line, (label_x, y_offset + (i * 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    else:
        draw_text_with_shadow(frame, note, (label_x, card_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


# ──────────────────────────────────────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, paused):
    h, w = frame.shape[:2]

    # Top floating bar
    overlay = frame.copy()
    draw_rounded_rect(overlay, (w//2 - 200, 15), (w//2 + 200, 60), (15, 15, 20), radius=15, fill=True)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    draw_rounded_rect(frame, (w//2 - 200, 15), (w//2 + 200, 60), (50, 50, 60), thickness=1, radius=15)
    
    # Title
    title = "SMART DUSTBIN VISION"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
    draw_text_with_shadow(frame, title, (w//2 - tw//2, 43), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_ACCENT, 2, shadow_offset=(1,1))

    # FPS pill (top right)
    fps_text = f"FPS: {fps:.0f}"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    overlay2 = frame.copy()
    draw_rounded_rect(overlay2, (w - fw - 40, 20), (w - 15, 50), (20, 20, 25), radius=10, fill=True)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
    draw_rounded_rect(frame, (w - fw - 40, 20), (w - 15, 50), (60, 60, 60), thickness=1, radius=10)
    
    color_fps = (50, 255, 50) if fps > 15 else (0, 150, 255) if fps > 5 else (0, 0, 255)
    draw_text_with_shadow(frame, fps_text, (w - fw - 28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_fps, 1)

    # Bottom control bar
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (0, h - 45), (w, h), (10, 10, 12), -1)
    cv2.addWeighted(overlay3, 0.85, frame, 0.15, 0, frame)
    
    # Separator line
    cv2.line(frame, (0, h - 45), (w, h - 45), COLOR_ACCENT, 1)

    controls = "Q / ESC to Quit   |   SPACE to Pause   |   S to Save Screenshot"
    (cw, _), _ = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    draw_text_with_shadow(frame, controls, (w//2 - cw//2, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Paused grand overlay
    if paused:
        overlay4 = frame.copy()
        cv2.rectangle(overlay4, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay4, 0.6, frame, 0.4, 0, frame)
        
        draw_rounded_rect(frame, (w // 2 - 120, h // 2 - 40), (w // 2 + 120, h // 2 + 30), (20, 20, 25), radius=15, fill=True)
        draw_rounded_rect(frame, (w // 2 - 120, h // 2 - 40), (w // 2 + 120, h // 2 + 30), (80, 80, 80), thickness=2, radius=15)
        
        draw_text_with_shadow(frame, "PAUSED", (w // 2 - 80, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (200, 200, 250), 2, shadow_offset=(2,2))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("🗑️   Smart Dustbin — Real-Time Webcam Classifier")
    print("=" * 55)

    model = load_model()

    # ── Open camera (macOS-compatible) ──────────────────────────────────
    cap = None
    # Try different backends — AVFoundation works best on macOS
    backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation"),
        (cv2.CAP_ANY, "Default"),
    ] if sys.platform == "darwin" else [
        (cv2.CAP_ANY, "Default"),
    ]

    for backend, name in backends:
        for cam_id in (0, 1):
            print(f"🔍  Trying camera {cam_id} with {name} backend...")
            cap = cv2.VideoCapture(cam_id, backend)
            if cap.isOpened():
                print(f"✅  Opened camera {cam_id} ({name})")
                break
            cap.release()
            cap = None
        if cap is not None and cap.isOpened():
            break

    if cap is None or not cap.isOpened():
        print("❌  Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Warm-up: discard initial black frames (macOS camera needs time) ──
    print("⏳  Warming up camera (discarding initial frames)...")
    for i in range(30):
        cap.read()
    time.sleep(0.5)  # Let auto-exposure settle
    print("✅  Camera ready!")

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
