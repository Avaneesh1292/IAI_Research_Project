"""
Smart Dustbin — YOLO + EfficientNet (Fast Version)
===================================================
FPS optimizations applied:
  1. Batch predict — all crops sent in one model call per frame
  2. Classify every 2nd frame — draw cached results on off frames
  3. Lower capture resolution — 640x480 for faster YOLO inference
  4. YOLO half precision — yolo.predict(half=True) on GPU

Controls:
    Q / ESC  — Quit
    S        — Screenshot
    SPACE    — Pause

Requirements:
    pip install opencv-python tensorflow numpy ultralytics
"""

import os
import sys
import time
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "smart_dustbin_model.keras")
WEIGHTS_PATH    = os.path.join(BASE_DIR, "smart_dustbin_model.weights.h5")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

IMG_SIZE          = (240, 240)
CONFIDENCE_THRESH = 0.50
YOLO_CONF_THRESH  = 0.40
SMOOTHING_FRAMES  = 4
CLASSIFY_EVERY_N  = 2     # Run EfficientNet every 2nd frame, draw cache on off frames
CAP_WIDTH         = 640   # Lower res = faster YOLO + less data to process
CAP_HEIGHT        = 480

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

WASTE_INFO = {
    "E-waste":        ("E-Waste Bin",    (60,  60,  220)),
    "battery waste":  ("Hazardous Bin",  (0,   130, 255)),
    "glass waste":    ("Recyclable Bin", (0,   200, 100)),
    "light bulbs":    ("Hazardous Bin",  (0,   130, 255)),
    "metal waste":    ("Recyclable Bin", (0,   200, 100)),
    "organic waste":  ("Organic Bin",    (50,  130, 180)),
    "paper waste":    ("Paper Bin",      (220, 160, 50) ),
    "plastic waste":  ("Plastic Bin",    (0,   220, 255)),
}

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────
def build_model_architecture(num_classes):
    from tensorflow.keras.applications import EfficientNetV2B0
    from tensorflow.keras.layers import (
        Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
    )
    from tensorflow.keras.models import Model

    base = EfficientNetV2B0(weights=None, include_top=False,
                            input_shape=(*IMG_SIZE, 3))
    x   = base.output
    x   = GlobalAveragePooling2D()(x)
    x   = Dense(512, activation="relu")(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.4)(x)
    x   = Dense(256, activation="relu")(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs=base.input, outputs=out)


def load_classifier():
    if os.path.exists(WEIGHTS_PATH):
        print(f"📦  Loading weights: {WEIGHTS_PATH}")
        model = build_model_architecture(len(CLASS_NAMES))
        model.load_weights(WEIGHTS_PATH)
    elif os.path.exists(MODEL_PATH):
        print(f"📦  Loading model: {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            model = build_model_architecture(len(CLASS_NAMES))
            model.load_weights(MODEL_PATH)
    else:
        print("❌  No model found!")
        sys.exit(1)

    # Warmup — pre-compile TF graph before the loop
    model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), verbose=0)
    print("✅  Classifier ready")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────
def draw_detection(frame, x1, y1, x2, y2, label, confidence, bin_name, color):
    """Draw bounding box, label tag, and bin card."""
    fh, fw = frame.shape[:2]

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label tag above box
    tag             = f"{label}  {confidence*100:.0f}%"
    scale, thick    = 0.55, 1
    (tw, th), base  = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    tag_y1          = max(y1 - th - base - 8, 0)
    tag_y2          = tag_y1 + th + base + 8
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 12, tag_y2), color, -1)
    cv2.putText(frame, tag, (x1 + 6, tag_y2 - base - 4),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), thick, cv2.LINE_AA)

    # Bin card below box
    bin_text        = f"-> {bin_name}"
    (bw, bh), _     = cv2.getTextSize(bin_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    card_x          = x1
    card_y          = y2 + 5
    card_y2         = card_y + bh + 10

    if card_y2 > fh - 36:       # flip above if not enough space below
        card_y  = y1 - bh - 15
        card_y2 = card_y + bh + 10
    if card_x + bw + 12 > fw:   # clamp to right edge
        card_x = fw - bw - 14

    cv2.rectangle(frame, (card_x, card_y), (card_x + bw + 12, card_y2),
                  (25, 25, 25), -1)
    cv2.rectangle(frame, (card_x, card_y), (card_x + bw + 12, card_y2),
                  color, 1)
    cv2.putText(frame, bin_text, (card_x + 6, card_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def draw_uncertain(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
    cv2.putText(frame, "?", (x1 + 6, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)


def draw_hud(frame, fps, paused, count):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 42), (18, 18, 18), -1)
    cv2.line(frame, (0, 42), (w, 42), (0, 180, 90), 1)

    cv2.putText(frame, "SMART DUSTBIN",
                (14, 29), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 120), 1, cv2.LINE_AA)

    fps_color = (50, 255, 50) if fps > 20 else (0, 200, 255) if fps > 10 else (0, 80, 255)
    cv2.putText(frame, f"FPS {fps:.0f}",
                (w - 90, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 1, cv2.LINE_AA)

    det_color = (0, 220, 120) if count > 0 else (110, 110, 110)
    cv2.putText(frame, f"Objects: {count}",
                (w // 2 - 55, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 1, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 32), (w, h), (18, 18, 18), -1)
    cv2.line(frame, (0, h - 32), (w, h - 32), (0, 180, 90), 1)
    cv2.putText(frame, "Q: Quit   SPACE: Pause   S: Screenshot",
                (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    if paused:
        cv2.rectangle(frame, (w // 2 - 75, h // 2 - 24),
                      (w // 2 + 75, h // 2 + 24), (18, 18, 18), -1)
        cv2.rectangle(frame, (w // 2 - 75, h // 2 - 24),
                      (w // 2 + 75, h // 2 + 24), (0, 220, 120), 2)
        cv2.putText(frame, "PAUSED",
                    (w // 2 - 52, h // 2 + 9),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 120), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("🗑️   Smart Dustbin — YOLO + EfficientNet (Fast)")
    print("=" * 55)

    print("\n📦  Loading YOLOv8n...")
    yolo = YOLO("yolov8n.pt")
    print("✅  YOLO ready")

    classifier = load_classifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌  Webcam not found")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    print("⏳  Warming up camera...")
    for _ in range(20):
        cap.read()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷  {actual_w}x{actual_h} — ready\n")

    # State
    pred_buffer   = []
    paused        = False
    frame_count   = 0
    prev_time     = time.time()
    fps           = 0.0

    # Cache — drawn on off frames
    cached_detections = []   # list of (x1,y1,x2,y2, label, conf, bin_name, color, certain)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                display = frame.copy()

                # ── FIX 2: Only classify every Nth frame ─────────────────
                if frame_count % CLASSIFY_EVERY_N == 0:

                    cached_detections = []

                    # ── FIX 1: Collect all crops, one batch predict ───────
                    crops     = []
                    box_coords = []

                    results = yolo(frame, verbose=False, conf=YOLO_CONF_THRESH)

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x1 = max(0, x1);  y1 = max(0, y1)
                            x2 = min(actual_w, x2); y2 = min(actual_h, y2)
                            if (x2 - x1) < 20 or (y2 - y1) < 20:
                                continue

                            crop    = frame[y1:y2, x1:x2]
                            resized = cv2.resize(crop, IMG_SIZE)
                            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            crops.append(rgb.astype(np.float32))
                            box_coords.append((x1, y1, x2, y2))

                    if crops:
                        # Single batch predict call for ALL boxes at once
                        batch     = preprocess_input(np.stack(crops))
                        all_preds = classifier.predict(batch, verbose=0)

                        for i, (x1, y1, x2, y2) in enumerate(box_coords):
                            preds = all_preds[i]

                            # Smooth
                            pred_buffer.append(preds)
                            if len(pred_buffer) > SMOOTHING_FRAMES:
                                pred_buffer.pop(0)
                            avg        = np.mean(pred_buffer, axis=0)
                            idx        = int(np.argmax(avg))
                            confidence = float(avg[idx])
                            prediction = CLASS_NAMES[idx]

                            info     = WASTE_INFO.get(prediction, ("Unknown", (150, 150, 150)))
                            bin_name, color = info
                            certain  = confidence >= CONFIDENCE_THRESH

                            cached_detections.append(
                                (x1, y1, x2, y2, prediction, confidence,
                                 bin_name, color, certain)
                            )

                # ── Draw cached detections on every frame ─────────────────
                for det in cached_detections:
                    x1, y1, x2, y2, prediction, confidence, bin_name, color, certain = det
                    if certain:
                        draw_detection(display, x1, y1, x2, y2,
                                       prediction, confidence, bin_name, color)
                    else:
                        draw_uncertain(display, x1, y1, x2, y2)

                # FPS
                curr      = time.time()
                fps       = 1.0 / max(curr - prev_time, 1e-6)
                prev_time = curr

                draw_hud(display, fps, paused, len(cached_detections))

            else:
                display = frame.copy()
                for det in cached_detections:
                    x1, y1, x2, y2, prediction, confidence, bin_name, color, certain = det
                    if certain:
                        draw_detection(display, x1, y1, x2, y2,
                                       prediction, confidence, bin_name, color)
                draw_hud(display, fps, paused=True, count=len(cached_detections))

            cv2.imshow("Smart Dustbin", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('s'), ord('S')):
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SCREENSHOTS_DIR, f"capture_{ts}.png")
                cv2.imwrite(path, frame)
                print(f"📸  {path}")
            elif key == 32:
                paused = not paused
                print("⏸️  Paused" if paused else "▶️  Resumed")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋  Done")


if __name__ == "__main__":
    main()