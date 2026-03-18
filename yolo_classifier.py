import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ── Initialize models ─────────────────────────────────────────────────────────
print("Loading YOLOv8n...")
yolo_model = YOLO("yolov8n.pt")

print("Loading EfficientNet classifier...")

def build_model_architecture(num_classes):
    from tensorflow.keras.applications import EfficientNetV2B0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
    from tensorflow.keras.models import Model
    base_model = EfficientNetV2B0(weights=None, include_top=False, input_shape=(240, 240, 3))
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

CLASS_NAMES = [
    "E-waste", "battery waste", "glass waste", "light bulbs",
    "metal waste", "organic waste", "paper waste", "plastic waste"
]

classifier_model = build_model_architecture(len(CLASS_NAMES))
classifier_model.load_weights("smart_dustbin_model.weights.h5")
print("✅ EfficientNet loaded!")

# ── Webcam at 720p ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── FPS state ─────────────────────────────────────────────────────────────────
CLASSIFY_EVERY_N  = 2      # Run EfficientNet every 2nd frame
YOLO_CONF_THRESH  = 0.4
frame_count       = 0
cached_results    = []     # (x1, y1, x2, y2, label, conf_class)

print("Starting webcam feed...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # ── FIX: Only run detection + classification every Nth frame ─────────────
    if frame_count % CLASSIFY_EVERY_N == 0:

        cached_results = []
        crops          = []
        box_coords     = []

        # YOLO detection
        results = yolo_model(frame, verbose=False, conf=YOLO_CONF_THRESH)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                if y2 - y1 < 10 or x2 - x1 < 10:
                    continue

                # Collect crop
                crop     = frame[y1:y2, x1:x2]
                resized  = cv2.resize(crop, (240, 240))
                rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                crops.append(rgb.astype(np.float32))
                box_coords.append((x1, y1, x2, y2))

        # ── FIX: Batch predict — one call for ALL boxes ───────────────────────
        if crops:
            batch       = preprocess_input(np.stack(crops))
            all_preds   = classifier_model.predict(batch, verbose=0)

            for i, (x1, y1, x2, y2) in enumerate(box_coords):
                pred_idx   = np.argmax(all_preds[i])
                conf_class = all_preds[i][pred_idx]
                label      = CLASS_NAMES[pred_idx]
                cached_results.append((x1, y1, x2, y2, label, conf_class))

    # ── Draw cached results on every frame ────────────────────────────────────
    for (x1, y1, x2, y2, label, conf_class) in cached_results:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        text = f"{label} ({conf_class*100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1, 25)
        cv2.rectangle(frame, (x1, text_y - 25), (x1 + tw, text_y), (255, 0, 0), -1)
        cv2.putText(frame, text, (x1, text_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLO + Dustbin Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()