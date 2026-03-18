import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# 1. Initialize models ONCE outside the loop
print("Loading YOLOv8n...")
yolo_model = YOLO("yolov8n.pt")  # Auto-downloads the nano model if missing

print("Loading EfficientNet classifier...")
# Make sure this points to your trained model file!
classifier_model = tf.keras.models.load_model("smart_dustbin_model_best.keras")

# Your trained classes
CLASS_NAMES = [
    "E-waste", "battery waste", "glass waste", "light bulbs", 
    "metal waste", "organic waste", "paper waste", "plastic waste"
]

cap = cv2.VideoCapture(0)

print("Starting webcam feed...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 2. Run YOLO Detection
    # stream=True is faster for video, verbose=False hides console spam
    results = yolo_model(frame, stream=True, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates (x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_yolo = float(box.conf[0])
            
            # Optional: Filter by YOLO confidence (e.g., > 0.4)
            if conf_yolo < 0.4:
                continue
                
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)

            # Skip invalid/tiny crops
            if y2 - y1 < 10 or x2 - x1 < 10:
                continue

            # 3. Crop, Resize, and Preprocess for EfficientNet
            crop = frame[y1:y2, x1:x2]
            
            # Resize to exactly what EfficientNet expects (240x240 for your model)
            crop_resized = cv2.resize(crop, (240, 240)) 
            
            # Convert BGR (OpenCV) to RGB (TensorFlow expects RGB)
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            # Expand dims to create batch of 1 and apply EfficientNetV2 preprocessing
            crop_preprocessed = preprocess_input(np.expand_dims(crop_rgb.astype(np.float32), axis=0))

            # Pass to EfficientNet classifier
            predictions = classifier_model.predict(crop_preprocessed, verbose=0)[0]
            pred_idx = np.argmax(predictions)
            conf_class = predictions[pred_idx]
            label = CLASS_NAMES[pred_idx]

            # 4. Output: Draw YOLO box and EfficientNet label
            # Draw bounding box (Blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Display EfficientNet label and Confidence above the box
            text = f"{label} ({conf_class*100:.1f}%)"
            
            # Background for text to make it readable
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Make sure text background doesn't go off top of screen
            text_y = max(y1, 25)
            cv2.rectangle(frame, (x1, text_y - 25), (x1 + tw, text_y), (255, 0, 0), -1)
            cv2.putText(frame, text, (x1, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLO + Dustbin Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
