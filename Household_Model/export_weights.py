"""
One-time script to export model weights for cross-platform use.
Run this ONCE in WSL where the model was trained:

    python export_weights.py

This creates:
    smart_dustbin_model.weights.h5   (portable weights file)
    class_names.txt                  (class order reference)
"""

import os
import tensorflow as tf

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "smart_dustbin_model.keras")
WEIGHTS_PATH = os.path.join(BASE_DIR, "smart_dustbin_model.weights.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "class_names.txt")

print("📦  Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅  Model loaded — {model.count_params():,} parameters")
print(f"   Output classes: {model.output_shape[-1]}")

# Save weights only
model.save_weights(WEIGHTS_PATH)
print(f"💾  Weights saved to: {WEIGHTS_PATH}")

# Also save class names from the training directory
TRAIN_DIR = os.path.join(BASE_DIR, "household_wastes", "wastes", "train")
if os.path.exists(TRAIN_DIR):
    class_names = sorted(os.listdir(TRAIN_DIR))
    # Filter out hidden files
    class_names = [c for c in class_names if not c.startswith('.')]
    with open(CLASSES_PATH, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"📝  Class names saved to: {CLASSES_PATH}")
    print(f"   Classes: {class_names}")

print("\n✅  Done! Now run webcam_classifier.py on Windows.")
