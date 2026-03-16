"""
Smart Dustbin Image Classification using EfficientNetV2B0
=========================================================
Optimized for:
  - RTX 4060 Laptop GPU (8GB VRAM)
  - 16GB system RAM on WSL
  - Mixed precision FP16 for Tensor Core acceleration
  - unbatch → shuffle → repeat → batch pipeline
  - Separate callbacks per phase (fixes ReduceLROnPlateau + CosineDecay conflict)

Before running, create C:\\Users\\YourUsername\\.wslconfig:
    [wsl2]
    memory=12GB
    swap=4GB
Then run: wsl --shutdown

Dataset structure expected:
    household_wastes/
    ├── train/
    │   ├── E-waste/
    │   ├── battery waste/
    │   ├── glass waste/
    │   ├── metal waste/
    │   ├── organic waste/
    │   ├── paper waste/
    │   └── plastic waste/
    └── test/
        ├── E-waste/
        ├── automobile wastes/
        ├── battery waste/
        ├── glass waste/
        ├── light bulbs/
        ├── metal waste/
        ├── organic waste/
        ├── paper waste/
        └── plastic waste/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomBrightness
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ──────────────────────────────────────────────────────────────────────────────
# GPU Setup
# ──────────────────────────────────────────────────────────────────────────────
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("   Mixed precision     : ✅ Enabled (float16 — RTX Tensor Cores active)")
    else:
        print("   ⚠️  No GPU found, running on CPU")
    return gpus


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR       = os.path.join(BASE_DIR, "household_wastes", "wastes", "train")
TEST_DIR        = os.path.join(BASE_DIR, "household_wastes", "wastes", "test")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "smart_dustbin_model.keras")
PLOTS_DIR       = os.path.join(BASE_DIR, "plots")

IMG_SIZE         = (240, 240)
BATCH_SIZE       = 32
EPOCHS_FROZEN    = 20
EPOCHS_FINE_TUNE = 25
FINE_TUNE_LAYERS = 80
LEARNING_RATE    = 1e-3
FINE_TUNE_LR     = 1e-5
VALIDATION_SPLIT = 0.2
SHUFFLE_BUFFER   = 500
PREFETCH_BUFFER  = 4

os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def create_datasets():
    print("\n" + "=" * 60)
    print("📂  Loading and Preparing Dataset")
    print("=" * 60)

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        label_mode="categorical",
        seed=42,
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        label_mode="categorical",
        seed=42,
    )

    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.15),
        RandomTranslation(0.15, 0.15),
        RandomBrightness(0.15),
    ], name="gpu_augmentation")

    def preprocess(images, labels):
        return preprocess_input(tf.cast(images, tf.float32)), labels

    def augment_and_preprocess(images, labels):
        images = augmentation(tf.cast(images, tf.float32), training=True)
        return preprocess_input(images), labels

    train_ds = (
        train_ds_raw
        .unbatch()
        .shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=True)
        .repeat()
        .map(augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=PREFETCH_BUFFER)
    )

    val_ds = (
        val_ds_raw
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=PREFETCH_BUFFER)
    )

    test_ds = (
        test_ds_raw
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=PREFETCH_BUFFER)
    )

    # Compute steps_per_epoch and class weights from directory counts
    class_counts = {}
    total_train  = 0
    for i, name in enumerate(class_names):
        path  = os.path.join(TRAIN_DIR, name)
        count = int(len(os.listdir(path)) * (1 - VALIDATION_SPLIT))
        class_counts[i] = max(1, count)
        total_train += count

    steps_per_epoch = total_train // BATCH_SIZE

    class_weight_dict = {
        i: total_train / (num_classes * class_counts[i])
        for i in range(num_classes)
    }

    print(f"\n✅  Classes ({num_classes}): {class_names}")
    print(f"   Training images   : {total_train}")
    print(f"   Steps per epoch   : {steps_per_epoch}")
    print(f"   Class weights     : { {k: round(v, 3) for k, v in class_weight_dict.items()} }")
    print(f"\n   Batch size        : {BATCH_SIZE}")
    print(f"   Shuffle buffer    : {SHUFFLE_BUFFER} images")
    print(f"   Prefetch buffer   : {PREFETCH_BUFFER}")
    print(f"   Repeat            : ✅ (continuous stream)")
    print(f"   Drop remainder    : ✅ (consistent batch shape)")
    print(f"   Cache             : ❌ Disabled (WSL memory safety)")
    print(f"   GPU augmentation  : ✅ Enabled")

    return (
        train_ds, val_ds, test_ds, test_ds_raw,
        class_names, num_classes, class_weight_dict, steps_per_epoch
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model
# ──────────────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    print("\n" + "=" * 60)
    print("🏗️   Building EfficientNetV2B0 Model")
    print("=" * 60)

    base_model = EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # float32 required — FP16 numerically unstable for softmax
    predictions = Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    total_params     = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    print(f"\n   Total parameters     : {total_params:,}")
    print(f"   Trainable parameters : {trainable_params:,}")
    print(f"   Base model layers    : {len(base_model.layers)}")
    print(f"   Precision policy     : {tf.keras.mixed_precision.global_policy().name}")

    return model, base_model


# ──────────────────────────────────────────────────────────────────────────────
# 3. Callbacks — separate per phase to avoid LR schedule conflict
# ──────────────────────────────────────────────────────────────────────────────
def get_callbacks_phase1():
    """
    Phase 1 callbacks — includes ReduceLROnPlateau which works with
    a fixed float learning rate (Adam with LEARNING_RATE).
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def get_callbacks_phase2():
    """
    Phase 2 callbacks — NO ReduceLROnPlateau.
    CosineDecay schedule handles LR decay automatically.
    ReduceLROnPlateau is incompatible with LearningRateSchedule objects
    and will raise a TypeError if included.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training
# ──────────────────────────────────────────────────────────────────────────────
def warmup(model, train_ds):
    """Pre-compile XLA kernels — prevents mid-epoch stall."""
    print("\n   🔥  Warming up — pre-compiling XLA kernels...")
    for images, _ in train_ds.take(1):
        model(images, training=False)
    print("   ✅  Warmup done — GPU ready\n")


def train_model(model, base_model, train_ds, val_ds, class_weight_dict, steps_per_epoch):
    """
    Phase 1: Train classification head only (base frozen).
             Uses ReduceLROnPlateau + EarlyStopping + ModelCheckpoint.
    Phase 2: Fine-tune top layers with CosineDecay LR schedule.
             Uses EarlyStopping + ModelCheckpoint only (no ReduceLROnPlateau).
    """

    # ── Phase 1 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🚀  Phase 1: Training Classification Head (base frozen)")
    print("=" * 60)

    warmup(model, train_ds)

    history_frozen = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_FROZEN,
        validation_data=val_ds,
        callbacks=get_callbacks_phase1(),
        class_weight=class_weight_dict,
        verbose=1,
    )

    # ── Phase 2 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"🔧  Phase 2: Fine-Tuning top {FINE_TUNE_LAYERS} layers")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    total_fine_tune_steps = steps_per_epoch * EPOCHS_FINE_TUNE
    cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=FINE_TUNE_LR,
        decay_steps=total_fine_tune_steps,
        alpha=1e-7,
    )

    # CosineDecay handles LR — do NOT include ReduceLROnPlateau here
    model.compile(
        optimizer=Adam(learning_rate=cosine_schedule),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"   Trainable parameters after unfreezing: {trainable_params:,}")

    warmup(model, train_ds)

    history_fine = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_FINE_TUNE,
        validation_data=val_ds,
        callbacks=get_callbacks_phase2(),
        class_weight=class_weight_dict,
        verbose=1,
    )

    return history_frozen, history_fine


# ──────────────────────────────────────────────────────────────────────────────
# 5. Evaluation & Visualization
# ──────────────────────────────────────────────────────────────────────────────
def plot_training_history(history_frozen, history_fine):
    acc      = history_frozen.history["accuracy"]     + history_fine.history["accuracy"]
    val_acc  = history_frozen.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss     = history_frozen.history["loss"]         + history_fine.history["loss"]
    val_loss = history_frozen.history["val_loss"]     + history_fine.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)
    phase1_end   = len(history_frozen.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs_range, acc,     "b-", label="Training Accuracy",   linewidth=2)
    ax1.plot(epochs_range, val_acc, "r-", label="Validation Accuracy", linewidth=2)
    ax1.axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.7, label="Fine-Tuning Start")
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, loss,     "b-", label="Training Loss",   linewidth=2)
    ax2.plot(epochs_range, val_loss, "r-", label="Validation Loss", linewidth=2)
    ax2.axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.7, label="Fine-Tuning Start")
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Smart Dustbin Classifier — Training History",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊  Training history saved to: {path}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im  = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊  Confusion matrix saved to: {path}")


def evaluate_model(model, test_ds, test_ds_raw):
    print("\n" + "=" * 60)
    print("📈  Evaluating Model on Test Set")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\n   Test Loss     : {test_loss:.4f}")
    print(f"   Test Accuracy : {test_accuracy:.4f}  ({test_accuracy * 100:.2f}%)")

    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([
        np.argmax(labels.numpy(), axis=1)
        for _, labels in test_ds_raw
    ])

    class_names = test_ds_raw.class_names
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names)
    return test_loss, test_accuracy


def plot_sample_predictions(model, test_ds_raw):
    class_names  = test_ds_raw.class_names
    idx_to_class = {i: name for i, name in enumerate(class_names)}

    raw_images, raw_labels = next(iter(test_ds_raw))
    preprocessed = preprocess_input(tf.cast(raw_images, tf.float32))
    predictions  = model.predict(preprocessed, verbose=0)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Sample Predictions — Smart Dustbin Classifier",
                 fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= len(raw_images):
            ax.axis("off")
            continue
        ax.imshow(raw_images[i].numpy().astype("uint8"))
        pred_idx   = np.argmax(predictions[i])
        true_idx   = np.argmax(raw_labels[i].numpy())
        confidence = predictions[i][pred_idx] * 100
        color = "green" if pred_idx == true_idx else "red"
        ax.set_title(
            f"Pred: {idx_to_class[pred_idx]}\n"
            f"True: {idx_to_class[true_idx]}\n"
            f"Conf: {confidence:.1f}%",
            fontsize=9, color=color, fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊  Sample predictions saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Inference Utility
# ──────────────────────────────────────────────────────────────────────────────
def predict_single_image(model, image_path, class_names):
    """Predict waste category for a single image."""
    img       = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    pred_idx    = np.argmax(predictions[0])
    confidence  = predictions[0][pred_idx] * 100

    print(f"\n🗑️   Prediction for: {os.path.basename(image_path)}")
    print(f"   Category   : {class_names[pred_idx]}")
    print(f"   Confidence : {confidence:.2f}%")
    print("   Top-3:")
    for rank, idx in enumerate(np.argsort(predictions[0])[::-1][:3], 1):
        print(f"     {rank}. {class_names[idx]:20s} — {predictions[0][idx] * 100:.2f}%")

    return class_names[pred_idx], confidence


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🗑️   Smart Dustbin — Waste Image Classifier")
    print("   EfficientNetV2B0 | RTX 4060 Laptop | WSL Optimized")
    print("=" * 60)

    gpus = setup_gpu()
    print(f"\n   TensorFlow version : {tf.__version__}")
    print(f"   GPU available      : {'✅ Yes' if gpus else '❌ No (using CPU)'}")
    if gpus:
        for gpu in gpus:
            print(f"   GPU device         : {gpu.name}")

    # Step 1 — Load data
    (train_ds, val_ds, test_ds, test_ds_raw,
     class_names, num_classes, class_weight_dict, steps_per_epoch) = create_datasets()

    # Step 2 — Build model
    model, base_model = build_model(num_classes)
    model.summary()

    # Step 3 — Train
    history_frozen, history_fine = train_model(
        model, base_model, train_ds, val_ds, class_weight_dict, steps_per_epoch
    )

    # Step 4 — Plot history
    plot_training_history(history_frozen, history_fine)

    # Step 5 — Evaluate
    evaluate_model(model, test_ds, test_ds_raw)

    # Step 6 — Sample predictions
    plot_sample_predictions(model, test_ds_raw)

    # Step 7 — Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾  Model saved to: {MODEL_SAVE_PATH}")

    # Uncomment to run inference on a single image:
    # predict_single_image(model, "/path/to/image.jpg", class_names)

    print("\n" + "=" * 60)
    print("✅  Training and evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()