# 🗑️ Smart Dustbin — Household Waste Classifier

An AI-powered household waste classification system built on **EfficientNetV2B0** with transfer learning. The model classifies waste images into **8 categories** and provides real-time sorting recommendations via webcam — helping automate smart dustbin operations.

> Part of the **IAI Research Project** — Intelligent Automated Infrastructure for waste management.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Waste Categories](#waste-categories)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Real-Time Inference](#real-time-inference)
- [Exporting Weights for Cross-Platform Use](#exporting-weights-for-cross-platform-use)
- [Configuration](#configuration)
- [Controls](#controls)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

The system uses a two-phase transfer learning approach on EfficientNetV2B0 (pre-trained on ImageNet) to classify household waste items captured via webcam. It offers three inference modes:

| Mode | Script | Description |
|------|--------|-------------|
| **Scan-Zone Webcam** | `webcam_classifier.py` | Futuristic HUD with a center scan zone — place items inside the box for classification |
| **YOLO + EfficientNet** | `yolo_webcam_classifier.py` | YOLOv8 detects objects anywhere in frame, then EfficientNet classifies each crop |
| **YOLO (Simple)** | `yolo_classifier.py` | Lightweight YOLO + EfficientNet pipeline without the styled UI |

---

## Waste Categories

The model classifies waste into **8 categories**, each mapped to a recommended bin:

| # | Category | Recommended Bin | Recyclable |
|---|----------|-----------------|:----------:|
| 1 | E-waste | E-Waste Bin | ❌ |
| 2 | Battery waste | Hazardous Bin | ❌ |
| 3 | Glass waste | Recyclable Bin | ✅ |
| 4 | Light bulbs | Hazardous Bin | ❌ |
| 5 | Metal waste | Recyclable Bin | ✅ |
| 6 | Organic waste | Organic / Wet Bin | ❌ |
| 7 | Paper waste | Paper Bin | ✅ |
| 8 | Plastic waste | Plastic Bin | ✅ |

---

## Architecture

### Model — EfficientNetV2B0 + Custom Head

```
EfficientNetV2B0 (ImageNet, frozen/fine-tuned)
        │
  GlobalAveragePooling2D
        │
  Dense(512, ReLU) + L2 Regularization
  BatchNormalization
  Dropout(0.4)
        │
  Dense(256, ReLU) + L2 Regularization
  BatchNormalization
  Dropout(0.3)
        │
  Dense(8, Softmax, float32)   ← output
```

### Two-Phase Training Strategy

| Phase | Epochs | Learning Rate | Strategy |
|-------|--------|---------------|----------|
| **Phase 1** — Head Training | 15 | 1e-3 (Adam) | Base model frozen; trains only the classification head. Uses `ReduceLROnPlateau` + `EarlyStopping`. |
| **Phase 2** — Fine-Tuning | 30 | 1e-4 → 1e-7 (CosineDecay) | Unfreezes top 80 layers of the base model. `ReduceLROnPlateau` is intentionally excluded (incompatible with `CosineDecay` schedule). |

### Key Training Features

- **Mixed Precision (FP16)** — Leverages NVIDIA Tensor Cores for faster training; softmax output remains in float32 for numerical stability
- **GPU Data Augmentation** — `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`, `RandomBrightness` applied on-GPU
- **Class Weighting** — Computed automatically from directory counts to handle class imbalance
- **Label Smoothing** — `CategoricalCrossentropy(label_smoothing=0.1)` for better generalization
- **Pipeline** — `unbatch → shuffle → batch (drop_remainder) → repeat → augment → prefetch` for optimal GPU utilization
- **XLA Warmup** — Pre-compiles computation graph before training to prevent mid-epoch stalls

---

## Project Structure

```
Household_Model/
├── smart_dustbin_classifier.py    # Training script (EfficientNetV2B0)
├── webcam_classifier.py           # Real-time scan-zone webcam classifier
├── yolo_webcam_classifier.py      # YOLO + EfficientNet webcam (fast, styled)
├── yolo_classifier.py             # YOLO + EfficientNet webcam (simple)
├── export_weights.py              # Export .weights.h5 for cross-platform use
├── test_webcam.py                 # Webcam connectivity test utility
├── class_names.txt                # Ordered class names (8 categories)
├── requirements.txt               # Python dependencies
├── smart_dustbin_model.keras      # Full saved model (Keras format)
├── smart_dustbin_model.weights.h5 # Exported weights only (portable)
├── yolov8n.pt                     # YOLOv8 Nano weights (for YOLO modes)
├── screenshots/                   # Saved screenshots (S key during inference)
├── plots/                         # Training history & confusion matrix plots
├── household_wastes/              # Dataset directory (not included in repo)
│   └── wastes/
│       ├── train/                 # Training images (per-class subdirectories)
│       └── test/                  # Test images (per-class subdirectories)
└── README.md                      # This file
```

---

## Setup & Installation

### Prerequisites

- **Python** 3.10+
- **GPU** (recommended): NVIDIA GPU with CUDA support, or Apple Silicon (M1/M2/M3)
- **Webcam** (for real-time inference)

### 1. Clone & Navigate

```bash
cd IAI_Project/Household_Model
```

### 2. Create Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For YOLO-based inference modes, also install:

```bash
pip install ultralytics opencv-python
```

### 4. Dataset Setup

Organize your dataset under `household_wastes/wastes/` with the following structure:

```
household_wastes/
└── wastes/
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
        ├── battery waste/
        ├── glass waste/
        ├── light bulbs/
        ├── metal waste/
        ├── organic waste/
        ├── paper waste/
        └── plastic waste/
```

> **Note:** The `light bulbs` class appears only in the test set in the original dataset configuration.

### WSL-Specific Setup (Windows)

If training on WSL with an NVIDIA GPU, create `C:\Users\<YourUsername>\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=4GB
```

Then restart WSL:

```bash
wsl --shutdown
```

---

## Training the Model

```bash
python smart_dustbin_classifier.py
```

**What happens:**

1. Loads and preprocesses the dataset with a 80/20 train/validation split
2. Builds the EfficientNetV2B0 model with a custom classification head
3. **Phase 1** — Trains the head (base frozen) for up to 15 epochs
4. **Phase 2** — Fine-tunes the top 80 base layers for up to 30 epochs with CosineDecay
5. Saves best model (`smart_dustbin_model_best.keras`) and last model (`smart_dustbin_model_last.keras`)
6. Generates training history plots and confusion matrix in `plots/`
7. Prints a full classification report with per-class precision, recall, and F1

### Training Outputs

| File | Description |
|------|-------------|
| `smart_dustbin_model_best.keras` | Best model (highest val_accuracy) |
| `smart_dustbin_model_last.keras` | Final model after all epochs |
| `plots/training_history.png` | Accuracy & loss curves with phase boundary |
| `plots/confusion_matrix.png` | Per-class confusion matrix |
| `plots/sample_predictions.png` | Grid of 12 sample test predictions |

---

## Real-Time Inference

### Option 1: Scan-Zone Webcam (Recommended)

```bash
python webcam_classifier.py
```

Features a futuristic HUD with:
- 🎯 Animated center scan zone with sweeping laser effect
- 📊 Glassmorphic result panel with confidence bar
- ♻️ Recyclable badge and bin routing recommendation
- 🎛️ FPS counter and pause/screenshot controls

### Option 2: YOLO + EfficientNet (Multi-Object)

```bash
python yolo_webcam_classifier.py
```

Detects **multiple objects** anywhere in the frame using YOLOv8, then classifies each detection with EfficientNet. Optimized for speed with:
- Batch prediction (all crops in a single model call)
- Classify every 2nd frame with cached result drawing
- Lower capture resolution (640×480)

### Option 3: YOLO Simple

```bash
python yolo_classifier.py
```

Minimal version of the YOLO pipeline — same detection logic, simpler UI.

### Testing Your Webcam

If you're having camera issues, run the diagnostic script:

```bash
python test_webcam.py
```

---

## Exporting Weights for Cross-Platform Use

If the model was trained on WSL/Linux and you need to run inference on macOS/Windows:

```bash
python export_weights.py
```

This creates:
- `smart_dustbin_model.weights.h5` — Portable weights file
- `class_names.txt` — Class order reference

The webcam classifiers automatically detect and prefer `.weights.h5` over `.keras` for maximum compatibility.

---

## Configuration

Key hyperparameters in `smart_dustbin_classifier.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | (240, 240) | Input image dimensions |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS_FROZEN` | 15 | Phase 1 max epochs |
| `EPOCHS_FINE_TUNE` | 30 | Phase 2 max epochs |
| `FINE_TUNE_LAYERS` | 80 | Number of base layers to unfreeze |
| `LEARNING_RATE` | 1e-3 | Phase 1 learning rate |
| `FINE_TUNE_LR` | 1e-4 | Phase 2 initial learning rate |
| `VALIDATION_SPLIT` | 0.2 | Train/validation split ratio |

Webcam inference settings in `webcam_classifier.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESH` | 0.50 | Minimum confidence to display prediction |
| `SMOOTHING_FRAMES` | 7 | Number of frames to average predictions over |
| `SCAN_ZONE_RATIO` | 0.45 | Scan zone size relative to frame |

---

## Controls

During real-time inference:

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot to `screenshots/` |
| `SPACE` | Pause / Resume |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning Framework** | TensorFlow / Keras 2.18+ |
| **Base Model** | EfficientNetV2B0 (ImageNet) |
| **Object Detection** | YOLOv8 Nano (Ultralytics) |
| **Computer Vision** | OpenCV |
| **Metrics** | scikit-learn |
| **Precision** | Mixed FP16 (training) / FP32 (inference softmax) |

---

## License

This project is developed as part of an academic research initiative.
