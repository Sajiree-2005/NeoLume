# NeoLume — Smartphone-Based Newborn Jaundice Risk Screening

> **An intelligent, explainable screening system that uses smartphone images and color analysis to estimate newborn jaundice risk.**

NeoLume is a lightweight AI-assisted screening prototype designed to make **early jaundice risk assessment more accessible in resource-constrained and rural settings**.

The system uses a smartphone camera to capture an image of a newborn's skin along with a calibration reference, performs image preprocessing and color normalization, extracts clinically relevant color features, and uses a machine-learning model to estimate jaundice risk.

> **Important:** NeoLume is a screening aid, not a diagnostic device. Any elevated-risk result should be followed by appropriate clinical evaluation and confirmation through established diagnostic methods such as a Total Serum Bilirubin (TSB) test.

---

# 🚀 Live Deployment

NeoLume is available as a live web application with a separately deployed backend API.

### 🌐 Frontend

**Live Application:**  
[NeoLume](https://neo-lume.vercel.app/)

Use the frontend to:
- Access the camera
- Capture newborn images
- Select calibration and skin regions
- Submit images for analysis
- View the screening result and explanation

### ⚙️ Backend API

**Live API:**  
[NeoLume Backend](https://neolume.onrender.com)

**API Health Check:**  
[https://your-backend-url.com/health](https://neolume.onrender.com/health)

The backend provides:
- Image preprocessing
- Color calibration
- Feature extraction
- Machine-learning inference
- Risk classification
- Screening explanation

---

## Table of Contents

- [Overview](#overview)
- [Problem](#problem)
- [Solution](#solution)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Class Imbalance Handling](#class-imbalance-handling)
- [Model Evaluation](#model-evaluation)
- [Risk Classification](#risk-classification)
- [Explainability](#explainability)
- [Frontend Workflow](#frontend-workflow)
- [Backend API](#backend-api)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Training the Model](#training-the-model)
- [Google Colab Training](#google-colab-training)
- [Offline / Fallback Operation](#offline--fallback-operation)
- [Deployment Architecture](#deployment-architecture)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Development Status](#development-status)
- [Clinical and Safety Disclaimer](#clinical-and-safety-disclaimer)

---

# Overview

Newborn jaundice is common, but excessive bilirubin levels can become dangerous if they are not identified and treated in time.

In many settings, especially where laboratory facilities or trained personnel are not immediately available, there is a need for a **low-cost, portable, and easy-to-use preliminary screening mechanism**.

NeoLume explores a smartphone-based approach where:

1. A caregiver or healthcare worker captures an image of the newborn.
2. A calibration reference is captured in the same image.
3. The relevant skin region is selected.
4. The image is color-corrected using the calibration reference.
5. Image color characteristics are extracted.
6. A trained machine-learning model estimates jaundice risk.
7. The system returns a simple risk category:
   - `LOW`
   - `MONITOR`
   - `REFER_URGENTLY`

The system is designed as a **screening aid, not a diagnostic device**.

---

# Problem

Visual assessment of jaundice can be affected by:

- Ambient lighting
- Camera differences
- White balance
- Skin tone
- Background colors
- Device hardware
- Image quality
- Subjective human assessment

A newborn may therefore appear more or less yellow depending on the environment in which the image is captured.

At the same time, laboratory-based bilirubin testing may not always be immediately accessible.

NeoLume addresses this gap by investigating whether **standardized smartphone image capture combined with color calibration and machine learning** can provide a useful preliminary screening signal.

---

# Solution

NeoLume combines a guided smartphone capture workflow with an explainable image-analysis pipeline.

```text
Smartphone Camera
       │
       ▼
Guided Image Capture
       │
       ├───────────────┐
       ▼               ▼
Calibration Card    Newborn Skin
       │               │
       └───────┬───────┘
               ▼
       White Balance Correction
               │
               ▼
       Image Preprocessing
               │
               ▼
       LAB / HSV Feature Extraction
               │
               ▼
       Machine Learning Model
               │
               ▼
       Risk Probability
               │
               ▼
       Risk Stratification
               │
       ┌───────┼───────────────┐
       ▼       ▼               ▼
      LOW   MONITOR    REFER_URGENTLY
               │
               ▼
       Human-readable explanation
```

The architecture separates:

**Image Capture → Calibration → Feature Extraction → ML Inference → Risk Stratification → Explanation**

This makes the system lightweight, interpretable, and suitable for future edge deployment.

---

# Key Features

## Smartphone-Based Guided Capture

The web interface provides an interactive camera workflow with visual guides for:

- Calibration card placement
- Newborn skin region
- Capture positioning

The application uses browser camera APIs and supports switching between available cameras.

---

## Color Calibration

Lighting conditions can significantly influence image-based color analysis.

NeoLume optionally uses a neutral-gray patch from a calibration card to perform **gain-based white-balance correction**.

The correction estimates the difference between:

- Observed calibration-card color
- Expected neutral reference color

and applies per-channel gains to the captured image.

This helps reduce variation caused by lighting and camera white balance.

---

## Explainable Machine Learning

The primary lightweight pipeline uses interpretable color statistics extracted from the image.

The system analyzes:

- LAB color distribution
- HSV color distribution
- Yellow-pixel proportion
- Red-to-blue relationship

This makes it possible to provide an explanation alongside the prediction.

---

## Class-Imbalance Handling

The dataset contains substantially more normal images than jaundice images.

The training pipeline addresses this using:

- `class_weight="balanced"`
- Minority-class image augmentation
- Group-aware cross-validation

The grouping mechanism prevents augmented versions of the same original image from appearing in both training and validation folds.

---

## Risk-Based Output

Instead of exposing only a binary prediction, NeoLume converts the model probability into three practical screening tiers:

| Risk Tier | Meaning |
|---|---|
| 🟢 `LOW` | Lower estimated screening risk |
| 🟡 `MONITOR` | Elevated signal requiring monitoring and clinical judgment |
| 🔴 `REFER_URGENTLY` | Higher-risk screening signal requiring prompt clinical evaluation |

These categories are intended for **screening**, not diagnosis.

---

## FastAPI Backend

The frontend communicates with a FastAPI backend through a REST API.

The backend accepts:

- Newborn skin image
- Optional calibration-card image

and returns structured information containing:

- Scan ID
- Timestamp
- Risk probability
- Risk tier
- Explanation
- Model status
- Calibration status

---

## Offline-Friendly Fallback

If the backend cannot be reached, the frontend performs a lightweight client-side color calculation.

This fallback is intentionally simple and should **not be considered equivalent to the trained ML model**.

Its purpose is to maintain the basic application workflow during:

- Temporary network failure
- Backend downtime
- Demonstration without the API
- Future offline-first development

---

# System Architecture

NeoLume consists of three logical layers.

### Frontend

Responsible for:

- Camera access
- Guided capture
- Region selection
- Image preparation
- API communication
- Result visualization
- Offline fallback

### Backend

Responsible for:

- Image decoding
- Calibration
- Feature extraction
- Model inference
- Risk classification
- Explanation generation

### Machine Learning Layer

The repository supports:

- Scikit-learn feature-based model
- Keras CNN inference path
- Rule-based fallback

The backend automatically determines which model is available.

---

# How It Works

## Step 1 — Capture

The user opens the NeoLume web application and grants camera access.

The interface displays guides for:

- Calibration card
- Skin/forehead region

The user captures an image when the required regions are positioned appropriately.

---

## Step 2 — Region Selection

After capture, the user can manually select:

1. The neutral calibration patch
2. The newborn skin region

The user can alternatively select **"Use whole photo"**, although this provides less controlled input.

---

## Step 3 — Calibration

If a calibration patch is provided, NeoLume calculates its average observed color.

The observed RGB/BGR values are compared against a predefined neutral reference.

The correction follows the general relationship:

```text
Gain = Reference Color / Observed Color
```

The gains are constrained to avoid excessive correction and are then applied to the image.

---

## Step 4 — Feature Extraction

The calibrated image is converted into:

- LAB
- HSV

color spaces.

Relevant statistics are calculated from the selected region.

These features form the numerical representation provided to the machine-learning model.

---

## Step 5 — Machine Learning Inference

The extracted feature vector is passed to the trained model.

The current lightweight model uses:

```text
StandardScaler
      ↓
RandomForestClassifier
```

The classifier produces a probability associated with the jaundice/at-risk class.

---

## Step 6 — Risk Stratification

The probability is converted into a screening category.

The current prototype uses:

```text
Probability < 0.35
        ↓
      LOW

0.35 ≤ Probability < 0.65
        ↓
    MONITOR

Probability ≥ 0.65
        ↓
 REFER_URGENTLY
```

These thresholds are **prototype thresholds** and should be clinically validated and tuned against appropriate ground-truth measurements before real-world use.

---

## Step 7 — Explanation

The system reports color-related information used by the feature pipeline.

Example:

```text
Yellow-tint index (LAB b*) = 148.2
Yellow-pixel share = 63%
```

This provides a more interpretable result than returning only a binary prediction.

---

# Technology Stack

## Frontend

- HTML5
- CSS3
- JavaScript
- Browser MediaDevices API
- Canvas API
- Fetch API

The frontend is intentionally lightweight and does not require a JavaScript framework.

## Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- OpenCV
- NumPy

## Machine Learning

- Scikit-learn
- Random Forest
- StandardScaler
- Joblib

The repository also contains a Keras-based CNN inference/training path.

## Training / Experimentation

- Jupyter Notebook
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- NumPy
- Matplotlib

---

# Machine Learning Pipeline

The primary lightweight model follows:

```text
Input Image
     ↓
Image Decode
     ↓
Optional White-Balance Calibration
     ↓
Skin/Region Masking
     ↓
LAB + HSV Conversion
     ↓
Color Feature Extraction
     ↓
StandardScaler
     ↓
Random Forest
     ↓
Jaundice Probability
     ↓
Risk Tier
```

The model operates on numerical color features rather than directly processing raw pixels.

This reduces:

- Computational requirements
- Model size
- Training time
- Hardware requirements

and makes the approach suitable for CPU-based inference and future edge deployment.

---

# Feature Engineering

NeoLume extracts a compact set of image features.

## LAB Features

| Feature | Description |
|---|---|
| `L_mean` | Mean luminance |
| `L_std` | Luminance variation |
| `L_median` | Median luminance |
| `A_mean` | Mean LAB A-channel |
| `A_std` | A-channel variation |
| `A_median` | Median A-channel |
| `B_mean` | Mean LAB B-channel |
| `B_std` | B-channel variation |
| `B_median` | Median B-channel |

The LAB B-channel represents the yellow-blue component of the image and is particularly relevant to the color-based analysis.

## HSV Features

| Feature | Description |
|---|---|
| `H_mean` | Mean hue |
| `H_std` | Hue variation |
| `S_mean` | Mean saturation |
| `S_std` | Saturation variation |
| `V_mean` | Mean brightness/value |
| `V_std` | Brightness variation |

## Additional Color Features

### Yellow Ratio

```text
yellow_ratio =
yellow-ish pixels / total selected pixels
```

A predefined HSV range is used to estimate the proportion of yellow-ish pixels.

### Red-to-Blue Ratio

The system also calculates:

```text
R_over_B_mean = mean(R / B)
```

This provides another representation of the relative red/blue composition of the selected region.

---

# Skin Region Masking

The feature extractor uses an HSV-based mask to reduce the influence of background pixels.

```text
Image
  ↓
HSV conversion
  ↓
Skin-tone range filtering
  ↓
Mask
  ↓
Color statistics
```

If the mask becomes nearly empty, the system falls back to the full image region to prevent inference failure.

---

# Model Training

The repository contains `backend/train_model.py` for training the lightweight feature-based model.

The expected dataset structure is:

```text
data/
├── jaundice/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── normal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The default model is:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)
```

The feature pipeline is:

```text
Image
 ↓
Feature Extraction
 ↓
StandardScaler
 ↓
Random Forest
```

The resulting model is stored as:

```text
backend/model/risk_model.joblib
```

Evaluation metrics are stored as:

```text
backend/model/metrics.json
```

---

# Class Imbalance Handling

The training data is imbalanced, with substantially more normal examples than jaundice examples.

NeoLume uses complementary strategies.

## 1. Balanced Class Weights

The Random Forest uses:

```python
class_weight="balanced"
```

This increases the importance of the minority class during training.

## 2. Minority-Class Augmentation

The training script generates additional variants for the minority class using lightweight transformations such as:

- Horizontal flipping
- Brightness increase
- Brightness decrease

## 3. Group-Aware Cross-Validation

If an original image and its augmented copy are placed into different validation folds, the model may effectively see nearly identical images during training and validation.

NeoLume addresses this using:

```text
StratifiedGroupKFold
```

All augmented versions originating from the same source image remain in the same fold.

---

# Model Evaluation

The currently stored evaluation results contain:

| Metric | Value |
|---|---:|
| Samples | 1,360 |
| Accuracy | 77.13% |
| Precision | 87.91% |
| Recall / Sensitivity | 70.88% |
| ROC-AUC | 86.19% |

### Confusion Matrix

```text
                 Predicted
              Normal  Jaundice
Actual Normal   482      78
Actual Jaundice 233     567
```

The model demonstrates useful discrimination between the two classes, with an ROC-AUC of approximately **0.862**.

However, the recall of approximately **70.9%** also highlights an important limitation: the current model can still miss some positive cases.

For a screening application, improving sensitivity/recall is therefore an important area for future development.

---

# Recall-First Thresholding

The training pipeline considers the needs of a screening application.

Instead of optimizing only for accuracy, the training script evaluates different probability thresholds and searches for a threshold capable of maintaining approximately **90% recall** on cross-validation predictions.

The currently generated report identifies:

```text
Suggested high-recall threshold: 0.30
```

This threshold is stored in `metrics.json`.

The application currently uses separate prototype tier boundaries, so the high-recall threshold should be explicitly incorporated into the final risk policy after proper validation.

---

# Alternative CNN Pipeline

The repository also contains a training notebook exploring a CNN-based approach using transfer learning.

The notebook includes a **MobileNetV3-Small** path.

The CNN workflow includes:

```text
Input Image
     ↓
Resize / Preprocessing
     ↓
Data Augmentation
     ↓
MobileNetV3-Small
     ↓
Fine-Tuning
     ↓
Risk Prediction
```

The notebook also explores:

- Training/validation curves
- Confusion matrix
- ROC curve
- Precision-recall curve
- Threshold tuning
- Grad-CAM
- Model comparison

This enables comparison between a lightweight engineered-feature model and a learned image representation.

---

# Explainability

NeoLume provides a human-readable explanation based on measurable image characteristics.

For the feature-based pipeline, the explanation can include:

- LAB B-channel value
- Yellow-pixel proportion
- Resulting risk tier

Example:

```text
Yellow-tint index (LAB b*) = 148.2,
yellow-pixel share = 63%.
Tier 'MONITOR' is a screening signal only.
```

The training notebook additionally explores **Grad-CAM** for the CNN pipeline to visualize image regions contributing to the model's prediction.

---

# Frontend Workflow

The browser application follows a three-step workflow.

## Step 1 — Guided Capture

```text
Open Camera
     ↓
Position Calibration Card
     +
Position Skin Region
     ↓
Capture
```

## Step 2 — Confirm Regions

```text
Captured Image
     ↓
Select Calibration Patch
     ↓
Select Skin Region
     ↓
Confirm
```

## Step 3 — Screening Result

```text
Image
 ↓
Analysis
 ↓
Risk Probability
 ↓
Risk Tier
 ↓
Explanation
```

The interface displays:

- Risk category
- Risk score
- Explanation
- Scan ID
- Calibration status
- Whether a trained model or fallback was used

---

# Backend API

The backend is implemented using FastAPI.

## Health Check

### `GET /health`

Returns backend and model status.

Example:

```json
{
  "status": "ok",
  "model_kind": "sklearn"
}
```

Possible model states include:

```text
sklearn
keras
rule_based
```

---

## Prediction

### `POST /predict`

Accepts multipart form data.

### Required

```text
skin_image
```

Supported formats:

- JPEG
- PNG

### Optional

```text
card_image
```

The calibration image should represent the neutral calibration patch from the same capture.

### Example Response

```json
{
  "scan_id": "a82f31c2",
  "timestamp": "2026-09-01T12:30:00",
  "risk_probability": 0.61,
  "risk_tier": "MONITOR",
  "explanation": "Yellow-tint index (LAB b*) = 148.2, yellow-pixel share = 63%. Tier 'MONITOR' is a screening signal only, not a diagnosis - confirm with a TSB blood test if MONITOR or REFER_URGENTLY.",
  "used_trained_model": true,
  "calibrated": true
}
```

---

# Model Loading Strategy

The backend automatically detects available models.

The current priority is:

```text
1. risk_model.joblib
        ↓
2. risk_model_keras.h5
        ↓
3. Rule-based fallback
```

## Feature-Based Model

If:

```text
backend/model/risk_model.joblib
```

exists, the Scikit-learn model is loaded.

## Keras Model

If the Scikit-learn model is unavailable but:

```text
backend/model/risk_model_keras.h5
```

exists, the backend attempts to load the Keras model.

TensorFlow is imported lazily only when the Keras model is required.

## Rule-Based Fallback

If neither trained model is available, the backend calculates a transparent color-based score using:

- LAB B-channel
- Yellow-pixel ratio

---

# Project Structure

```text
NeoLume/
│
├── backend/
│   │
│   ├── main.py
│   │   ├── FastAPI application
│   │   ├── /health endpoint
│   │   ├── /predict endpoint
│   │   ├── model loading
│   │   ├── risk classification
│   │   └── explanation generation
│   │
│   ├── feature_extraction.py
│   │   ├── image decoding
│   │   ├── white-balance calibration
│   │   ├── skin masking
│   │   ├── LAB/HSV conversion
│   │   └── feature extraction
│   │
│   ├── train_model.py
│   │   ├── dataset loading
│   │   ├── minority augmentation
│   │   ├── Random Forest training
│   │   ├── cross-validation
│   │   └── metric generation
│   │
│   ├── requirements.txt
│   ├── runtime.txt
│   │
│   ├── data_sample/
│   │   ├── jaundice/
│   │   └── ...
│   │
│   └── model/
│       ├── risk_model.joblib
│       ├── risk_model_keras.h5
│       └── metrics.json
│
├── frontend/
│   │
│   ├── index.html
│   │   ├── camera interface
│   │   ├── guided capture
│   │   ├── region selection
│   │   ├── API communication
│   │   └── result display
│   │
│   └── assets/
│       └── calibration-card.svg
│
├── neolume_training.ipynb
│
└── README.md
```

---

# Installation

## Prerequisites

Recommended:

- Python 3.10+
- Modern web browser
- Webcam or smartphone camera
- Git

For CNN training:

- Google Colab or a machine with GPU support is recommended.

---

# Backend Setup

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Backend

Start the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

# Running the Frontend

The frontend is a standalone HTML application.

Navigate to:

```bash
cd frontend
```

Start a simple local web server:

```bash
python -m http.server 5500
```

Open:

```text
http://localhost:5500
```

The application will request camera permission.

### Camera Requirements

Modern browsers generally require camera access through:

- `localhost`
- HTTPS

Camera access may be blocked when the page is served from a non-secure HTTP origin.

---

# Connecting a Smartphone

For testing with a smartphone on the same Wi-Fi network:

1. Start the backend on the development machine.
2. Start the frontend web server.
3. Find the computer's local network IP.
4. Open the frontend from the phone.
5. Configure the backend URL using the computer's local IP.

Example:

```text
http://192.168.1.23:8000
```

The phone and computer must be connected to the same network.

Browser security restrictions may still require HTTPS for camera access depending on how the page is hosted.

---

# Training the Model

The lightweight training script expects:

```text
data/
├── jaundice/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── normal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Run:

```bash
cd backend
python train_model.py --data_dir ./data --out model/risk_model.joblib
```

The training pipeline will:

1. Load images
2. Identify the minority class
3. Apply minority-class augmentation
4. Extract color features
5. Perform grouped stratified cross-validation
6. Generate predictions
7. Calculate evaluation metrics
8. Tune a high-recall threshold
9. Train the final model on the complete dataset
10. Save the trained model

Output:

```text
backend/model/risk_model.joblib
backend/model/metrics.json
```

---

# Training Options

### Disable augmentation

```bash
python train_model.py \
    --data_dir ./data \
    --out model/risk_model.joblib \
    --no_augment
```

### Change the number of folds

```bash
python train_model.py \
    --data_dir ./data \
    --folds 5
```

---

# Google Colab Training

The repository includes:

```text
neolume_training.ipynb
```

This notebook provides a more comprehensive experimentation environment than the lightweight local training script.

It can be run using Google Colab with a GPU runtime.

The notebook explores:

### Dataset Analysis

- Class distribution
- Sample image visualization
- Image dimensions
- Data inspection

### Preprocessing

- Image resizing
- Color calibration proxy
- Training augmentation

### CNN

- MobileNetV3-Small
- Transfer learning
- Fine-tuning
- Training curves
- Confusion matrix
- ROC analysis
- Precision-recall analysis
- Threshold tuning
- Grad-CAM

### Feature-Based Model

- LAB features
- HSV features
- Feature-based experimentation
- Model comparison

The goal is to evaluate different approaches and select a model based on metrics relevant to screening rather than simply choosing the most complex architecture.

---

# Offline / Fallback Operation

NeoLume has multiple fallback levels.

## Backend Fallback

If no trained model exists:

```text
Image
 ↓
Feature Extraction
 ↓
LAB / Yellow Analysis
 ↓
Rule-Based Probability
 ↓
Risk Tier
```

## Frontend Fallback

If the backend cannot be reached:

```text
Captured Image
 ↓
Browser Canvas
 ↓
Average RGB
 ↓
Yellow-ness Estimate
 ↓
Risk Tier
```

The frontend identifies this as an **offline fallback estimate** rather than a trained-model result.

This architecture provides a foundation for future edge deployment using technologies such as:

- TensorFlow Lite
- ONNX Runtime
- WebAssembly
- Native mobile inference

---

# Deployment Architecture

A simple deployment can be structured as:

```text
             Smartphone
                  │
                  ▼
          Web Application
                  │
                  │ HTTPS
                  ▼
             FastAPI API
                  │
         ┌────────┴────────┐
         ▼                 ▼
  Feature Pipeline     ML Model
         │                 │
         └────────┬────────┘
                  ▼
           Risk Assessment
                  │
                  ▼
             JSON Result
```

The frontend is designed so that the backend URL can be configured independently, allowing local or hosted backend deployments.

---

# Design Principles

## 1. Screening Before Diagnosis

The application is designed to identify **potential risk**, not replace clinical diagnosis.

## 2. Explainability

Color features and screening signals are exposed instead of providing only an unexplained prediction.

## 3. Lightweight Computation

The primary feature-based model avoids requiring GPU inference.

## 4. Calibration Awareness

The system recognizes that uncontrolled lighting can significantly affect image-based color analysis.

## 5. Recall-Oriented Evaluation

Because false negatives can be particularly important in screening scenarios, recall is considered alongside accuracy and precision.

## 6. Offline-First Direction

The frontend and backend architecture provide fallback mechanisms that can eventually evolve into a fully edge-based system.

---

# Limitations

NeoLume is currently a **research and prototyping system** and has important limitations.

### Dataset Limitations

The current model is trained on an image dataset and does not yet represent the full diversity of:

- Newborn populations
- Skin tones
- Cameras
- Lighting environments
- Clinical conditions

### Clinical Ground Truth

Image classification alone is not sufficient to establish bilirubin concentration.

Proper validation requires comparison against clinically established measurements such as TSB.

### Calibration

The current calibration mechanism uses a simplified neutral-patch white-balance correction.

It is not a complete colorimetric calibration system.

### Risk Thresholds

The current risk boundaries are prototype thresholds and require validation using appropriate clinical ground truth.

### Environmental Variation

Performance may change with:

- Indoor lighting
- Outdoor lighting
- Shadows
- Camera exposure
- Camera processing
- White balance
- Background colors

### Skin Segmentation

The current skin mask is based on broad HSV thresholds and is not a dedicated newborn skin-segmentation model.

### Offline Fallback

The browser-based RGB fallback is intentionally simple and should not be treated as equivalent to the trained model.

---

# Future Improvements

## Improved Image Standardization

- Better calibration cards
- Multi-point color calibration
- Exposure normalization
- Automatic blur detection
- Image-quality scoring

## Improved Machine Learning

- Larger and more diverse datasets
- External validation datasets
- CNN/Transformer comparison
- Ensemble models
- Better probability calibration
- Cost-sensitive learning
- Advanced segmentation

## True On-Device Inference

Convert the trained model to:

- TensorFlow Lite
- ONNX
- WebAssembly

This would reduce or eliminate backend dependency.

## Automatic Region Detection

Replace manual region selection with automatic detection of:

- Forehead
- Chest
- Sole
- Eye/sclera region

## Clinical Validation

Future work should include:

- Prospective clinical studies
- TSB-based ground truth
- Sensitivity/specificity analysis
- Calibration curves
- External validation
- Subgroup analysis
- Robustness testing

## Remote Monitoring

A future backend could support:

- Patient records
- Longitudinal monitoring
- Healthcare-worker dashboards
- Alerts
- Referral workflows
- Population-level analytics

---

# Development Status

| Component | Status |
|---|---|
| Guided camera capture | ✅ Implemented |
| Calibration-card workflow | ✅ Implemented |
| Manual region selection | ✅ Implemented |
| Color preprocessing | ✅ Implemented |
| LAB/HSV feature extraction | ✅ Implemented |
| Random Forest model | ✅ Implemented |
| Class-imbalance handling | ✅ Implemented |
| Group-aware cross-validation | ✅ Implemented |
| FastAPI backend | ✅ Implemented |
| Risk stratification | ✅ Implemented |
| Explainable output | ✅ Implemented |
| Browser fallback | ✅ Implemented |
| CNN experimentation | ✅ Included in notebook |
| Grad-CAM experimentation | ✅ Included in notebook |
| Clinical validation | 🔬 Future work |
| Fully edge-deployed ML | 🔬 Future work |
| Production medical deployment | 🔬 Future work |

---

# Getting Started — Quick Version

If you simply want to run the existing application:

### 1. Start Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Frontend

In another terminal:

```bash
cd frontend
python -m http.server 5500
```

### 3. Open

```text
http://localhost:5500
```

### 4. Use the Application

```text
Allow Camera
     ↓
Position Calibration Card + Skin
     ↓
Capture
     ↓
Select Calibration Patch
     ↓
Select Skin Region
     ↓
Analyze
     ↓
View Risk Result
```

---

# Project Vision

NeoLume explores how **accessible computer vision, lightweight machine learning, and smartphone-based imaging** can contribute to earlier health-risk screening in environments where conventional diagnostic resources may not always be immediately available.

The long-term goal is not to replace clinical testing, but to create a **portable first layer of screening and referral support** that can help identify cases requiring further medical attention.

> **Capture → Calibrate → Analyze → Screen → Refer**

---

# Clinical and Safety Disclaimer

**NeoLume is a prototype screening aid and is not a medical diagnostic device.**

A `LOW`, `MONITOR`, or `REFER_URGENTLY` result must not be interpreted as a definitive diagnosis or as a replacement for professional medical evaluation.

Image-based predictions can be affected by lighting, camera characteristics, skin tone, image quality, calibration, dataset limitations, and other factors.

A `MONITOR` or `REFER_URGENTLY` result should be evaluated by an appropriately qualified healthcare professional and confirmed using clinically appropriate methods, including bilirubin testing where indicated.

The current model and thresholds have **not been clinically validated for independent medical decision-making**.
