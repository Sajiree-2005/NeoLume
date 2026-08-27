"""
NeoLume - Backend API
-----------------------
Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict
        multipart/form-data:
          - skin_image: file (required)   - crop of baby's skin (or eye/sclera)
          - card_image: file (optional)    - crop of the calibration card's
                                              neutral patch from the SAME photo
        returns JSON risk result.

    GET /health
        basic liveness check.

If no trained model is found at model/risk_model.joblib, falls back to a
transparent rule-based score (based on LAB b*-channel / yellow ratio) so the
demo still works end-to-end before training data is ready. Swap in the
trained model at any time — no API contract changes.
"""

from pathlib import Path
from datetime import datetime
import uuid

import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import cv2
from feature_extraction import features_from_bytes, FEATURE_NAMES, calibrate_white_balance, _read_image

MODEL_DIR = Path(__file__).parent / "model"
SKLEARN_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
KERAS_MODEL_PATH = MODEL_DIR / "risk_model_keras.h5"
# If you trained with colab_train_keras.py, check keras_metrics.json's
# "class_order" - it tells you whether index 0 or 1 means "jaundice".
# Default assumes alphabetical ['jaundice', 'normal'] -> label 0 = jaundice,
# so probability of the AT-RISK class is (1 - sigmoid_output). Flip this if
# your class_order differs.
KERAS_JAUNDICE_IS_LABEL_ZERO = True
KERAS_IMG_SIZE = (224, 224)

app = FastAPI(title="NeoLume Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model priority: trained sklearn (feature-based) > trained Keras CNN >
# transparent rule-based fallback. sklearn is preferred by default since
# it's the primary, better-suited path for this dataset size - Keras is
# opt-in by simply not deleting the sklearn model file.
_sklearn_model = None
_keras_model = None
MODEL_KIND = "rule_based"

if SKLEARN_MODEL_PATH.exists():
    _sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
    MODEL_KIND = "sklearn"
    print(f"Loaded trained sklearn model from {SKLEARN_MODEL_PATH}")
elif KERAS_MODEL_PATH.exists():
    try:
        import tensorflow as tf  # lazy import - only needed for this path
        _keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        MODEL_KIND = "keras"
        print(f"Loaded trained Keras model from {KERAS_MODEL_PATH}")
    except ImportError:
        print(f"Found {KERAS_MODEL_PATH} but tensorflow isn't installed. "
              f"Run: pip install tensorflow  -- falling back to rule-based scoring.")
else:
    print("No trained model found - using rule-based fallback scoring. "
          "Run train_model.py (or colab_train_keras.py) once you have labeled data.")


class RiskResult(BaseModel):
    scan_id: str
    timestamp: str
    risk_probability: float
    risk_tier: str          # LOW | MONITOR | REFER_URGENTLY
    explanation: str
    used_trained_model: bool
    calibrated: bool


def probability_to_tier(p: float) -> str:
    # Thresholds are placeholders - tune against TSB ground truth in the
    # clinical pilot. Biased toward high sensitivity (few missed high-risk
    # cases), per the "screening not diagnosis" design goal.
    if p < 0.35:
        return "LOW"
    elif p < 0.65:
        return "MONITOR"
    else:
        return "REFER_URGENTLY"


def rule_based_probability(feats: np.ndarray) -> float:
    """
    Transparent fallback: higher B* (LAB yellow-blue) and higher yellow_ratio
    correlate with more yellow/jaundiced tissue. Min-max squashed into 0-1
    using rough empirical bounds. Replace with the trained model ASAP -
    this is only here so the app is demoable before you finish training.
    """
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    b_mean = feats[idx["B_mean"]]
    yellow_ratio = feats[idx["yellow_ratio"]]

    # OpenCV's 8-bit LAB encodes b* around a ~128 midpoint (not 0), with
    # higher values = more yellow. ~135-165 is a rough working range for
    # skin tones; RETUNE this against real photos before relying on it -
    # it exists only so the app is demoable before the trained model lands.
    b_score = np.clip((b_mean - 135) / (165 - 135), 0, 1)
    y_score = np.clip(yellow_ratio, 0, 1)
    score = 0.6 * b_score + 0.4 * y_score
    return float(np.clip(score, 0, 1))


def build_explanation(feats: np.ndarray, tier: str) -> str:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    b_mean = feats[idx["B_mean"]]
    yellow_ratio = feats[idx["yellow_ratio"]]
    return (
        f"Yellow-tint index (LAB b*) = {b_mean:.1f}, "
        f"yellow-pixel share = {yellow_ratio*100:.0f}%. "
        f"Tier '{tier}' is a screening signal only, not a diagnosis - "
        f"confirm with a TSB blood test if MONITOR or REFER_URGENTLY."
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_kind": MODEL_KIND}


def keras_probability(skin_bytes: bytes, card_bytes: bytes | None) -> float:
    """Preprocess like colab_train_keras.py expects and run inference."""
    img = _read_image(skin_bytes)
    if card_bytes is not None:
        card = _read_image(card_bytes)
        img = calibrate_white_balance(img, card)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, KERAS_IMG_SIZE)
    batch = img[np.newaxis, ...].astype("float32")  # MobileNetV3 layer handles its own preprocessing
    raw = float(_keras_model.predict(batch, verbose=0)[0, 0])  # P(label==1)
    return (1.0 - raw) if KERAS_JAUNDICE_IS_LABEL_ZERO else raw


@app.post("/predict", response_model=RiskResult)
async def predict(skin_image: UploadFile = File(...),
                   card_image: UploadFile | None = File(None)):
    if skin_image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "skin_image must be a JPEG or PNG")

    skin_bytes = await skin_image.read()
    card_bytes = await card_image.read() if card_image is not None else None

    try:
        # feature extraction is always computed - used for the sklearn
        # model AND for the human-readable explanation text either way.
        feats = features_from_bytes(skin_bytes, card_bytes)
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")

    if MODEL_KIND == "sklearn":
        prob = float(_sklearn_model.predict_proba(feats.reshape(1, -1))[0, 1])
        used_trained = True
    elif MODEL_KIND == "keras":
        try:
            prob = keras_probability(skin_bytes, card_bytes)
        except Exception as e:
            raise HTTPException(500, f"Keras inference failed: {e}")
        used_trained = True
    else:
        prob = rule_based_probability(feats)
        used_trained = False

    tier = probability_to_tier(prob)
    explanation = build_explanation(feats, tier)

    return RiskResult(
        scan_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow().isoformat(),
        risk_probability=round(prob, 3),
        risk_tier=tier,
        explanation=explanation,
        used_trained_model=used_trained,
        calibrated=card_bytes is not None,
    )
