"""
NeoLume - Model Training
-------------------------
Trains a risk classifier on top of the colour-statistics features in
feature_extraction.py.

Expected dataset layout (this matches how most Kaggle "jaundice-image-data"
style dumps are organised — adjust DATA_DIR / class folder names if yours
differ):

    data/
      jaundice/       <-- images of jaundiced skin/eyes
      normal/          <-- images of normal skin/eyes

Usage:
    python train_model.py --data_dir ./data --out model/risk_model.joblib

Output:
    - model/risk_model.joblib      (trained sklearn pipeline)
    - model/metrics.json           (accuracy / precision / recall / AUC)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from feature_extraction import extract_features, FEATURE_NAMES

# --- imbalance handling -----------------------------------------------
# With ~200 jaundice vs ~1100 normal (typical Kaggle split), a classifier
# trained naively will lean toward predicting "normal" and hurt recall on
# the class that matters most. Two independent mitigations are applied:
#   1. class_weight="balanced" in the classifier (cheap, always on).
#   2. Image-level augmentation of the MINORITY class only, so it's
#      represented more densely in feature-space, not just reweighted.
# Both apply; either alone is a weaker fix.


def augment_image(img: np.ndarray) -> list[np.ndarray]:
    """A few cheap, label-preserving augmentations for the minority class."""
    out = [img]
    out.append(cv2.flip(img, 1))  # horizontal flip
    bright = cv2.convertScaleAbs(img, alpha=1.15, beta=10)   # brighter
    dim = cv2.convertScaleAbs(img, alpha=0.85, beta=-10)     # dimmer
    out.extend([bright, dim])
    return out

# Map your dataset's folder names -> label. 1 = jaundice/at-risk, 0 = normal.
CLASS_MAP = {
    "jaundice": 1,
    "normal": 0,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_dataset(data_dir: str, augment_minority: bool = True):
    X, y, paths = [], [], []
    data_dir = Path(data_dir)

    # find which label has fewer files so we know who to augment
    counts = {}
    for folder_name, label in CLASS_MAP.items():
        folder = data_dir / folder_name
        if folder.exists():
            counts[label] = sum(1 for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS)
    minority_label = min(counts, key=counts.get) if counts else None
    if counts:
        print(f"Class counts: {counts}  (minority = label {minority_label})")

    for folder_name, label in CLASS_MAP.items():
        folder = data_dir / folder_name
        if not folder.exists():
            print(f"[warn] folder not found, skipping: {folder}")
            continue
        for f in folder.rglob("*"):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(f))
            if img is None:
                continue

            variants = [img]
            if augment_minority and label == minority_label:
                variants = augment_image(img)

            for v in variants:
                feats = extract_features(v)
                X.append(feats)
                y.append(label)
                paths.append(str(f))
    return np.array(X), np.array(y, dtype=int), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="model/risk_model.joblib")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no_augment", action="store_true",
                         help="Disable minority-class augmentation.")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_dir} ...")
    X, y, paths = load_dataset(args.data_dir, augment_minority=not args.no_augment)
    print(f"Loaded {len(X)} samples (after any augmentation). "
          f"Class balance: {np.bincount(y)}")

    if len(X) < 20:
        raise SystemExit(
            "Not enough samples to train reliably. Point --data_dir at the "
            "extracted Kaggle dataset (jaundice/ and normal/ subfolders)."
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42,
        )),
    ])

    # Cross-validated predictions for honest metrics on small datasets.
    # IMPORTANT: augmented copies of the same source image are grouped so
    # they never split across train/val - otherwise the CV score is
    # inflated by the model "recognizing" near-duplicate augmented images
    # it already saw during training.
    groups = np.array(paths)
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    y_prob = cross_val_predict(pipeline, X, y, cv=sgkf, groups=groups, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Find the lowest threshold that still hits >=90% recall, as a
    # suggested REFER_URGENTLY cutoff - screening tools should err toward
    # over-referring rather than missing a case.
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thresh = 0.5
    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        r = recall_score(y, pred_t, zero_division=0)
        if r >= 0.90:
            best_thresh = t  # keep raising threshold while recall holds
        else:
            break

    metrics = {
        "n_samples": int(len(X)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "suggested_high_recall_threshold": float(best_thresh),
        "feature_names": FEATURE_NAMES,
    }
    print(json.dumps(metrics, indent=2))
    print(
        "\nNOTE: for a screening tool, RECALL (sensitivity) matters most — "
        "missing a jaundiced newborn is far worse than a false alarm. "
        f"Suggested threshold for >=90% recall: {best_thresh:.2f} "
        "(default decision cutoff is 0.5 - update probability_to_tier() "
        "in main.py's MONITOR/REFER_URGENTLY bounds using this)."
    )

    # Fit final model on all data for deployment
    pipeline.fit(X, y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    with open(out_path.parent / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {out_path}")
    print(f"Saved metrics -> {out_path.parent / 'metrics.json'}")


if __name__ == "__main__":
    main()
