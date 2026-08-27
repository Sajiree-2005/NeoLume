# NeoLume — Portable Newborn Jaundice Screening

Working scaffold for the Omnikon submission (Dynamic Duo). This implements the
core of the pitch deck — Layers 1-3 (capture, on-device-style AI, risk tiering)
— as a browser demo + trainable Python model, scoped to what two people can
realistically ship in 1-2 days. Layers 4-5 (cloud dashboard, SMS/WhatsApp
alerts) are stubbed as "future work" in the deck and not built here — see
"What's cut" below.

## What's actually in this build

```
neolume/
  backend/
    feature_extraction.py   # calibration + colour-feature extraction (LAB/HSV stats)
    train_model.py          # trains RandomForest on Kaggle jaundice-image-data
    main.py                 # FastAPI: POST /predict -> risk tier + explanation
    requirements.txt
    model/                  # trained model + metrics.json land here
  frontend/
    index.html              # single-file web app: guided capture -> crop -> result
```

- **No native app** — a mobile-web page (works on any phone browser) instead
  of Android/Kotlin, since that's a multi-week effort for two people.
- **No cloud dashboard / SMS alerts (Layers 4-5)** — out of scope for the
  time budget. The API response is structured so a dashboard could consume
  it later without changes.
- **Calibration** is a simplified white-balance gain correction against a
  grey patch you crop from the calibration card, not full colour-checker
  matching — good enough to demo the concept and correct for the worst
  lighting error.
- **Model** is colour-statistics + RandomForest, not a CNN — trains in
  seconds on a laptop CPU with no GPU, and is honest about being explainable
  (the deck's "Explainability" box in Layer 2).

## Quickstart

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Without a trained model, `/predict` still works using a transparent
rule-based fallback (LAB yellow-tint threshold) — so the demo runs before
training finishes. Check `GET /health` to see `model_loaded: true/false`.

### 2. Train the real model (once you have the dataset)

**Recommended: use `NeoLume_Training_Colab.ipynb`** (in this folder) instead
of the plain `train_model.py` script if you want the full picture — proper
train/val/test splits, augmentation visualized, class-imbalance handling,
EDA plots, a CNN (MobileNetV3) *and* an XGBoost baseline trained side by
side, ROC/PR curves, threshold tuning for recall, Grad-CAM explainability,
and exports in every format the backend can consume. Upload it to
https://colab.research.google.com, set the runtime to a T4 GPU, and run
top to bottom — it downloads the Kaggle dataset itself if you supply your
`kaggle.json` API token when prompted. See the notebook's own markdown
cells for details; it's self-contained.

It ends by downloading `risk_model.joblib` (XGBoost — matches
`backend/train_model.py`'s output format) and/or `risk_model_keras.h5`
(CNN) — drop whichever wins on recall/F1/AUC into `backend/model/` and the
FastAPI backend auto-detects and loads it, no code changes needed.

**Or, the lighter/faster local path:** download the Kaggle set referenced
in the deck (https://www.kaggle.com/datasets/aiolapo/jaundice-image-data),
arrange it as:
```
backend/data/
  jaundice/   *.jpg
  normal/     *.jpg
```
(If the Kaggle folder names differ, edit `CLASS_MAP` at the top of
`train_model.py` — takes 30 seconds.)

Then:
```bash
cd backend
python train_model.py --data_dir ./data --out model/risk_model.joblib
```

This is the **primary model** — a RandomForest on hand-crafted colour
features (not a CNN). It's the right call for a ~200 jaundice / ~1100
normal split: needs far less data than a CNN, trains in seconds on a
laptop CPU, and stays explainable ("yellow-tint index was X") for judges.

**Handling the 200 vs 1100 imbalance:** the script already does two things
automatically — `class_weight="balanced"` in the classifier, and it
augments (flip/brighten/dim) only the minority (jaundice) class so it's
represented more densely, not just reweighted. Cross-validation groups
augmented copies of the same source photo together so scores aren't
inflated by near-duplicate leakage.

The printed report includes a **suggested high-recall threshold** — for a
screening tool, prioritize recall (sensitivity) over accuracy, since a
missed jaundiced newborn is worse than a false alarm. Plug that suggested
threshold into `probability_to_tier()` in `main.py`.

Restart the backend — it auto-loads `model/risk_model.joblib` if present.

### 2b. Full training notebook (recommended): `neolume_training.ipynb`

Run this on **Google Colab** (Runtime → Change runtime type → GPU) for the
complete, judge-ready training pipeline:

- EDA: class balance chart, sample image grid, image-size distribution
- Stratified train/val/test split
- Gray-world calibration proxy (since the public dataset has no physical
  calibration card) with before/after visualization
- `tf.data` pipeline with augmentation (flip/rotate/zoom/brightness/contrast)
  applied only to training data, visualized on a sample batch
- Class-imbalance handling for your ~200/1100 split via `class_weight`
- **CNN path:** MobileNetV3-Small transfer learning (frozen → fine-tune),
  training curves, confusion matrix, ROC curve, PR curve, a **recall-first
  threshold sweep** (screening tools should prioritize recall over
  accuracy), and **Grad-CAM explainability** visualizations
- **XGBoost path:** same hand-crafted LAB/HSV colour features as
  `feature_extraction.py`, trained and evaluated side-by-side with the CNN
  for a direct accuracy/precision/recall/F1/AUC-ROC comparison — useful to
  show judges you evaluated a trade-off, not just defaulted to "bigger
  model"
- Exports `risk_model_keras.h5`, `risk_model.tflite` (quantized), and
  `risk_model_xgb.joblib`, plus a `deployment_meta.json` with the tuned
  threshold and class order

Open `neolume_training.ipynb` directly in Colab (File → Upload notebook, or
drag it into colab.research.google.com), fix `CLASS_DIRS` in the config
cell if your Kaggle folder names differ from `jaundice` / `normal`, and run
all cells top to bottom.

Drop whichever exported model you prefer (`risk_model_keras.h5` or
`risk_model_xgb.joblib`) into `backend/model/` — `main.py` auto-detects and
loads it, in priority order: `risk_model.joblib` (sklearn, matches what
`train_model.py` locally produces) → `risk_model_keras.h5` (Keras CNN) →
rule-based fallback. `tensorflow` is only imported if the Keras file is
present, so skipping the notebook entirely still works fine.

### 3. Frontend

Just open `frontend/index.html` in a phone or laptop browser (Chrome/Safari
need camera permission; serve over `https://` or `localhost` — camera APIs
are blocked on plain `http://` for non-localhost hosts). Simplest way to
serve it locally:

```bash
cd frontend
python3 -m http.server 5500
```
Then visit `http://localhost:5500` (or your machine's LAN IP from a phone
on the same wifi, e.g. `http://192.168.1.23:5500`).

Set the "Backend URL" field at the top to wherever `uvicorn` is running
(e.g. `http://192.168.1.23:8000` if testing from a phone on the same wifi).

If the backend is unreachable, the frontend **still works** using an
on-device JS fallback (crude RGB yellow-ness estimate) — this demonstrates
the "fully offline" pillar from the deck even without a real edge model
compiled in.

## How this maps to the deck

| Deck layer | This build |
|---|---|
| Layer 1 — Capture & Input | `index.html` guided capture UI with card + skin guides |
| Layer 2 — On-device AI | `feature_extraction.py` (calibration, LAB/HSV features) + `train_model.py` (RandomForest risk model) — logic is portable to TFLite/ONNX later |
| Layer 3 — Risk Assessment | `main.py` `/predict`: probability → LOW / MONITOR / REFER_URGENTLY |
| Layer 4 — Cloud backend/DB | Not built (FastAPI could be extended with PostgreSQL later — response schema is already structured for it) |
| Layer 5 — Dashboard/alerts | Not built (future work) |

