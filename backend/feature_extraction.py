"""
NeoLume - Feature Extraction & Colour Calibration
--------------------------------------------------
Turns a raw RGB image (skin crop, optionally with a calibration-card patch)
into a fixed-length numeric feature vector that the risk model consumes.

Pipeline:
  1. (optional) White-balance / colour-calibrate using a reference patch
     cropped from a printed calibration card in the same shot.
  2. Convert to LAB + HSV colour spaces (LAB is closer to how bilirubin's
     yellow tint shows up than raw RGB).
  3. Compute summary statistics (mean/median/std per channel) -> feature vector.

This intentionally avoids deep-learning image models so it trains fast on a
small/medium dataset and runs instantly on-device later (TFLite / ONNX
conversion of a tree model, or just re-implementing these stats in Kotlin/JS).
"""

import numpy as np
import cv2

FEATURE_NAMES = [
    "L_mean", "L_std", "L_median",
    "A_mean", "A_std", "A_median",
    "B_mean", "B_std", "B_median",
    "H_mean", "H_std",
    "S_mean", "S_std",
    "V_mean", "V_std",
    "yellow_ratio",       # fraction of pixels in a "yellow-ish" HSV band
    "R_over_B_mean",      # simple redness/blueness ratio, drops with jaundice
]


def _read_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Could not decode image")
    return img


def calibrate_white_balance(img_bgr: np.ndarray, card_patch_bgr: np.ndarray,
                             reference_rgb=(200, 200, 200)) -> np.ndarray:
    """
    Simple gain-based white balance correction.

    card_patch_bgr: a crop of the neutral-grey patch on the calibration card,
                     taken from the SAME photo as img_bgr.
    reference_rgb:  the ground-truth colour that patch is printed as
                     (measure once with a colorimeter / use manufacturer spec).

    Returns a corrected copy of img_bgr.
    """
    observed = card_patch_bgr.reshape(-1, 3).mean(axis=0)  # B, G, R
    ref_bgr = np.array([reference_rgb[2], reference_rgb[1], reference_rgb[0]], dtype=np.float32)
    gains = ref_bgr / np.clip(observed, 1, 255)
    gains = np.clip(gains, 0.3, 3.0)  # avoid wild overcorrection

    corrected = img_bgr.astype(np.float32) * gains
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Loose HSV skin-tone mask to exclude background pixels from stats."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 15, 40], dtype=np.uint8)
    upper = np.array([50, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # fall back to whole image if mask is nearly empty (e.g. sclera crops)
    if mask.mean() < 5:
        mask = np.full(img_bgr.shape[:2], 255, dtype=np.uint8)
    return mask


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    mask = skin_mask(img_bgr)
    mask_bool = mask > 0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    L, A, B = lab[..., 0][mask_bool], lab[..., 1][mask_bool], lab[..., 2][mask_bool]
    H, S, V = hsv[..., 0][mask_bool], hsv[..., 1][mask_bool], hsv[..., 2][mask_bool]

    if L.size == 0:  # degenerate mask, use full image
        L, A, B = lab[..., 0].ravel(), lab[..., 1].ravel(), lab[..., 2].ravel()
        H, S, V = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()

    b_img = img_bgr[..., 0][mask_bool].astype(np.float32)
    r_img = img_bgr[..., 2][mask_bool].astype(np.float32)
    r_over_b = float(np.mean(r_img / np.clip(b_img, 1, 255)))

    # "yellow-ish" band in HSV (hue roughly 20-45 on OpenCV's 0-179 scale)
    yellow_pixels = np.logical_and(H >= 15, H <= 45)
    yellow_ratio = float(np.mean(yellow_pixels)) if H.size else 0.0

    feats = [
        float(np.mean(L)), float(np.std(L)), float(np.median(L)),
        float(np.mean(A)), float(np.std(A)), float(np.median(A)),
        float(np.mean(B)), float(np.std(B)), float(np.median(B)),
        float(np.mean(H)), float(np.std(H)),
        float(np.mean(S)), float(np.std(S)),
        float(np.mean(V)), float(np.std(V)),
        yellow_ratio,
        r_over_b,
    ]
    return np.array(feats, dtype=np.float32)


def features_from_bytes(image_bytes: bytes, card_patch_bytes: bytes | None = None) -> np.ndarray:
    img = _read_image(image_bytes)
    if card_patch_bytes is not None:
        card = _read_image(card_patch_bytes)
        img = calibrate_white_balance(img, card)
    return extract_features(img)
