from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")
TONE_ORDER = ("dark", "brown", "tan", "intermediate", "light", "very light")


def bgr_to_true_lab(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a BGR image to interpretable LAB channels.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_8bit, a_8bit, b_8bit = cv2.split(lab)

    l_true = l_8bit.astype(np.float32) * (100.0 / 255.0)
    a_true = a_8bit.astype(np.float32) - 128.0
    b_true = b_8bit.astype(np.float32) - 128.0

    return l_true, a_true, b_true


def compute_ita(l_true: np.ndarray, b_true: np.ndarray) -> np.ndarray:
    """
    Compute the Individual Typology Angle (ITA).
    """
    return np.degrees(np.arctan2(l_true - 50.0, b_true + 1e-6))


def classify_ita(ita_value: float) -> str:
    """
    Map ITA value to a skin-tone category.
    """
    if ita_value > 55:
        return "very light"
    if ita_value > 41:
        return "light"
    if ita_value > 28:
        return "intermediate"
    if ita_value > 10:
        return "tan"
    if ita_value > -30:
        return "brown"
    return "dark"


def estimate_skin_tone(image_bgr: np.ndarray) -> dict[str, float | str]:
    """
    Estimate skin tone from a segmented image.

    Steps:
    - Exclude black background
    - Remove extreme shadows and glare using L* percentiles
    - Compute ITA on remaining pixels
    - Filter ITA outliers with Tukey IQR rule
    """
    l_true, _, b_true = bgr_to_true_lab(image_bgr)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray > 0

    l_roi = l_true[roi]
    if l_roi.size == 0:
        return {"ita_median": np.nan, "ita_iqr": np.nan, "tone": "unknown"}

    low_thr, high_thr = np.percentile(l_roi, [5, 98])
    roi &= (l_true > low_thr) & (l_true < high_thr)

    ita = compute_ita(l_true, b_true)
    ita_values = ita[roi]
    ita_values = ita_values[np.isfinite(ita_values)]

    if ita_values.size == 0:
        return {"ita_median": np.nan, "ita_iqr": np.nan, "tone": "unknown"}

    q1, q3 = np.percentile(ita_values, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    ita_filtered = ita_values[(ita_values > lower_bound) & (ita_values < upper_bound)]
    if ita_filtered.size == 0:
        ita_filtered = ita_values

    ita_median = float(np.median(ita_filtered))
    q1_f, q3_f = np.percentile(ita_filtered, [25, 75])
    ita_iqr = float(q3_f - q1_f)
    tone = classify_ita(ita_median)

    return {
        "ita_median": ita_median,
        "ita_iqr": ita_iqr,
        "tone": tone,
    }


def find_image_paths(
    folder: str | Path,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
) -> list[Path]:
    """
    Collect supported image paths from a folder.
    """
    folder = Path(folder)
    paths: list[Path] = []

    for pattern in extensions:
        paths.extend(folder.glob(pattern))

    return sorted(paths)


def compute_ita_for_segmented_images(
    segmented_folder: str | Path,
    out_csv: str | Path,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute ITA statistics and skin-tone labels for all segmented images in a folder.
    """
    segmented_folder = Path(segmented_folder)
    out_csv = Path(out_csv)

    rows: list[dict[str, float | str]] = []
    image_paths = find_image_paths(segmented_folder, extensions)

    if verbose:
        print(f"Found {len(image_paths)} segmented skin images.")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            if verbose:
                print(f"Skipping unreadable file: {image_path}")
            continue

        result = estimate_skin_tone(image_bgr)
        result["filename"] = image_path.name
        rows.append(result)

    df = pd.DataFrame(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if verbose:
        print(f"ITA values saved to: {out_csv}")

    return df


def load_skin_tone_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV containing skin-tone results.
    """
    return pd.read_csv(csv_path)