from __future__ import annotations
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np


SUPPORTED_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.npy")


def load_image_bgr(path: str | Path) -> np.ndarray:
    """
    Load an image as BGR uint8.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def create_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Create a binary skin mask using adaptive thresholds in YCrCb space.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)

    cr_median = np.median(cr)
    cb_median = np.median(cb)

    lower = np.array(
        [0, max(0, cr_median - 20), max(0, cb_median - 20)],
        dtype=np.uint8,
    )
    upper = np.array(
        [255, min(255, cr_median + 20), min(255, cb_median + 20)],
        dtype=np.uint8,
    )

    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = np.zeros_like(mask)
    clean_mask[labels == largest_label] = 255
    return clean_mask


def apply_inverted_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply the inverted binary mask to the image.
    """
    inverted_mask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(image_bgr, image_bgr, mask=inverted_mask)


def find_image_paths(
    input_folder: str | Path,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
) -> list[Path]:
    """
    Collect supported image paths from a folder.
    """
    input_folder = Path(input_folder)
    paths: list[Path] = []

    for pattern in extensions:
        paths.extend(input_folder.glob(pattern))

    return sorted(paths)


def segment_single_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    Segment one image and return the segmented result.
    """
    mask = create_skin_mask(image_bgr)
    segmented = apply_inverted_mask(image_bgr, mask)
    return segmented


def segment_folder(input_folder: str | Path, output_folder: str | Path) -> None:
    """
    Segment skin regions for all images in a folder and save results.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = find_image_paths(input_folder)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_folder}")

    for image_path in image_paths:
        print(f"[INFO] Processing {image_path.stem} ...")

        image_bgr = load_image_bgr(image_path)
        segmented = segment_single_image(image_bgr)

        output_path = output_folder / f"{image_path.stem}_segmented.png"
        cv2.imwrite(str(output_path), segmented)

    print(f"[DONE] Segmented images saved to: {output_folder}")