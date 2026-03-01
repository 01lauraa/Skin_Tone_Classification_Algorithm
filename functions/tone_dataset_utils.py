from __future__ import annotations
from pathlib import Path
import shutil
import pandas as pd


def export_images_by_tone(
    csv_path: str | Path,
    image_folder: str | Path,
    output_folder: str | Path,
    tone: str,
) -> None:
    """
    Copy images of a specific skin tone into a designated folder.
    """
    csv_path = Path(csv_path)
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)

    df = pd.read_csv(csv_path)

    if "tone" not in df.columns or "filename" not in df.columns:
        raise ValueError("CSV must contain 'filename' and 'tone' columns.")

    df_tone = df[df["tone"] == tone]
    if df_tone.empty:
        print(f"No images found for tone: {tone}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in df_tone.iterrows():
        filename = row["filename"]
        src = image_folder / filename

        if not src.exists():
            print(f"Missing file: {src}")
            continue

        dst = output_folder / filename
        shutil.copy(src, dst)
        copied += 1

    print(f"Copied {copied} images for tone '{tone}' to:")
    print(output_folder)


def sort_images_by_tone(
    csv_path: str | Path,
    image_folder: str | Path,
    output_root: str | Path,
) -> None:
    """
    Copy each image into a folder corresponding to its skin-tone label.
    """
    csv_path = Path(csv_path)
    image_folder = Path(image_folder)
    output_root = Path(output_root)

    df = pd.read_csv(csv_path)

    if "tone" not in df.columns or "filename" not in df.columns:
        raise ValueError("CSV must contain 'filename' and 'tone' columns.")

    output_root.mkdir(parents=True, exist_ok=True)

    tones = sorted(df["tone"].dropna().unique())
    for tone in tones:
        (output_root / tone).mkdir(parents=True, exist_ok=True)

    missing = 0
    copied = 0

    for _, row in df.iterrows():
        filename = row["filename"]
        tone = row["tone"]

        src = image_folder / filename
        dst = output_root / tone / filename

        if not src.exists():
            missing += 1
            continue

        shutil.copy(src, dst)
        copied += 1

    print(f"Done! Copied {copied} images into folders at: {output_root}")
    if missing > 0:
        print(f"Warning: {missing} images were missing from the image folder.")


def count_images_per_tone(csv_path: str | Path) -> pd.Series:
    """
    Count how many images belong to each tone category.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "tone" not in df.columns:
        raise ValueError("CSV must contain a 'tone' column.")

    tone_counts = df["tone"].value_counts().sort_index()

    print("\nImages per skin tone:")
    for tone, count in tone_counts.items():
        print(f"{tone:>12} : {count}")

    return tone_counts