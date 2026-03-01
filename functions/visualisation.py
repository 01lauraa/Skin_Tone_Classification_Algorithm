from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TONE_ORDER = ("dark", "brown", "tan", "intermediate", "light", "very light")


def resolve_original_image_path(filename: str, original_folder: str | Path) -> Path | None:
    """
    Try to resolve the original image path corresponding to a segmented filename.
    """
    original_folder = Path(original_folder)
    base_name = Path(filename).name.replace("_segmented", "")

    candidates = [
        original_folder / base_name,
        original_folder / base_name.replace(".png", ".jpg"),
        original_folder / base_name.replace(".jpg", ".png"),
        original_folder / base_name.replace(".jpeg", ".jpg"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def visualize_examples_by_tone(
    df: pd.DataFrame,
    original_folder: str | Path,
    save_path: str | Path,
    n_cols: int = 4,
    tone_order: tuple[str, ...] = TONE_ORDER,
    random_state: int = 42,
    fig_width: int = 30,
    row_height: int = 8,
    verbose: bool = True,
) -> Path | None:
    """
    Create a grid visualization of original images grouped by skin-tone label.
    """
    original_folder = Path(original_folder)
    save_path = Path(save_path)

    if "tone" not in df.columns or "filename" not in df.columns:
        raise ValueError("DataFrame must contain 'tone' and 'filename' columns.")

    df_vis = df.copy()

    if "ita_median" in df_vis.columns:
        df_vis = df_vis.dropna(subset=["ita_median"])

    if df_vis.empty:
        if verbose:
            print("No valid ITA entries to visualize.")
        return None

    present_tones = [tone for tone in tone_order if tone in df_vis["tone"].unique()]
    if not present_tones:
        if verbose:
            print("No tones from tone_order are present in the DataFrame.")
        return None

    n_rows = len(present_tones)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, row_height * n_rows))
    plt.subplots_adjust(wspace=0.05, hspace=0.25, left=0.12, right=0.98, top=0.98, bottom=0.05)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, tone in enumerate(present_tones):
        subset = df_vis[df_vis["tone"] == tone]

        axes[row_idx, 0].set_ylabel(
            tone,
            fontsize=30,
            rotation=0,
            labelpad=120,
            va="center",
            ha="right",
            fontweight="bold",
        )

        if subset.empty:
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].axis("off")
            continue

        examples = subset.sample(min(n_cols, len(subset)), random_state=random_state)

        for col_idx, (_, row) in enumerate(examples.iterrows()):
            ax = axes[row_idx, col_idx]
            original_path = resolve_original_image_path(str(row["filename"]), original_folder)

            if original_path is None:
                ax.axis("off")
                continue

            image_bgr = cv2.imread(str(original_path))
            if image_bgr is None:
                ax.axis("off")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(image_rgb)
            ax.axis("off")

        for col_idx in range(len(examples), n_cols):
            axes[row_idx, col_idx].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"Visualization saved to: {save_path}")

    return save_path