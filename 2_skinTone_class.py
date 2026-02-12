import os, glob, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt

#Classify dataset of segmented skin images by ITA and skin tone and visualize results

segmented_folder = r"dataset/valles/segmented" 
original_folder  = r"dataset/valles_all"  
out_csv          = r"outputs/valles_skin.csv"
save_path        = r"outputs/valles_skin_visualization.png"

def bgr_to_true_lab(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L8, a8, b8 = cv2.split(lab)
    L_true = L8.astype(np.float32) * (100.0 / 255.0)
    a_true = a8.astype(np.float32) - 128.0
    b_true = b8.astype(np.float32) - 128.0
    return L_true, a_true, b_true

def compute_ita(L_true, b_true):
    return np.degrees(np.arctan2(L_true - 50.0, b_true + 1e-6))

def classify_ita(ita_val):
    if ita_val > 55:  return "very light"
    if ita_val > 41:  return "light"
    if ita_val > 28:  return "intermediate"
    if ita_val > 10:  return "tan"
    if ita_val > -30: return "brown"
    return "dark"

def estimate_skin_tone(bgr_img):
    L_true, a_true, b_true = bgr_to_true_lab(bgr_img)

    # Mask out black background 
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    roi = gray > 0

    # Remove shadows + glare using L* 5–98 percentile 
    L_roi = L_true[roi]
    if L_roi.size == 0:
        return dict(ita_median=np.nan, ita_iqr=np.nan, tone="unknown")

    low_thr, high_thr = np.percentile(L_roi, [5, 98])
    roi &= (L_true > low_thr) & (L_true < high_thr)

    # Compute ITA only on remaining pixels 
    ita = compute_ita(L_true, b_true)
    ita_vals = ita[roi]
    ita_vals = ita_vals[np.isfinite(ita_vals)]
    if ita_vals.size == 0:
        return dict(ita_median=np.nan, ita_iqr=np.nan, tone="unknown")

    # IQR OUTLIER FILTERING 
    q1, q3 = np.percentile(ita_vals, [25, 75])
    iqr = q3 - q1

    # Tukey range: remove everything far from central 50%
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    ita_vals_filtered = ita_vals[(ita_vals > lower) & (ita_vals < upper)]
    if ita_vals_filtered.size == 0:
        # fallback: use initial ITA if filtering removed too much
        ita_vals_filtered = ita_vals

   
    ita_med = float(np.median(ita_vals_filtered))
    q1_f, q3_f = np.percentile(ita_vals_filtered, [25, 75])
    ita_iqr = float(q3_f - q1_f)

    tone = classify_ita(ita_med)

    return dict(ita_median=ita_med, ita_iqr=ita_iqr, tone=tone)

def compute_ita_for_segmented_images(
    segmented_folder,
    out_csv,
    extensions=("*.png", "*.jpg", "*.jpeg"),
    verbose=True
):
    """
    Computes ITA statistics and skin-tone labels for all segmented images
    in a folder and saves the results to a CSV file.

    Args:
        segmented_folder (str): Path to folder with segmented skin images
        out_csv (str): Path where the output CSV will be saved
        extensions (tuple): Image extensions to search for
        verbose (bool): Whether to print progress messages

    Returns:
        pd.DataFrame: DataFrame with ITA statistics and tone labels
    """
    rows = []

    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(segmented_folder, ext)))

    if verbose:
        print(f"Found {len(files)} segmented skin images")

    for path in files:
        bgr = cv2.imread(path)
        if bgr is None:
            if verbose:
                print("Skipping unreadable:", path)
            continue

        res = estimate_skin_tone(bgr)
        res["filename"] = os.path.basename(path)
        rows.append(res)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    if verbose:
        print(f"ITA values saved to {out_csv}")

    return df

#df_ita = compute_ita_for_segmented_images(segmented_folder, out_csv)

df = pd.read_csv(out_csv)
def visualize_examples_by_tone(
    df,
    original_folder,
    save_path,
    n_cols=4,
    tone_order=("dark", "brown", "tan", "intermediate", "light", "very light"),
    random_state=42,
    fig_width=30,
    row_height=8,
    verbose=True,
):
    """
    Creates a grid visualization of  images grouped by skin-tone label and saves it.

    """
   
    df_vis = df.copy()

    # Drop invalid ITA rows
    if "ita_median" in df_vis.columns:
        df_vis = df_vis.dropna(subset=["ita_median"])

    if df_vis.empty:
        if verbose:
            print("No valid ITA entries to visualize.")
        return None

    if "tone" not in df_vis.columns or "filename" not in df_vis.columns:
        raise ValueError("df must contain 'tone' and 'filename' columns (and ideally 'ita_median').")

    present_tones = [t for t in tone_order if t in df_vis["tone"].unique()]
    if len(present_tones) == 0:
        if verbose:
            print("No tones from tone_order are present in df. Nothing to plot.")
        return None

    n_rows = len(present_tones)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, row_height * n_rows))
    plt.subplots_adjust(wspace=0.05, hspace=0.25, left=0.12, right=0.98, top=0.98, bottom=0.05)

    # Ensure axes is 2D even if one row
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, tone in enumerate(present_tones):
        subset = df_vis[df_vis["tone"] == tone]

        if subset.empty:
            for c_idx in range(n_cols):
                axes[r_idx, c_idx].axis("off")
            continue

        examples = subset.sample(min(n_cols, len(subset)), random_state=random_state)

        # Row label on the left
        axes[r_idx, 0].set_ylabel(
            tone,
            fontsize=30,
            rotation=0,
            labelpad=120,
            va="center",
            ha="right",
            fontweight="bold",
        )

        for c_idx, (_, row) in enumerate(examples.iterrows()):
            ax = axes[r_idx, c_idx]

            base_name = os.path.basename(str(row["filename"])).replace("_segmented", "")
            orig_candidates = [
                os.path.join(original_folder, base_name),
                os.path.join(original_folder, base_name.replace(".png", ".jpg")),
                os.path.join(original_folder, base_name.replace(".jpg", ".png")),
                os.path.join(original_folder, base_name.replace(".jpeg", ".jpg")),
            ]
            orig_path = next((p for p in orig_candidates if os.path.exists(p)), None)

            if not orig_path:
                ax.axis("off")
                continue

            img = cv2.imread(orig_path)
            if img is None:
                ax.axis("off")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.axis("off")

        # Hide any unused slots in the row
        n_imgs = min(n_cols, len(subset))
        for c_idx in range(n_imgs, n_cols):
            axes[r_idx, c_idx].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"📊 Visualization saved → {save_path}")

    return save_path

visualize_examples_by_tone(original_folder,save_path,df,n_cols=4)
