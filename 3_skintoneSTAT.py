import pandas as pd
import os
import shutil
import pandas as pd
import os
import shutil
import pandas as pd

# sort images into folders based on skin tone labels from a CSV file and count skin tone distribution

csv_path = r"valles_skin.csv"

def sort_images_by_detector_tone(csv_path, image_folder, output_root):
    """
    Reads CSV and copies each image into a folder corresponding to skin tone.
    """
    df = pd.read_csv(csv_path)

    if "tone label" not in df.columns:
        raise ValueError("CSV must contain column 'tone label'")

    os.makedirs(output_root, exist_ok=True)

    tones = sorted(df["tone label"].unique())
    for tone in tones:
        os.makedirs(os.path.join(output_root, tone), exist_ok=True)

    missing = 0
    for _, row in df.iterrows():
        fname = row["file"]
        tone = row["tone label"]

        src = os.path.join(image_folder, fname)
        dst = os.path.join(output_root, tone, fname)

        if not os.path.exists(src):
            missing += 1
            continue

        shutil.copy(src, dst)

    print(f"Done! Images sorted into folders at: {output_root}")
    if missing > 0:
        print(f"Warning: {missing} images were missing in the image folder.")

def export_images_by_tone(csv_path, image_folder, output_folder, tone):
    """
    copies images of a specific skin tone from the dataset to a designated folder.
    """
    df = pd.read_csv(csv_path)
    if "tone" not in df.columns or "filename" not in df.columns:
        raise ValueError("CSV must contain 'filename' and 'tone' columns.")
    
    df_tone = df[df["tone"] == tone]
    if df_tone.empty:
        print(f"⚠️ No images found for tone: {tone}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    copied = 0
    
    for _, row in df_tone.iterrows():
        fname = row["filename"]
        src = os.path.join(image_folder, fname)
        
        if not os.path.exists(src):
            print(f"⚠️ Missing file: {src}")
            continue
        
        dst = os.path.join(output_folder, fname)
        shutil.copy(src, dst)
        copied += 1
    
    print(f"\n Copied {copied} images for tone '{tone}' to:")
    print(output_folder)

def count_images_per_tone(csv_path):
    """
    Reads a skin-tone CSV file and counts how many images belong to each tone category.
    """
    df = pd.read_csv(csv_path)

    if "tone" not in df.columns:
        raise ValueError("CSV must contain a 'tone' column")

    tone_counts = df["tone"].value_counts().sort_index()

    print("\nImages per Skin Tone:")
    for tone, count in tone_counts.items():
        print(f"{tone:>12} : {count}")

    return tone_counts

count_images_per_tone(csv_path)

image_folder = r"dataset\valles\segmented"
output_folder = r"outputs\toneClassification\dark"

# export_images_by_tone(csv_path, image_folder, output_folder, tone="dark")
