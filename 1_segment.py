import cv2
import numpy as np
import os
import glob

#segment skin regions from images in a folder (adapted for VALLES dataset) and save the results

def load_to_bgr_uint8(path):
    """
    Converts image to BGR uint8 format.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {path}")
    return bgr

def create_skin_mask(bgr):
    #convert to YCrCb color space
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # compute median Cr and Cb values to define skin tone range
    Cr_med, Cb_med = np.median(Cr), np.median(Cb)
    #print(f"[DEBUG] Median Cr: {Cr_med:.2f}, Median Cb: {Cb_med:.2f}")
    lower = np.array([0, max(0, Cr_med - 20), max(0, Cb_med - 20)], np.uint8)
    upper = np.array([255, min(255, Cr_med + 20), min(255, Cb_med + 20)], np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    # Morphological operations to clean up the mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(mask)
        clean[labels == largest] = 255
    else:
        clean = mask

    return clean

def apply_mask_always_invert(img, mask):
    mask_inv = cv2.bitwise_not(mask)
    segmented = cv2.bitwise_and(img, img, mask=mask_inv)
    return segmented

def segment_folder(input_folder, output_folder="segmented_results"):
    os.makedirs(output_folder, exist_ok=True)
    paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.npy"):
        paths.extend(glob.glob(os.path.join(input_folder, ext)))
    if not paths:
        raise FileNotFoundError(f"No supported images found in {input_folder}")

    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"[INFO] Processing {base} ...")
        img = load_to_bgr_uint8(path)
        mask = create_skin_mask(img)
        segmented = apply_mask_always_invert(img, mask)
        cv2.imwrite(os.path.join(output_folder, f"{base}_segmented.png"), segmented)

    print(f"[DONE] Segmented images saved to '{output_folder}'.")
    
if __name__ == "__main__":
    segment_folder(
        input_folder=r"dataset\valles\raw\Normal\ex",
        output_folder=r"dataset\valles\ex_seg"
    )
