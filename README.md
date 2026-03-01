# Skin Tone Classification 
This repository contains a modular pipeline for **skin segmentation**, **skin tone estimation**, **visualization**, and **dataset organization**.
The current implementation was used and tested on the Coral et al. (2025) dataset. 

<img src="assets/Screenshot 2026-02-07 192527.png" width="300">

## Overview

The pipeline consists of four main steps:

1. **Skin segmentation**
   Segment the skin region from each image using adaptive thresholding in **YCrCb** color space.

2. **Skin tone estimation**
   Estimate skin tone from segmented images using the **Individual Typology Angle (ITA)** computed in **CIELAB** space.

3. **Visualization**
   Create a figure showing example original images grouped by predicted skin tone category.

4. **Dataset organization utilities**
   Count images per tone category, export subsets of a chosen tone, or sort images into folders by tone.

---


### Module files

* **`skin_segmentation.py`**
  Contains reusable functions for loading images, creating a skin mask, and segmenting a folder of images.

* **`tone_estimation.py`**
  Contains reusable functions for converting images to LAB, computing ITA, classifying skin tone, and exporting results to CSV.

* **`visualization.py`**
  Contains reusable functions to visualize example images grouped by predicted skin tone.

* **`tone_dataset_utils.py`**
  Contains helper functions to count images per skin-tone class, export images of a given tone, and sort images into tone-specific folders.

### Run scripts

* **`run_segment_skin.py`**
  Runs the skin segmentation step on a folder of raw images.

* **`run_estimate_skin_tone.py`**
  Computes ITA statistics and skin-tone labels for segmented images and saves the results to CSV.

* **`run_visualize_skin_tones.py`**
  Generates a visualization of example original images grouped by predicted tone.

* **`run_export_tone_subset.py`**
  Copies images belonging to one selected tone category into a new folder.

* **`run_count_tones.py`**
  Prints the number of images in each tone category.

---

## Method

### 1. Skin segmentation

Skin regions are segmented using **adaptive thresholds in YCrCb color space**:

* the input image is converted from BGR to YCrCb
* the median **Cr** and **Cb** values are computed
* a threshold range is defined around those medians
* morphological opening and closing are applied to clean the mask
* only the **largest connected component** is kept

The mask is then inverted and applied to the image to obtain the segmented result.

### 2. Skin tone estimation

Skin tone is estimated from the segmented images using **ITA (Individual Typology Angle)**.

The process is:

* convert the image from BGR to **CIELAB**

* remove the black background from the segmented image

* exclude extreme shadows and highlights using **L*** percentiles

* compute **ITA** values from **L*** and **b***

* apply **IQR-based outlier filtering**

* classify the median ITA into one of the following tone categories:

* **very light**

* **light**

* **intermediate**

* **tan**

* **brown**

* **dark**

### 3. Visualization

The visualization step groups images by predicted tone category and saves a figure containing example original images for each class.

### 4. Dataset utilities

Additional utilities are included to:

* count how many images belong to each tone category
* export all images of a chosen tone
* sort images into separate folders by predicted tone

---

## Input and output

### Input

Depending on the stage, the pipeline expects:

* a folder of **raw images** for segmentation
* a folder of **segmented images** for ITA computation
* a CSV file containing at least:

  * `filename`
  * `tone`

### Output

The pipeline produces:

* segmented images saved as `.png`
* a CSV file with:

  * `filename`
  * `ita_median`
  * `ita_iqr`
  * `tone`
* a visualization figure showing tone-grouped examples
* optional folders containing images grouped or filtered by tone

---

## Example usage

### 1. Segment skin regions

```bash
python run_segment_skin.py
```

### 2. Compute ITA and skin-tone labels

```bash
python run_estimate_skin_tone.py
```

### 3. Visualize examples by tone

```bash
python run_visualize_skin_tones.py
```

### 4. Export a subset of one tone

```bash
python run_export_tone_subset.py
```

### 5. Count images per tone

```bash
python run_count_tones.py
```

## Notes

* The pipeline assumes that segmented images use a **black background**, which is removed before ITA computation.
* The CSV produced by the estimation step uses the column names:

  * `filename`
  * `tone`

## Dependencies

This project uses:

* `python`
* `opencv-python`
* `numpy`
* `pandas`
* `matplotlib`

Install them with:

```bash
pip install opencv-python numpy pandas matplotlib
```

