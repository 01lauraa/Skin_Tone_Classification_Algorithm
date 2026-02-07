# Skin Tone Classification (VALLES)

This repository contains Python scripts for **skin segmentation**, **skin tone estimation**, and **dataset-level analysis** using the Individual Typology Angle (ITA).  
The code was developed and tested on the **VALLES dataset**.

## Important notes
- The segmentation algorithm assumes **black backgrounds** or **already segmented images**
- It is not designed for images with complex or cluttered backgrounds
- Thresholds and heuristics are dataset-specific and may require tuning for other datasets

## Skin tone estimation details
- Skin tone is estimated using **ITA computed from true CIE L\*a\*b\*** values
- Background pixels are excluded using intensity-based masking
- **Highlights and shadows are removed** by discarding extreme L\* values:
  - Only pixels between the **5th and 98th percentiles of L\*** are retained
- Additional robustness is achieved using **IQR-based outlier filtering** before computing final ITA statistics

## Scripts
- `segment.py` – YCrCb-based skin segmentation with morphological cleanup  
- `skintoneclass.py` – ITA computation, tone classification, CSV export, and visualization  
- `skin_tone_stat.py` – Dataset statistics and tone-based image organization  

## Output
- CSV files with ITA statistics and skin tone labels  
- Optional visualizations grouped by skin tone  

## Author
Laura Gozzo
