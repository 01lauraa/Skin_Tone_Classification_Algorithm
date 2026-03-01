import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from functions.tone_estimation import compute_ita_for_segmented_images

def main() -> None:
    compute_ita_for_segmented_images(
        segmented_folder=r"outputs\segmented",
        out_csv=r"outputs/example.csv",
    )


if __name__ == "__main__":
    main()