import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from functions.tone_dataset_utils import export_images_by_tone


def main() -> None:
    export_images_by_tone(
        csv_path=r"outputs\example.csv",
        image_folder=r"outputs\segmented",
        output_folder=r"outputs\toneClassification\dark",
        tone="dark",
    )


if __name__ == "__main__":
    main()