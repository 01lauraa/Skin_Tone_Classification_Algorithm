import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from functions.tone_dataset_utils import count_images_per_tone


def main() -> None:
    count_images_per_tone(r"outputs\example.csv")


if __name__ == "__main__":
    main()