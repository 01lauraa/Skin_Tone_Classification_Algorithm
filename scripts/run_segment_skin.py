import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from functions.skin_segmentation import segment_folder


def main() -> None:
    segment_folder(
        input_folder=r"dataset\example",
        output_folder=r"outputs\segmented",
    )


if __name__ == "__main__":
    main()