import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from functions.visualisation import visualize_examples_by_tone


def main() -> None:
    df = pd.read_csv(r"outputs\example.csv")

    visualize_examples_by_tone(
        df=df,
        original_folder=r"dataset\example",
        save_path=r"outputs/example_visualization.png",
        n_cols=4,
    )


if __name__ == "__main__":
    main()