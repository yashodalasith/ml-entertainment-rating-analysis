from pathlib import Path

from src.data_utils import load_raw_tables
from src.preprocessing import prepare_supervised_dataframe
from src.training import train_evaluate_and_save


def main():
    base_dir = Path(__file__).resolve().parent
    data = load_raw_tables(start_dir=base_dir)

    X, y, _, meta = prepare_supervised_dataframe(data["animes"], data["reviews"])
    output = train_evaluate_and_save(X, y, artifacts_dir=base_dir / "artifacts")

    print("Training complete.")
    print("Best model:", output["best_model_name"])
    print("Saved model:", output["model_path"])
    print("Saved results:", output["results_path"])
    print("Saved metadata:", output["metadata_path"])
    print("Feature summary:")
    print("- numeric columns:", len(meta["numeric_cols"]))
    print("- categorical columns:", len(meta["categorical_cols"]))


if __name__ == "__main__":
    main()
