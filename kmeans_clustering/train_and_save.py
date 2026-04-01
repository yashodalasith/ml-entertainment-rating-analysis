from pathlib import Path

from src.clustering import train_cluster_and_save
from src.data_utils import load_raw_tables
from src.preprocessing import prepare_kmeans_features


def main():
    base_dir = Path(__file__).resolve().parent
    data = load_raw_tables(start_dir=base_dir)

    X, analysis_df, meta = prepare_kmeans_features(
        data["animes"], data["profiles"], data["reviews"]
    )

    output = train_cluster_and_save(
        X=X,
        analysis_df=analysis_df,
        artifacts_dir=base_dir / "artifacts",
    )

    print("Training complete.")
    print("Best K:", output["best_k"])
    print("SVD components:", output["n_svd"])
    print("Saved model:", output["model_path"])
    print("\nCluster meanings:")
    print(output["cluster_meanings"].to_string(index=False))
    print("\nRepresentative samples per cluster:")
    print(output["cluster_examples"].head(12).to_string(index=False))
    print("Feature summary:")
    print("- numeric columns:", len(meta["numeric_cols"]))
    print("- categorical columns:", len(meta["categorical_cols"]))


if __name__ == "__main__":
    main()
