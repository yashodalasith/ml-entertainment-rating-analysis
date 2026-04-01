from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_NAME = "kmeans_pipeline.joblib"
INPUT_NAME = "cluster_input_template.csv"
OUTPUT_NAME = "cluster_predictions.csv"
MEANINGS_NAME = "cluster_meanings.csv"


def main():
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"

    model_path = artifacts_dir / MODEL_NAME
    input_path = artifacts_dir / INPUT_NAME
    output_path = artifacts_dir / OUTPUT_NAME
    meanings_path = artifacts_dir / MEANINGS_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    model = joblib.load(model_path)
    preprocessor = model.named_steps["preprocessor"]
    expected_columns = list(preprocessor.feature_names_in_)

    user_df = pd.read_csv(input_path)

    for col in expected_columns:
        if col not in user_df.columns:
            user_df[col] = np.nan

    X_user = user_df[expected_columns].copy()
    clusters = model.predict(X_user)

    output = user_df.copy()
    output["predicted_cluster"] = clusters

    if meanings_path.exists():
        meanings = pd.read_csv(meanings_path)
        if {"cluster", "cluster_label"}.issubset(meanings.columns):
            output = output.merge(
                meanings[["cluster", "cluster_label"]].rename(columns={"cluster": "predicted_cluster"}),
                on="predicted_cluster",
                how="left",
            )

    final_output_path = output_path
    try:
        output.to_csv(output_path, index=False)
    except PermissionError:
        fallback_path = artifacts_dir / "cluster_predictions_latest.csv"
        output.to_csv(fallback_path, index=False)
        final_output_path = fallback_path

    print("Cluster predictions saved:", final_output_path)
    cols_to_show = ["predicted_cluster"]
    if "cluster_label" in output.columns:
        cols_to_show.append("cluster_label")
    print(output[cols_to_show].head(10))


if __name__ == "__main__":
    main()
