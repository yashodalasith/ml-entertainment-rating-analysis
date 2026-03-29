from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_NAME = "best_model.joblib"
INPUT_NAME = "prediction_input_template.csv"
OUTPUT_NAME = "prediction_output.csv"


def main():
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"

    model_path = artifacts_dir / MODEL_NAME
    if not model_path.exists():
        model_path = artifacts_dir / "best_model_from_notebook.joblib"
    input_path = artifacts_dir / INPUT_NAME
    output_path = artifacts_dir / OUTPUT_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    model = joblib.load(model_path)
    preprocessor = model.named_steps["preprocessor"]
    expected_columns = list(preprocessor.feature_names_in_)

    user_df = pd.read_csv(input_path)

    # Keep only required columns and preserve training order.
    for col in expected_columns:
        if col not in user_df.columns:
            user_df[col] = np.nan

    X_user = user_df[expected_columns].copy()

    preds = model.predict(X_user)

    result = user_df.copy()
    result["predicted_score"] = preds
    result.to_csv(output_path, index=False)

    print("Predictions saved:", output_path)
    print(result[["predicted_score"]].head(10))


if __name__ == "__main__":
    main()
