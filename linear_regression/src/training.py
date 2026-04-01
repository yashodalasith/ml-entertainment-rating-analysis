import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def build_pipelines(X_train: pd.DataFrame):
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    transformed = preprocessor.fit_transform(X_train)
    n_features = transformed.shape[1]
    n_svd = min(120, max(2, n_features - 1))

    pipelines = {
        "LinearRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("svd", TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE)),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("svd", TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE)),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "SGDRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("svd", TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE)),
                (
                    "model",
                    SGDRegressor(
                        loss="squared_error",
                        penalty="l2",
                        alpha=1e-4,
                        max_iter=3000,
                        tol=1e-3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    return pipelines, n_svd


def train_evaluate_and_save(X: pd.DataFrame, y: pd.Series, artifacts_dir: Path):
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipelines, n_svd = build_pipelines(X_train)

    rows = []
    fitted_models = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

        y_pred = pipe.predict(X_test)
        mae, rmse, r2 = regression_metrics(y_test, y_pred)

        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")

        rows.append(
            {
                "model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2_test": r2,
                "R2_cv_mean": cv_r2.mean(),
                "R2_cv_std": cv_r2.std(),
            }
        )

    results = pd.DataFrame(rows).sort_values("R2_test", ascending=False)
    best_model_name = results.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    results_path = artifacts_dir / "results.csv"
    model_path = artifacts_dir / "best_model.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    results.to_csv(results_path, index=False)
    joblib.dump(best_model, model_path)

    metadata = {
        "best_model": best_model_name,
        "n_svd_components": int(n_svd),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "artifacts": {
            "results_csv": str(results_path),
            "model_joblib": str(model_path),
            "metadata_json": str(metadata_path),
        },
    }

    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    return {
        "results": results,
        "best_model_name": best_model_name,
        "model_path": model_path,
        "results_path": results_path,
        "metadata_path": metadata_path,
    }
