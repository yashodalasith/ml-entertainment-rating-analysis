import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def pick_svd_components(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> int:
    transformed = preprocessor.fit_transform(X_train)
    n_features = transformed.shape[1]
    return min(60, max(2, n_features - 1))


def evaluate_k_range(embedding: np.ndarray, k_values: list[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(embedding)

        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(embedding, labels),
                "calinski_harabasz": calinski_harabasz_score(embedding, labels),
                "davies_bouldin": davies_bouldin_score(embedding, labels),
            }
        )
    return pd.DataFrame(rows)


def choose_best_k(k_eval: pd.DataFrame) -> int:
    ranked = k_eval.sort_values(
        by=["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True],
    )
    return int(ranked.iloc[0]["k"])


def build_cluster_profile(assignments: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    profile = assignments.groupby("cluster").agg(cluster_size=("cluster", "size")).reset_index()

    if "score" in assignments.columns:
        score_stats = assignments.groupby("cluster").agg(
            score_mean=("score", "mean"),
            score_median=("score", "median"),
            score_std=("score", "std"),
        )
        profile = profile.merge(score_stats.reset_index(), on="cluster", how="left")

    if "members" in assignments.columns:
        members_stats = assignments.groupby("cluster").agg(
            members_mean=("members", "mean"),
            members_median=("members", "median"),
        )
        profile = profile.merge(members_stats.reset_index(), on="cluster", how="left")

    if "popularity" in assignments.columns:
        popularity_stats = assignments.groupby("cluster").agg(
            popularity_mean=("popularity", "mean"),
            popularity_median=("popularity", "median"),
        )
        profile = profile.merge(popularity_stats.reset_index(), on="cluster", how="left")

    return profile.sort_values("cluster").reset_index(drop=True)


def assign_cluster_meanings(cluster_profile: pd.DataFrame) -> pd.DataFrame:
    meanings = cluster_profile.copy()
    meanings["cluster_label"] = "Segment"
    meanings["reason"] = "General content segment"

    if meanings.empty:
        return meanings[["cluster", "cluster_label", "reason"]]

    if "members_mean" in meanings.columns and meanings["members_mean"].notna().any():
        mainstream_cluster = int(meanings.loc[meanings["members_mean"].idxmax(), "cluster"])
    elif "popularity_mean" in meanings.columns and meanings["popularity_mean"].notna().any():
        mainstream_cluster = int(meanings.loc[meanings["popularity_mean"].idxmin(), "cluster"])
    elif "score_mean" in meanings.columns and meanings["score_mean"].notna().any():
        mainstream_cluster = int(meanings.loc[meanings["score_mean"].idxmax(), "cluster"])
    else:
        mainstream_cluster = int(meanings["cluster"].min())

    low_engagement_cluster = None
    if "members_mean" in meanings.columns and meanings["members_mean"].notna().any():
        low_engagement_cluster = int(meanings.loc[meanings["members_mean"].idxmin(), "cluster"])

    for idx in meanings.index:
        cluster_id = int(meanings.loc[idx, "cluster"])
        if cluster_id == mainstream_cluster:
            meanings.loc[idx, "cluster_label"] = "Mainstream / High Engagement"
            meanings.loc[idx, "reason"] = "High members and concentrated popularity"
        elif low_engagement_cluster is not None and cluster_id == low_engagement_cluster:
            meanings.loc[idx, "cluster_label"] = "Long-tail / Lower Engagement"
            meanings.loc[idx, "reason"] = "Lower members and broader long-tail catalog"
        else:
            meanings.loc[idx, "cluster_label"] = f"Niche Segment {cluster_id}"
            meanings.loc[idx, "reason"] = "Intermediate behavior compared with other clusters"

    return meanings[["cluster", "cluster_label", "reason"]].sort_values("cluster").reset_index(drop=True)


def collect_cluster_examples(assignments: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if "title" not in assignments.columns:
        return pd.DataFrame(columns=["cluster", "cluster_label", "title", "members", "score", "popularity"])

    rows = []
    for cluster_id, grp in assignments.groupby("cluster"):
        if "title" in grp.columns:
            grp = grp.drop_duplicates(subset=["title"])

        rank_cols = [c for c in ["members", "score"] if c in grp.columns]
        if rank_cols:
            ordered = grp.sort_values(rank_cols, ascending=[False] * len(rank_cols))
        else:
            ordered = grp

        examples = ordered[[c for c in ["title", "members", "score", "popularity"] if c in ordered.columns]].head(top_n)
        for _, row in examples.iterrows():
            row_dict = {
                "cluster": int(cluster_id),
                "title": row.get("title"),
                "members": row.get("members") if "members" in examples.columns else np.nan,
                "score": row.get("score") if "score" in examples.columns else np.nan,
                "popularity": row.get("popularity") if "popularity" in examples.columns else np.nan,
            }
            rows.append(row_dict)

    return pd.DataFrame(rows)


def train_cluster_and_save(X: pd.DataFrame, analysis_df: pd.DataFrame, artifacts_dir: Path):
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=RANDOM_STATE)

    preprocessor = build_preprocessor(X_train)
    n_svd = pick_svd_components(preprocessor, X_train)
    svd = TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE)

    X_train_t = preprocessor.fit_transform(X_train)
    Z_train = svd.fit_transform(X_train_t)

    k_values = list(range(2, 11))
    k_eval = evaluate_k_range(Z_train, k_values)
    best_k = choose_best_k(k_eval)

    kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE)
    train_labels = kmeans.fit_predict(Z_train)

    X_test_t = preprocessor.transform(X_test)
    Z_test = svd.transform(X_test_t)
    test_labels = kmeans.predict(Z_test)

    train_silhouette = silhouette_score(Z_train, train_labels)
    test_silhouette = silhouette_score(Z_test, test_labels)

    inference_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svd", svd),
            ("kmeans", kmeans),
        ]
    )

    full_labels = inference_pipeline.predict(X)
    Z_full = svd.transform(preprocessor.transform(X))
    assignments = pd.DataFrame({"cluster": full_labels})
    assignments = pd.concat([analysis_df.reset_index(drop=True), assignments], axis=1)

    cluster_profile = build_cluster_profile(assignments, analysis_df)
    cluster_meanings = assign_cluster_meanings(cluster_profile)
    assignments = assignments.merge(cluster_meanings, on="cluster", how="left")
    cluster_examples = collect_cluster_examples(assignments, top_n=5)
    if not cluster_examples.empty:
        cluster_examples = cluster_examples.merge(cluster_meanings, on="cluster", how="left")
        cluster_examples = cluster_examples[
            ["cluster", "cluster_label", "title", "members", "score", "popularity"]
        ]

    cluster_projection = pd.DataFrame(
        {
            "svd_1": Z_full[:, 0],
            "svd_2": Z_full[:, 1] if Z_full.shape[1] > 1 else np.zeros(Z_full.shape[0]),
            "cluster": full_labels,
        }
    )
    cluster_projection = cluster_projection.merge(cluster_meanings, on="cluster", how="left")
    optional_cols = [c for c in ["title", "members", "popularity", "score"] if c in assignments.columns]
    if optional_cols:
        cluster_projection = pd.concat(
            [cluster_projection, assignments[optional_cols].reset_index(drop=True)], axis=1
        )

    k_eval_path = artifacts_dir / "k_evaluation.csv"
    assignments_path = artifacts_dir / "cluster_assignments.csv"
    profile_path = artifacts_dir / "cluster_profile.csv"
    meanings_path = artifacts_dir / "cluster_meanings.csv"
    examples_path = artifacts_dir / "cluster_examples.csv"
    projection_path = artifacts_dir / "cluster_projection.csv"
    model_path = artifacts_dir / "kmeans_pipeline.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    k_eval.to_csv(k_eval_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    cluster_profile.to_csv(profile_path, index=False)
    cluster_meanings.to_csv(meanings_path, index=False)
    cluster_examples.to_csv(examples_path, index=False)
    cluster_projection.to_csv(projection_path, index=False)
    joblib.dump(inference_pipeline, model_path)

    metadata = {
        "best_k": best_k,
        "n_svd_components": int(n_svd),
        "rows_total": int(X.shape[0]),
        "rows_train": int(X_train.shape[0]),
        "rows_test": int(X_test.shape[0]),
        "train_silhouette": float(train_silhouette),
        "test_silhouette": float(test_silhouette),
        "silhouette_gap": float(train_silhouette - test_silhouette),
        "artifacts": {
            "model": str(model_path),
            "k_evaluation": str(k_eval_path),
            "cluster_assignments": str(assignments_path),
            "cluster_profile": str(profile_path),
            "cluster_meanings": str(meanings_path),
            "cluster_examples": str(examples_path),
            "cluster_projection": str(projection_path),
            "metadata": str(metadata_path),
        },
    }

    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    return {
        "best_k": best_k,
        "n_svd": n_svd,
        "k_eval": k_eval,
        "assignments": assignments,
        "cluster_profile": cluster_profile,
        "cluster_meanings": cluster_meanings,
        "cluster_examples": cluster_examples,
        "cluster_projection": cluster_projection,
        "metadata": metadata,
        "model_path": model_path,
    }
