import ast
import re

import numpy as np
import pandas as pd


def find_first_present(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def extract_year(text):
    if pd.isna(text):
        return np.nan
    years = re.findall(r"(19\d{2}|20\d{2})", str(text))
    return float(years[0]) if years else np.nan


def count_list_items(text):
    if pd.isna(text):
        return np.nan
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    return float(len(items))


def text_len(text):
    if pd.isna(text):
        return np.nan
    return float(len(str(text)))


def parse_favorites_list(value):
    if pd.isna(value):
        return []
    raw = str(value)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return re.findall(r"\d+", raw)


def build_profile_favorite_counts(profiles: pd.DataFrame) -> pd.DataFrame:
    profile_col = find_first_present(profiles.columns, ["profile", "username", "user"])
    favorites_col = find_first_present(profiles.columns, ["favorites_anime", "favorites"])

    if profile_col is None or favorites_col is None:
        return pd.DataFrame(columns=["uid", "favorites_count"])

    work = profiles[[profile_col, favorites_col]].copy()
    work["favorites_list"] = work[favorites_col].map(parse_favorites_list)
    exploded = work.explode("favorites_list")
    exploded = exploded[exploded["favorites_list"].notna()].copy()

    if exploded.empty:
        return pd.DataFrame(columns=["uid", "favorites_count"])

    exploded["uid"] = pd.to_numeric(exploded["favorites_list"], errors="coerce")
    exploded = exploded[exploded["uid"].notna()].copy()

    favorites_count = exploded.groupby("uid").size().reset_index(name="favorites_count")
    return favorites_count


def build_review_aggregates(reviews: pd.DataFrame) -> pd.DataFrame:
    anime_col = find_first_present(reviews.columns, ["anime_uid", "anime_id", "uid"])
    text_col = find_first_present(reviews.columns, ["text", "review", "content"])

    if anime_col is None:
        return pd.DataFrame(columns=["uid", "review_count", "review_text_len_mean"])

    work = reviews[[anime_col] + ([text_col] if text_col else [])].copy()

    if text_col:
        work["review_text_len"] = work[text_col].map(text_len)
        agg = work.groupby(anime_col).agg(
            review_count=(anime_col, "size"),
            review_text_len_mean=("review_text_len", "mean"),
        )
    else:
        agg = work.groupby(anime_col).agg(review_count=(anime_col, "size"))
        agg["review_text_len_mean"] = np.nan

    agg = agg.reset_index().rename(columns={anime_col: "uid"})
    return agg


def prepare_kmeans_features(animes: pd.DataFrame, profiles: pd.DataFrame, reviews: pd.DataFrame):
    anime_id_col = find_first_present(animes.columns, ["uid", "anime_id", "anime_uid", "id"])
    if anime_id_col is None:
        raise ValueError("No anime identifier column found in animes.csv")

    df = animes.copy()

    # Basic engineered features from anime metadata.
    if "aired" in df.columns:
        df["aired_year"] = df["aired"].map(extract_year)
    if "genre" in df.columns:
        df["genre_count"] = df["genre"].map(count_list_items)
    if "title" in df.columns:
        df["title_len"] = df["title"].map(text_len)
    if "synopsis" in df.columns:
        df["synopsis_len"] = df["synopsis"].map(text_len)

    # Ensure numeric fields are numeric.
    for col in ["episodes", "members", "popularity", "ranked", "score"]:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    # Merge profile-derived popularity signal.
    favorites_count = build_profile_favorite_counts(profiles)
    if not favorites_count.empty:
        df = df.merge(favorites_count, left_on=anime_id_col, right_on="uid", how="left", suffixes=("", "_fav"))
        if "uid_fav" in df.columns:
            df = df.drop(columns=["uid_fav"])

    # Merge review-derived engagement (without using review scores).
    review_agg = build_review_aggregates(reviews)
    if not review_agg.empty:
        df = df.merge(review_agg, left_on=anime_id_col, right_on="uid", how="left", suffixes=("", "_rev"))
        if "uid_rev" in df.columns:
            df = df.drop(columns=["uid_rev"])

    # Keep score only for post-hoc comparison; do not use it for clustering.
    analysis_cols = [anime_id_col]
    for col in ["title", "score", "members", "popularity"]:
        if col in df.columns:
            analysis_cols.append(col)
    analysis_df = df[analysis_cols].copy()

    drop_candidates = [
        "score",      # target-like column not used in unsupervised training
        "rank",
        "ranked",     # score-derived ranking
        "link",
        "img_url",
        "title_synonyms",
        "synopsis",
        "aired",
    ]

    feature_df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

    for id_col in [anime_id_col, "uid", "anime_uid", "anime_id", "id"]:
        if id_col in feature_df.columns:
            feature_df = feature_df.drop(columns=[id_col])

    cat_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    safe_cat_cols = []
    for c in cat_cols:
        if feature_df[c].nunique(dropna=True) <= 40:
            safe_cat_cols.append(c)

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    X = pd.concat([feature_df[numeric_cols], feature_df[safe_cat_cols]], axis=1).copy()

    meta = {
        "anime_id_col": anime_id_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": safe_cat_cols,
        "dropped_high_cardinality_categoricals": [c for c in cat_cols if c not in safe_cat_cols],
    }
    return X, analysis_df, meta
