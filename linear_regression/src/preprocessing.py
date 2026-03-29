import re

import numpy as np
import pandas as pd


def find_first_present(columns, candidates):
    for column in candidates:
        if column in columns:
            return column
    return None


def duration_to_minutes(text):
    if pd.isna(text):
        return np.nan
    value = str(text).lower()
    hr = re.search(r"(\d+)\s*hr", value)
    mn = re.search(r"(\d+)\s*min", value)
    total = 0
    if hr:
        total += int(hr.group(1)) * 60
    if mn:
        total += int(mn.group(1))
    if total == 0:
        just_num = re.search(r"(\d+)", value)
        return float(just_num.group(1)) if just_num else np.nan
    return float(total)


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


def prepare_supervised_dataframe(animes: pd.DataFrame, reviews: pd.DataFrame):
    anime_id_col = find_first_present(animes.columns, ["uid", "anime_id", "anime_uid", "id"])
    target_col = "score" if "score" in animes.columns else None

    if anime_id_col is None:
        raise ValueError("No anime identifier column found in animes.csv")
    if target_col is None:
        raise ValueError("No score column found in animes.csv")

    df = animes.copy()

    if "duration" in df.columns:
        df["duration_minutes"] = df["duration"].map(duration_to_minutes)
    if "aired" in df.columns:
        df["aired_year"] = df["aired"].map(extract_year)
    if "genre" in df.columns:
        df["genre_count"] = df["genre"].map(count_list_items)
    if "title" in df.columns:
        df["title_len"] = df["title"].map(text_len)
    if "synopsis" in df.columns:
        df["synopsis_len"] = df["synopsis"].map(text_len)

    review_anime_col = find_first_present(reviews.columns, ["anime_uid", "anime_id", "uid"])
    review_score_col = find_first_present(reviews.columns, ["score", "overall"])

    if review_anime_col and review_score_col:
        review_agg = reviews.groupby(review_anime_col).agg(
            review_count=(review_score_col, "size"),
            review_score_mean=(review_score_col, "mean"),
            review_score_std=(review_score_col, "std"),
        )
        review_agg = review_agg.reset_index().rename(columns={review_anime_col: anime_id_col})
        df = df.merge(review_agg, on=anime_id_col, how="left")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    drop_candidates = [
        target_col,
        "rank",
        "ranked",
        "link",
        "img_url",
        "title_synonyms",
        "synopsis",
        "aired",
        "duration",
    ]

    feature_df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

    for id_col in [anime_id_col, "uid", "anime_uid", "id"]:
        if id_col in feature_df.columns:
            feature_df = feature_df.drop(columns=id_col)

    cat_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    safe_cat_cols = []
    for col in cat_cols:
        if feature_df[col].nunique(dropna=True) <= 40:
            safe_cat_cols.append(col)

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    X = pd.concat([feature_df[numeric_cols], feature_df[safe_cat_cols]], axis=1).copy()
    y = df[target_col].copy()

    meta = {
        "anime_id_col": anime_id_col,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": safe_cat_cols,
        "dropped_high_cardinality_categoricals": [c for c in cat_cols if c not in safe_cat_cols],
    }
    return X, y, df, meta
