import pandas as pd
import numpy as np
import ast
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

pd.options.mode.chained_assignment = None
MODELS_DIR = "models"


def clean_data(df):
    """
    Cleans the data, handles missing values, and creates the binary target.
    Target: Hit (score > 8.0) = 1, Standard = 0.
    """
    print("Cleaning data...")
    df = df.dropna(subset=['score']).copy()
    df.loc[:, 'target'] = (df['score'] > 8.0).astype(int)

    features_num = ['members', 'popularity', 'episodes', 'ranked']
    medians = df[features_num].median().to_dict()

    for col, val in medians.items():
        df.loc[:, col] = df[col].fillna(val)

    return df, medians


def preprocess_training_data(df):
    """
    Full preprocessing pipeline for training.
    """
    print("Preprocessing training data...")
    df, medians = clean_data(df)

    df['genre'] = df['genre'].astype(object)
    df.loc[:, 'genre'] = df['genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genre'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

    features_num = ['members', 'popularity', 'episodes', 'ranked']
    X_num = df[features_num]

    X = pd.concat([X_num.reset_index(drop=True), genre_df.reset_index(drop=True)], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(mlb, os.path.join(MODELS_DIR, "mlb.pkl"))
    joblib.dump(medians, os.path.join(MODELS_DIR, "medians.pkl"))
    print("Preprocessors and training medians saved to models/")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def preprocess_inference_data(input_dict, models_dir="models"):
    """
    Preprocesses a single instance for inference.
    """
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    mlb = joblib.load(os.path.join(models_dir, "mlb.pkl"))
    medians = joblib.load(os.path.join(models_dir, "medians.pkl"))

    df_input = pd.DataFrame([input_dict])

    features_num = ['members', 'popularity', 'episodes', 'ranked']
    for col in features_num:
        if col not in df_input.columns or pd.isna(df_input[col][0]):
            df_input[col] = medians[col]

    X_num = df_input[features_num]

    genres = input_dict.get('genre', [])
    if genres is None:
        genres = []
    genre_encoded = mlb.transform([genres])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

    X = pd.concat([X_num.reset_index(drop=True), genre_df.reset_index(drop=True)], axis=1)
    X_scaled = scaler.transform(X)

    return X_scaled
