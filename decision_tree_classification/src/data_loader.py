import pandas as pd
import kagglehub
import os
import shutil


def load_data():
    """
    Loads the MyAnimeList dataset.
    - If dataset exists locally → load it
    - Else → download via kagglehub and save locally
    """

    print("Loading dataset...")

    # Project root (go up from src/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Data folder path
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Expected file
    animes_csv = os.path.join(data_dir, "animes.csv")

    # ✅ Case 1: Already exists
    if os.path.exists(animes_csv):
        print("Dataset found locally. Loading from data folder...")
        df = pd.read_csv(animes_csv)
        print(f"Loaded {len(df)} records.")
        return df

    # ❌ Case 2: Not found → Download
    print("Dataset not found locally. Downloading from Kaggle...")

    try:
        path = kagglehub.dataset_download(
            "marlesson/myanimelist-dataset-animes-profiles-reviews"
        )

        # Copy all CSV files to data folder
        for file in os.listdir(path):
            if file.endswith(".csv"):
                src_file = os.path.join(path, file)
                dest_file = os.path.join(data_dir, file)
                shutil.copy(src_file, dest_file)

        print("Download complete. Files saved to data folder.")

        # Load main dataset
        df = pd.read_csv(animes_csv)
        print(f"Loaded {len(df)} records.")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None