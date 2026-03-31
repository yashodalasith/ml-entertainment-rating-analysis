import pandas as pd
import kagglehub
import os

def load_data():
    """
    Downloads and loads the MyAnimeList dataset.
    """
    print("Downloading/Loading dataset...")
    try:
        path = kagglehub.dataset_download("marlesson/myanimelist-dataset-animes-profiles-reviews")
        animes_csv = os.path.join(path, "animes.csv")
        df = pd.read_csv(animes_csv)
        print(f"Successfully loaded {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
