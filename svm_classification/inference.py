import joblib
import pandas as pd
import os
import argparse
from src.preprocessing import preprocess_inference_data

MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_svm_model.pkl")

def check_model_exists():
    """
    Checks if the model and preprocessors are present.
    """
    if not (os.path.exists(BEST_MODEL_PATH) and
            os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")) and
            os.path.exists(os.path.join(MODELS_DIR, "mlb.pkl"))):
        print("Model or preprocessors not found. Please run svm_model.py first to train the model.")
        return False
    return True

def predict_anime_hit(members, popularity, episodes, ranked, genre):
    """
    Predicts if an anime will be a 'Hit' (> 8.0) or 'Standard'.
    Args:
        members (int): Total members on MyAnimeList.
        popularity (int): Popularity rank.
        episodes (int): Number of episodes.
        ranked (int): Score rank.
        genre (list): List of genres.
    Returns:
        prediction (str): 'Hit' or 'Standard'.
        probability (float): Confidence probability for 'Hit'.
    """
    if not check_model_exists():
        return None, None
    
    # Load model
    model = joblib.load(BEST_MODEL_PATH)
    
    # Prepare input
    input_dict = {
        'members': members,
        'popularity': popularity,
        'episodes': episodes,
        'ranked': ranked,
        'genre': genre
    }
    
    # Preprocess
    X_scaled = preprocess_inference_data(input_dict)
    
    # Predict
    prob = model.predict_proba(X_scaled)[0, 1] # Probability of 'Hit'
    prediction = "Hit" if prob > 0.5 else "Standard" # Default threshold 0.5
    
    return prediction, prob

if __name__ == "__main__":
    # Example usage via command line
    parser = argparse.ArgumentParser(description="Predict if an anime is a 'Hit'.")
    parser.add_argument("--members", type=int, default=1000000, help="Number of members.")
    parser.add_argument("--popularity", type=int, default=10, help="Popularity rank.")
    parser.add_argument("--episodes", type=int, default=12, help="Number of episodes.")
    parser.add_argument("--ranked", type=int, default=5, help="Score rank.")
    parser.add_argument("--genre", nargs="+", default=["Action", "Adventure", "Fantasy"], help="Genres.")
    
    args = parser.parse_args()
    
    print(f"\n--- Predicting for Anime Profile ---")
    print(f"Members: {args.members} | Popularity Rank: {args.popularity}")
    print(f"Episodes: {args.episodes} | Ranked: {args.ranked}")
    print(f"Genres: {', '.join(args.genre)}")
    
    pred, prob = predict_anime_hit(args.members, args.popularity, args.episodes, args.ranked, args.genre)
    
    if pred:
        print(f"\nPrediction: {pred.upper()}")
        print(f"Confidence (Hit Probability): {prob:.2%}")
        if pred == "Hit":
            print("Woot! This anime is likely a massive Hit (> 8.0 score)!")
        else:
            print("This anime is predicted to be a 'Standard Release'.")
