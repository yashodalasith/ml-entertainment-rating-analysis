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

def predict_anime_hit(members=None, popularity=None, episodes=None, ranked=None, genre=None):
    """
    Predicts if an anime will be a 'Hit' (> 8.0) or 'Standard'.
    Args:
        members (int, optional): Total members on MyAnimeList.
        popularity (int, optional): Popularity rank.
        episodes (int, optional): Number of episodes.
        ranked (int, optional): Score rank.
        genre (list, optional): List of genres.
    Returns:
        prediction (str): 'Hit' or 'Standard'.
        probability (float): Confidence probability for 'Hit'.
    """
    if not check_model_exists():
        return None, None
    
    # Load model
    model = joblib.load(BEST_MODEL_PATH)
    
    # Prepare input (passing None is now handled by src.preprocessing)
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
    parser.add_argument("--members", type=int, help="Number of members.")
    parser.add_argument("--popularity", type=int, help="Popularity rank.")
    parser.add_argument("--episodes", type=int, help="Number of episodes.")
    parser.add_argument("--ranked", type=int, help="Score rank.")
    parser.add_argument("--genre", nargs="+", help="Genres.")
    
    args = parser.parse_args()
    
    # If all arguments are None, show a warning but proceed to show how it handles missing data
    if all(v is None for v in [args.members, args.popularity, args.episodes, args.ranked, args.genre]):
        print("Note: All inputs are missing. Using training set medians for prediction...\n")
    
    print(f"--- Predicting for Anime Profile ---")
    print(f"Members: {args.members if args.members is not None else '[Missing - Using Median]'}")
    print(f"Popularity Rank: {args.popularity if args.popularity is not None else '[Missing - Using Median]'}")
    print(f"Episodes: {args.episodes if args.episodes is not None else '[Missing - Using Median]'}")
    print(f"Ranked: {args.ranked if args.ranked is not None else '[Missing - Using Median]'}")
    print(f"Genres: {', '.join(args.genre) if args.genre is not None else '[Missing - Using Empty List]'}")
    
    pred, prob = predict_anime_hit(args.members, args.popularity, args.episodes, args.ranked, args.genre)
    
    if pred:
        print(f"\nPrediction: {pred.upper()}")
        print(f"Confidence (Hit Probability): {prob:.2%}")
        if pred == "Hit":
            print("Woot! This anime is likely a massive Hit (> 8.0 score)!")
        else:
            print("This anime is predicted to be a 'Standard Release'.")
