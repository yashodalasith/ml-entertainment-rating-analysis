import joblib
import pandas as pd
import os
import argparse
from src.preprocessing import preprocess_inference_data

MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_logistic_model.pkl")


def check_model_exists():
    """
    Checks if the model and preprocessors are present.
    """
    if not (os.path.exists(BEST_MODEL_PATH) and
            os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")) and
            os.path.exists(os.path.join(MODELS_DIR, "mlb.pkl"))):
        print("Model or preprocessors not found. Please run logistic_model.py first to train the model.")
        return False
    return True


def predict_anime_hit(members=None, popularity=None, episodes=None, ranked=None, genre=None):
    """
    Predicts if an anime will be a 'Hit' (> 8.0) or 'Standard'.
    """
    if not check_model_exists():
        return None, None

    model = joblib.load(BEST_MODEL_PATH)

    input_dict = {
        'members': members,
        'popularity': popularity,
        'episodes': episodes,
        'ranked': ranked,
        'genre': genre
    }

    X_scaled = preprocess_inference_data(input_dict)

    prob = model.predict_proba(X_scaled)[0, 1]
    prediction = "Hit" if prob > 0.5 else "Standard"

    return prediction, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an anime is a 'Hit' using Logistic Regression.")
    parser.add_argument("--members", type=int, help="Number of members.")
    parser.add_argument("--popularity", type=int, help="Popularity rank.")
    parser.add_argument("--episodes", type=int, help="Number of episodes.")
    parser.add_argument("--ranked", type=int, help="Score rank.")
    parser.add_argument("--genre", nargs="+", help="Genres.")

    args = parser.parse_args()

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
        print(f"\nPrediction: {pred.upper()} (Threshold: > 8.0)")
        print(f"Confidence (Hit Probability): {prob:.2%}")
        if pred == "Hit":
            print("This anime is likely a massive Hit!")
        else:
            print("This anime is predicted to be a 'Standard Release'.")
