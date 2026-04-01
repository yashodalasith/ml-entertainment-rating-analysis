import argparse
import joblib
import os
import pandas as pd
from preprocessing import preprocess_inference_data

def predict(input_data):
    """
    Loads the best SVM model and performs prediction on the input data.
    """
    models_dir = "models"
    model_path = os.path.join(models_dir, "best_svm_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = joblib.load(model_path)
    
    # Preprocess the input data
    X_scaled = preprocess_inference_data(input_data, models_dir=models_dir)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.decision_function(X_scaled)[0] # SVM doesn't always have predict_proba
    
    result = "Hit" if prediction == 1 else "Standard"
    print(f"\n--- Prediction Result ---")
    print(f"Prediction: {result}")
    print(f"Threshold: > 8.0 Score")
    print(f"Decision Score: {probability:.4f}")
    print(f"-------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Inference for Entertainment Ratings")
    parser.add_argument("--members", type=float, help="Number of members")
    parser.add_argument("--popularity", type=float, help="Popularity rank")
    parser.add_argument("--episodes", type=float, help="Number of episodes")
    parser.add_argument("--ranked", type=float, help="Ranked position")
    parser.add_argument("--genre", nargs="+", help="List of genres (e.g., --genre Action Adventure)")
    
    args = parser.parse_args()
    
    input_data = {
        "members": args.members,
        "popularity": args.popularity,
        "episodes": args.episodes,
        "ranked": args.ranked,
        "genre": args.genre if args.genre else []
    }
    
    predict(input_data)
