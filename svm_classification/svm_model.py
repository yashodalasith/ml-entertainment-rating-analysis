import pandas as pd
import numpy as np
import os
import joblib
from src.data_loader import load_data
from src.preprocessing import preprocess_training_data
from src.train import train_and_select_best
from src.evaluate import calculate_metrics, plot_comparison_curves, plot_decision_boundary, plot_feature_importance

def main():
    """
    Main orchestration script for the SVM classification project.
    """
    # 1. Load Data
    df = load_data()
    if df is None:
        return
    
    # 2. Preprocess Data
    X_train, X_test, y_train, y_test, feature_names = preprocess_training_data(df)
    
    # 3. Train Models and Select Best
    results, best_kernel = train_and_select_best(X_train, y_train, X_test, y_test)
    
    # 4. Detailed Evaluation for Best Model
    best_results = results[best_kernel]
    metrics = calculate_metrics(y_test, best_results['model'].predict(X_test), best_results['y_scores'])
    
    print(f"\n--- Best Model ({best_kernel.upper()}) Metrics ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f} | Cohen Kappa: {metrics['kappa']:.4f}")
    print("\nClassification Report:\n", metrics['report'])
    
    # 5. Visualizations
    print("\n--- Generating Visualizations ---")
    plot_comparison_curves(y_test, results)
    plot_decision_boundary(X_test, y_test, kernel=best_kernel)
    plot_feature_importance(best_results['model'], X_test, y_test, feature_names)
    
    print("\nTraining and evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
