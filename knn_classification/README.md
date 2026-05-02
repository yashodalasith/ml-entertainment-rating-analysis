# KNN Classification for MyAnimeList Hits

This project uses K-Nearest Neighbors (KNN) to classify whether an anime will be a "Hit" (score > 8.0) or a "Standard Release" based on features like members, popularity, episodes, ranked, and genres.

## How to Run the Analysis

1.  **Dependencies**: Ensure you have the required libraries installed:
    ```bash
    pip install kagglehub pandas scikit-learn numpy matplotlib seaborn
    ```
2.  **Execution**: Run the main script to train the models and generate visualizations:
    ```bash
    python knn_model.py
    ```
    This script will:
    - Download the dataset (approx. 200MB).
    - Preprocess and encode features.
    - Train KNN with **multiple K values and weighting schemes** (K=3, 5, 7, 11).
    - Select the best model based on PR-AUC.
    - Generate performance metrics and plots.

## Key Validation Metrics

All 4 classification models in this project use the **same validation metrics** for fair comparison:

-   **Accuracy**: Overall correct predictions.
-   **Precision**: Of predicted Hits, how many are actual Hits.
-   **Recall**: Of actual Hits, how many were correctly identified.
-   **F1 Score**: Harmonic mean of Precision and Recall.
-   **ROC-AUC**: Measures the model's ability to distinguish between classes.
-   **PR-AUC (Precision-Recall AUC)**: Crucial for imbalanced data.
-   **MCC (Matthews Correlation Coefficient)**: Robust metric for imbalanced classes.
-   **Cohen's Kappa**: Agreement adjusted for chance.

## Visualizations

1.  **ROC & PR Curves** (`knn_comparison_curves.png`): Compare different K/weight configurations.
2.  **Confusion Matrix** (`knn_confusion_matrix.png`): Shows true/false positives and negatives.
3.  **Feature Importance** (`knn_feature_importance.png`): Based on permutation importance.
