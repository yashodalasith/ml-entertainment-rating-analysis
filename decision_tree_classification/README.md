# Decision Tree Classification for MyAnimeList Hits

This project uses Decision Tree to classify whether an anime will be a "Hit" (score > 8.0) or a "Standard Release" based on features like members, popularity, episodes, ranked, and genres.

## How to Run the Analysis

1.  **Dependencies**: Ensure you have the required libraries installed:
    ```bash
    pip install kagglehub pandas scikit-learn numpy matplotlib seaborn
    ```
2.  **Execution**: Run the main script to train the models and generate visualizations:
    ```bash
    python decision_tree_model.py
    ```
    This script will:
    - Download the dataset (approx. 200MB).
    - Preprocess and encode features.
    - Train Decision Trees with **multiple depth/split configurations**.
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

1.  **ROC & PR Curves** (`dt_comparison_curves.png`): Compare different tree configurations.
2.  **Confusion Matrix** (`dt_confusion_matrix.png`): Shows true/false positives and negatives.
3.  **Feature Importance** (`dt_feature_importance.png`): Based on Gini impurity.
