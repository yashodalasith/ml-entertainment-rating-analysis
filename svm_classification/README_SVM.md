# SVM Classification for MyAnimeList Hits

This project uses Support Vector Machines (SVM) to classify whether an anime will be a "Hit" (score > 8.0) or a "Standard Release" based on features like members, popularity, episodes, and genres.

## How to Run the Analysis

1.  **Dependencies**: Ensure you have the required libraries installed:
    ```bash
    pip install kagglehub pandas scikit-learn numpy matplotlib seaborn
    ```
2.  **Execution**: Run the main script to train the models and generate advanced visualizations:
    ```bash
    python svm_model.py
    ```
    This script will:
    - Download the dataset (approx. 200MB).
    - Preprocess and encode features.
    - Train **Linear**, **RBF**, and **Polynomial** SVM kernels.
    - Generate performance metrics and plots.

## Key Validation Metrics Explained

-   **ROC-AUC**: Measures the model's ability to distinguish between classes. 1.0 is perfect; 0.5 is random guessing.
-   **PR-AUC (Precision-Recall AUC)**: **Crucial for imbalanced data**. It focuses on the minority class ("Hits"). Higher is better.
-   **MCC (Matthews Correlation Coefficient)**: A more robust metric for imbalanced classes than Accuracy. Values range from -1 to 1.
-   **Cohen's Kappa**: Measures the agreement between predicted and actual classes, accounting for the possibility of agreement occurring by chance.

## Interpreting the Visualizations

1.  **ROC & PR Curves (`svm_comparison_curves.png`)**:
    -   Compare the three kernels. The one with the highest area under the curve is the best performer.
2.  **Decision Boundary (`svm_decision_boundary.png`)**:
    -   Shows how the SVM divides the feature space. Red/Blue regions indicate predicted classes. The dots are actual samples.
3.  **Permutation Importance (`svm_feature_importance.png`)**:
    -   Displays which features are most influential. If `members` is at the top, it means shuffling that column severely drops the model's accuracy, indicating it's a key predictor.
4.  **EDA Visuals (`eda_svm.ipynb`)**:
    -   Explore the raw relationships before modeling using Violin plots and Joint plots.
