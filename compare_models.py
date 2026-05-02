"""
===================================================================================
Cross-Model Comparison Script
===================================================================================
Trains all 4 classification models on the SAME dataset with the SAME preprocessing,
and compares them using IDENTICAL validation metrics.

Models:
  1. Logistic Regression
  2. Decision Tree
  3. K-Nearest Neighbors (KNN)
  4. Gradient Boosting

Problem: Binary Classification — Hit (score > 8.0) vs Standard
Dataset: MyAnimeList (animes.csv)
===================================================================================
"""

import pandas as pd
import numpy as np
import ast
import os
import kagglehub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score, classification_report,
    confusion_matrix
)

pd.options.mode.chained_assignment = None


def load_and_preprocess():
    """
    Loads and preprocesses the data ONCE so all models use the exact same split.
    """
    print("=" * 70)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 70)

    path = kagglehub.dataset_download("marlesson/myanimelist-dataset-animes-profiles-reviews")
    animes_csv = os.path.join(path, "animes.csv")
    df = pd.read_csv(animes_csv)
    print(f"Loaded {len(df)} records.")

    # Clean
    df = df.dropna(subset=['score']).copy()
    df.loc[:, 'target'] = (df['score'] > 8.0).astype(int)

    features_num = ['members', 'popularity', 'episodes', 'ranked']
    medians = df[features_num].median().to_dict()
    for col, val in medians.items():
        df.loc[:, col] = df[col].fillna(val)

    # Genre Encoding
    df['genre'] = df['genre'].astype(object)
    df.loc[:, 'genre'] = df['genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genre'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

    X_num = df[features_num]
    X = pd.concat([X_num.reset_index(drop=True), genre_df.reset_index(drop=True)], axis=1)
    y = df['target']

    # SAME split for all models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Class distribution — Train: {dict(y_train.value_counts())} | Test: {dict(y_test.value_counts())}")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_all_models(X_train, y_train):
    """
    Trains all 4 models and returns them in a dictionary.
    """
    print("\n" + "=" * 70)
    print("TRAINING ALL MODELS")
    print("=" * 70)

    models = {}

    # 1. Logistic Regression
    print("\n[1/4] Training Logistic Regression...")
    lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced',
                            solver='lbfgs', max_iter=5000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # 2. Decision Tree
    print("[2/4] Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=2,
                                class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt

    # 3. KNN
    print("[3/4] Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)
    models['KNN'] = knn

    # 4. Gradient Boosting
    print("[4/4] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=5, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb

    print("\nAll models trained successfully!")
    return models


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluates all models using identical metrics and returns a comparison DataFrame.
    """
    print("\n" + "=" * 70)
    print("EVALUATING ALL MODELS")
    print("=" * 70)

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
        pr_auc_val = auc(recall_curve, precision_curve)

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_scores),
            'PR-AUC': pr_auc_val,
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Cohen Kappa': cohen_kappa_score(y_test, y_pred),
        }

        results.append(metrics)

        print(f"\n--- {name} ---")
        print(f"  Accuracy:     {metrics['Accuracy']:.4f}")
        print(f"  Precision:    {metrics['Precision']:.4f}")
        print(f"  Recall:       {metrics['Recall']:.4f}")
        print(f"  F1 Score:     {metrics['F1 Score']:.4f}")
        print(f"  ROC-AUC:      {metrics['ROC-AUC']:.4f}")
        print(f"  PR-AUC:       {metrics['PR-AUC']:.4f}")
        print(f"  MCC:          {metrics['MCC']:.4f}")
        print(f"  Cohen Kappa:  {metrics['Cohen Kappa']:.4f}")

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('Model')

    return comparison_df


def plot_roc_comparison(models, X_test, y_test, filename='comparison_roc_curves.png'):
    """
    Plots ROC curves for all models on a single chart.
    """
    plt.figure(figsize=(10, 8))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for (name, model), color in zip(models.items(), colors):
        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc_val = roc_auc_score(y_test, y_scores)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc_val:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
    plt.title('ROC Curve Comparison — All Models', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"ROC comparison saved to {filename}")


def plot_pr_comparison(models, X_test, y_test, filename='comparison_pr_curves.png'):
    """
    Plots Precision-Recall curves for all models on a single chart.
    """
    plt.figure(figsize=(10, 8))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for (name, model), color in zip(models.items(), colors):
        y_scores = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc_val = auc(recall, precision)
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{name} (AUC = {pr_auc_val:.4f})')

    plt.title('Precision-Recall Curve Comparison — All Models', fontsize=14, fontweight='bold')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"PR comparison saved to {filename}")


def plot_metrics_comparison(comparison_df, filename='comparison_metrics_bar.png'):
    """
    Creates a grouped bar chart comparing all metrics across models.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    metrics = comparison_df.columns
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 4][idx % 4]
        values = comparison_df[metric]
        bars = ax.bar(range(len(values)), values, color=colors)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Model Comparison — All Validation Metrics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison bar chart saved to {filename}")


def plot_confusion_matrices(models, X_test, y_test, filename='comparison_confusion_matrices.png'):
    """
    Plots confusion matrices for all 4 models side by side.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']

    for idx, ((name, model), cmap) in enumerate(zip(models.items(), cmaps)):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx],
                    xticklabels=['Standard', 'Hit'],
                    yticklabels=['Standard', 'Hit'])
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Confusion matrices saved to {filename}")


def main():
    # 1. Load and preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # 2. Train all models
    models = train_all_models(X_train, y_train)

    # 3. Evaluate all models
    comparison_df = evaluate_all_models(models, X_test, y_test)

    # 4. Print comparison table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)
    print(comparison_df.to_string())

    # 5. Determine the best model
    best_model_name = comparison_df['PR-AUC'].idxmax()
    print(f"\n*** BEST MODEL (by PR-AUC): {best_model_name} ***")
    print(f"   PR-AUC: {comparison_df.loc[best_model_name, 'PR-AUC']:.4f}")
    print(f"   ROC-AUC: {comparison_df.loc[best_model_name, 'ROC-AUC']:.4f}")
    print(f"   F1 Score: {comparison_df.loc[best_model_name, 'F1 Score']:.4f}")

    # 6. Generate comparison visualizations
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 70)
    plot_roc_comparison(models, X_test, y_test)
    plot_pr_comparison(models, X_test, y_test)
    plot_metrics_comparison(comparison_df)
    plot_confusion_matrices(models, X_test, y_test)

    # 7. Save comparison table to CSV
    comparison_df.to_csv('model_comparison_results.csv')
    print(f"\nComparison results saved to model_comparison_results.csv")

    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
