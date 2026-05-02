import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score, f1_score,
    precision_score, recall_score
)


def calculate_metrics(y_test, y_pred, y_scores):
    """
    Calculates advanced classification metrics — same metrics used across all 4 models.
    """
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'kappa': kappa,
        'report': classification_report(y_test, y_pred)
    }
    return metrics


def plot_comparison_curves(y_test, results, filename='gb_comparison_curves.png'):
    """
    Plots ROC and PR curves for multiple configurations.
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for config, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_scores'])
        plt.plot(fpr, tpr, label=f"{config} (AUC = {roc_auc_score(y_test, res['y_scores']):.2f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title("ROC Curves Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    for config, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res['y_scores'])
        plt.plot(recall, precision, label=f"{config} (AUC = {auc(recall, precision):.2f})")
    plt.title("Precision-Recall Curves Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Comparison curves saved to {filename}")


def plot_confusion_matrix(y_test, y_pred, filename='gb_confusion_matrix.png'):
    """
    Plots the confusion matrix for the best model.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Standard', 'Hit'],
                yticklabels=['Standard', 'Hit'])
    plt.title("Confusion Matrix — Gradient Boosting")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def plot_feature_importance(model, X_test, y_test, feature_names, filename='gb_feature_importance.png'):
    """
    Plots the built-in feature importance for Gradient Boosting.
    """
    print("Computing feature importance from model...")
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[-10:]  # Top 10 features

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='mediumpurple')
    plt.title("Feature Importance — Gradient Boosting")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Feature importance plot saved to {filename}")
