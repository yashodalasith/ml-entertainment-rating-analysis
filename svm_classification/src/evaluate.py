import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC

def calculate_metrics(y_test, y_pred, y_scores):
    """
    Calculates advanced classification metrics.
    """
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'kappa': kappa,
        'report': classification_report(y_test, y_pred)
    }
    return metrics

def plot_comparison_curves(y_test, results, filename='svm_comparison_curves.png'):
    """
    Plots ROC and PR curves for multiple kernels.
    """
    plt.figure(figsize=(14, 6))
    
    # ROC Plot
    plt.subplot(1, 2, 1)
    for kernel, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_scores'])
        plt.plot(fpr, tpr, label=f"{kernel.capitalize()} (AUC = {roc_auc_score(y_test, res['y_scores']):.2f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title("ROC Curves Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    
    # PR Plot
    plt.subplot(1, 2, 2)
    for kernel, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res['y_scores'])
        plt.plot(recall, precision, label=f"{kernel.capitalize()} (AUC = {auc(recall, precision):.2f})")
    plt.title("Precision-Recall Curves Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Comparison curves saved to {filename}")

def plot_decision_boundary(X, y, kernel='poly', filename='svm_decision_boundary.png'):
    """
    Plots the decision boundary in 2D using PCA.
    """
    print(f"Generating decision boundary plot for {kernel} kernel...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Train a "shadow" model on PCA space for visualization
    indices = np.random.choice(len(X_pca), 1000, replace=False)
    svc_pca = SVC(kernel=kernel, class_weight='balanced')
    svc_pca.fit(X_pca[indices], y.iloc[indices])
    
    h = .02 # step size
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = svc_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    sns.scatterplot(x=X_pca[indices, 0], y=X_pca[indices, 1], hue=y.iloc[indices], palette='coolwarm', s=20)
    plt.title(f"SVM Decision Boundary ({kernel.capitalize()}) - PCA Space")
    plt.savefig(filename)
    print(f"Decision boundary plot saved to {filename}")

def plot_feature_importance(model, X_test, y_test, feature_names, filename='svm_feature_importance.png'):
    """
    Calculates and plots permutation importance.
    """
    print("Computing permutation importance...")
    perm_importance = permutation_importance(model, X_test[:500], y_test[:500], n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()[-10:] # Top 10 features
    
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], perm_importance.importances_mean[sorted_idx], color='teal')
    plt.title("Permutation Importance (Top 10 Features)")
    plt.xlabel("Importance (Decrease in Accuracy)")
    plt.savefig(filename)
    print(f"Feature importance plot saved to {filename}")
