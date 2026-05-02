import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, auc

MODELS_DIR = "models"


def train_knn(X_train, y_train, n_neighbors=5, weights='uniform'):
    """
    Trains a K-Nearest Neighbors classifier with the specified hyperparameters.
    """
    print(f"Training KNN (n_neighbors={n_neighbors}, weights={weights})...")
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_and_select_best(X_train, y_train, X_test, y_test):
    """
    Trains multiple KNN variants (different K values and weighting schemes)
    and selects the best one based on PR-AUC.
    """
    configs = [
        {'n_neighbors': 3, 'weights': 'uniform'},
        {'n_neighbors': 5, 'weights': 'uniform'},
        {'n_neighbors': 7, 'weights': 'distance'},
        {'n_neighbors': 11, 'weights': 'distance'},
    ]

    best_auc = 0
    best_model = None
    best_config = None

    results = {}

    for config in configs:
        label = f"K{config['n_neighbors']}_{config['weights']}"
        model = train_knn(X_train, y_train, config['n_neighbors'], config['weights'])

        # Evaluate for selection (PR-AUC)
        y_scores = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)

        print(f"Config: {label} | PR-AUC: {pr_auc:.4f}")

        results[label] = {
            'model': model,
            'pr_auc': pr_auc,
            'y_scores': y_scores
        }

        if pr_auc > best_auc:
            best_auc = pr_auc
            best_model = model
            best_config = label

    print(f"\nBest Model: {best_config} with PR-AUC: {best_auc:.4f}")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_knn_model.pkl"))
    print(f"Best model saved to {os.path.join(MODELS_DIR, 'best_knn_model.pkl')}")

    return results, best_config
