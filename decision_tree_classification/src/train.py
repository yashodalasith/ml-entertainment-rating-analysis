import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, auc

MODELS_DIR = "models"


def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2):
    """
    Trains a Decision Tree classifier with the specified hyperparameters.
    """
    print(f"Training Decision Tree (max_depth={max_depth}, min_samples_split={min_samples_split})...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_and_select_best(X_train, y_train, X_test, y_test):
    """
    Trains multiple Decision Tree variants (different depths/splits)
    and selects the best one based on PR-AUC.
    """
    configs = [
        {'max_depth': 5, 'min_samples_split': 2},
        {'max_depth': 10, 'min_samples_split': 2},
        {'max_depth': 15, 'min_samples_split': 5},
        {'max_depth': None, 'min_samples_split': 10},
    ]

    best_auc = 0
    best_model = None
    best_config = None

    results = {}

    for config in configs:
        label = f"depth{config['max_depth']}_split{config['min_samples_split']}"
        model = train_decision_tree(X_train, y_train, config['max_depth'], config['min_samples_split'])

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
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_decision_tree_model.pkl"))
    print(f"Best model saved to {os.path.join(MODELS_DIR, 'best_decision_tree_model.pkl')}")

    return results, best_config
