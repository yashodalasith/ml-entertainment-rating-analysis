import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc

MODELS_DIR = "models"


def train_logistic_regression(X_train, y_train, penalty='l2', C=1.0):
    """
    Trains a Logistic Regression classifier with the specified regularization.
    """
    print(f"Training Logistic Regression (penalty={penalty}, C={C})...")
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=5000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_and_select_best(X_train, y_train, X_test, y_test):
    """
    Trains multiple Logistic Regression variants (different regularization strengths)
    and selects the best one based on PR-AUC.
    """
    configs = [
        {'penalty': 'l2', 'C': 0.01},
        {'penalty': 'l2', 'C': 0.1},
        {'penalty': 'l2', 'C': 1.0},
        {'penalty': 'l2', 'C': 10.0},
    ]

    best_auc = 0
    best_model = None
    best_config = None

    results = {}

    for config in configs:
        label = f"L2_C{config['C']}"
        model = train_logistic_regression(X_train, y_train, config['penalty'], config['C'])

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

    # Save the best model
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_logistic_model.pkl"))
    print(f"Best model saved to {os.path.join(MODELS_DIR, 'best_logistic_model.pkl')}")

    return results, best_config
