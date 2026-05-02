import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc

MODELS_DIR = "models"


def train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Trains a Gradient Boosting classifier with the specified hyperparameters.
    """
    print(f"Training Gradient Boosting (n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth})...")
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        subsample=0.8
    )
    model.fit(X_train, y_train)
    return model


def train_and_select_best(X_train, y_train, X_test, y_test):
    """
    Trains multiple Gradient Boosting variants and selects the best one based on PR-AUC.
    """
    configs = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5},
    ]

    best_auc = 0
    best_model = None
    best_config = None

    results = {}

    for config in configs:
        label = f"n{config['n_estimators']}_lr{config['learning_rate']}_d{config['max_depth']}"
        model = train_gradient_boosting(
            X_train, y_train,
            config['n_estimators'],
            config['learning_rate'],
            config['max_depth']
        )

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
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_gradient_boosting_model.pkl"))
    print(f"Best model saved to {os.path.join(MODELS_DIR, 'best_gradient_boosting_model.pkl')}")

    return results, best_config
