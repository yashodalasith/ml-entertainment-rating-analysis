import joblib
import os
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, auc

MODELS_DIR = "models"

def train_svm(X_train, y_train, kernel='poly'):
    """
    Trains an SVM with the specified kernel.
    """
    print(f"Training SVM with {kernel} kernel...")
    model = SVC(kernel=kernel, class_weight='balanced', probability=True, random_state=42, max_iter=5000)
    model.fit(X_train, y_train)
    return model

def train_and_select_best(X_train, y_train, X_test, y_test, kernels=['linear', 'rbf', 'poly']):
    """
    Trains multiple models and selects the best one based on PR-AUC.
    """
    best_auc = 0
    best_model = None
    best_kernel = None
    
    results = {}
    
    for kernel in kernels:
        model = train_svm(X_train, y_train, kernel)
        
        # Evaluate for selection (PR-AUC)
        y_scores = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)
        
        print(f"Kernel: {kernel} | PR-AUC: {pr_auc:.4f}")
        
        results[kernel] = {
            'model': model,
            'pr_auc': pr_auc,
            'y_scores': y_scores
        }
        
        if pr_auc > best_auc:
            best_auc = pr_auc
            best_model = model
            best_kernel = kernel
            
    print(f"\nBest Model: {best_kernel.upper()} with PR-AUC: {best_auc:.4f}")
    
    # Save the best model
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_svm_model.pkl"))
    print(f"Best model saved to {os.path.join(MODELS_DIR, 'best_svm_model.pkl')}")
    
    return results, best_kernel
