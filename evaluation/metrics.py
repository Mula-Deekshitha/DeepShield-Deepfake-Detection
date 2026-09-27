import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray):
    """
    Computes comprehensive research metrics for deepfake detection.
    """
    y_pred = (y_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        roc_auc = 0.5
        
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_pts, precision_pts)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion_Matrix": cm
    }
    
    return metrics