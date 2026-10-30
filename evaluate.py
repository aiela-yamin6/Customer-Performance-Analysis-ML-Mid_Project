"""Convert model predictions into metrics safe for JSON responses."""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def classification_metrics(y, pred, prob):
    """Return common binary metrics and a 2 by 2 confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "specificity": float(tn / (tn + fp) if tn + fp else 0), "f1": float(f1_score(y, pred, zero_division=0)), "roc_auc": float(roc_auc_score(y, prob)), "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]], "caught": int(tp), "missed": int(fn), "false_alarms": int(fp)}

def regression_metrics(y, pred):
    """Return R2, MSE, RMSE, and MAE as ordinary Python numbers."""
    mse = mean_squared_error(y, pred)
    return {"r2": float(r2_score(y, pred)), "mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(y, pred))}
