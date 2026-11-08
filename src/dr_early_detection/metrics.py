import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
