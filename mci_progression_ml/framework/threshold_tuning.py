from sklearn.metrics import f1_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


def tune_threshold_balanced(y_true, y_score, alpha=0.5):
    """
    alpha: weight for F1 vs balance
        0.5 → equal importance
        >0.5 → prioritize F1
        <0.5 → prioritize sensitivity/specificity balance
    """
    thresholds = np.unique(y_score)

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        # f1 = f1_score(y_true, y_pred, average="macro")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # balance term (penalizes imbalance)
        youden = sensitivity + specificity - 1

        score = youden

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
