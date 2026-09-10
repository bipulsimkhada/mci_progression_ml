import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
    log_loss,
)

from mci_progression_ml.utils import get_binary_scores


def evaluate_outer_fold(best_estimator, X_fold_test, y_fold_test, threshold, diag_change_period):
    """Evaluate on outer fold with optional custom threshold + calibration diagnostics."""

    y_score = get_binary_scores(best_estimator, X_fold_test)

    # ---- Use custom threshold if provided ----
    if threshold is not None and y_score is not None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = best_estimator.predict(X_fold_test)

    f1 = f1_score(y_fold_test, y_pred, average="macro")
    pr_auc = average_precision_score(y_fold_test, y_score) if y_score is not None else np.nan

    tn, fp, fn, tp = confusion_matrix(y_fold_test, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    roc = roc_auc_score(y_fold_test, y_score) if y_score is not None else np.nan
    balanced_acc = balanced_accuracy_score(y_fold_test, y_pred)

    # ----- Calibration diagnostics -----
    if y_score is not None:
        y_true_arr = np.asarray(y_fold_test)
        y_score_arr = np.asarray(y_score)

        brier = brier_score_loss(y_true_arr, y_score_arr)

        mean_predicted_risk = np.mean(y_score_arr)
        observed_event_rate = np.mean(y_true_arr)

        calibration_bias = mean_predicted_risk - observed_event_rate

        if calibration_bias > 0:
            calibration_direction = "overestimate"
        elif calibration_bias < 0:
            calibration_direction = "underestimate"
        else:
            calibration_direction = "well_calibrated_on_average"
    else:
        brier = np.nan
        mean_predicted_risk = np.nan
        observed_event_rate = np.nan
        calibration_bias = np.nan
        calibration_direction = None

    # ----- Curve data -----
    if y_score is not None:
        precision, recall, pr_thresholds = precision_recall_curve(y_fold_test, y_score)
        fpr, tpr, roc_thresholds = roc_curve(y_fold_test, y_score)
    else:
        precision, recall, pr_thresholds = None, None, None
        fpr, tpr, roc_thresholds = None, None, None

    
    # ---- Sensitivity per diag_change_period (only for positive class) ----
    sensitivity_by_period = {}

    if diag_change_period is not None:
        y_true_arr = np.asarray(y_fold_test)
        y_pred_arr = np.asarray(y_pred)
        period_arr = np.asarray(diag_change_period)

        valid_mask = (~pd.isna(period_arr)) & (period_arr != -1)
        unique_periods = np.unique(period_arr[valid_mask])

        for p in unique_periods:
            mask = period_arr == p

            y_true_p = y_true_arr[mask]
            y_pred_p = y_pred_arr[mask]

            # Only consider actual positives (y=1)
            tp = np.sum((y_true_p == 1) & (y_pred_p == 1))
            fn = np.sum((y_true_p == 1) & (y_pred_p == 0))

            sensitivity_p = tp / (tp + fn) if (tp + fn) > 0 else np.nan

            sensitivity_by_period[int(p)] = sensitivity_p

    return {
        "f1_macro": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc,
        "balanced_acc": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,

        # Threshold used
        "threshold": threshold if threshold is not None else 0.5,

        # Calibration diagnostics
        "brier_score": brier,
        "mean_predicted_risk": mean_predicted_risk,
        "observed_event_rate": observed_event_rate,
        "calibration_bias": calibration_bias,
        "calibration_direction": calibration_direction,

        "y_true": np.asarray(y_fold_test),
        "y_pred": np.asarray(y_pred),
        "y_score": np.asarray(y_score) if y_score is not None else None,

        "pr_curve": {
            "precision": precision,
            "recall": recall,
            "thresholds": pr_thresholds,
        },
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
        },
        "sensitivity_by_period": sensitivity_by_period,
    }