import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, average_precision_score, make_scorer,
                            recall_score, matthews_corrcoef, precision_score,
                            confusion_matrix, roc_auc_score, brier_score_loss, log_loss)    

from mci_progression_ml.utils import get_binary_scores

eps=1e-15

def compute_metrics(y_true, y_pred, y_score):
    """
    Compute classification metrics for a binary classifier.
    y_score should be the positive-class score/probability.
    """

    

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Class-wise metrics
    precision_class0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    precision_class1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    recall_class0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_class1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    f1_class0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_class1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Macro metrics
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Positive-class metrics
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Predictive values
    # Positive Predictive Value (PPV)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    # Negative Predictive Value (NPV)
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    # Sensitivity / specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Overall metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    # Optional log loss
    y_score_clipped = np.clip(y_score, eps, 1 - eps)
    ll = log_loss(y_true, y_score_clipped)

    metrics = {
        # Confusion-matrix counts
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        # Class-wise
        "precision_class0": precision_class0,
        "precision_class1": precision_class1,
        "recall_class0": recall_class0,
        "recall_class1": recall_class1,
        "f1_class0": f1_class0,
        "f1_class1": f1_class1,

        # Macro
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,

        # Positive class
        "precision": precision,
        "recall": recall,
        "f1": f1,

        # Predictive values
        "ppv": ppv,
        "npv": npv,


        # Other
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": ll,
    }

    return metrics

def bootstrap_metrics(model, X_test, y_true, n_bootstrap=2000, random_state=42):
    """
    Bootstrap CIs on the held-out test set.
    Returns:
      - point_metrics: metrics computed on full test set
      - ci_df: dataframe with mean, lower, upper, point estimate
    """
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    y_score = np.asarray(get_binary_scores(model, X_test))
    threshold = getattr(model, "threshold", 0.5)
    y_pred = (y_score >= threshold).astype(int)

    n = len(y_true)
    point_metrics = compute_metrics(y_true, y_pred, y_score)

    # Only bootstrap scalar performance metrics, not raw counts
    metric_names = [
        "precision_class0", "precision_class1",
        "recall_class0", "recall_class1",
        "f1_class0", "f1_class1",
        "precision_macro", "recall_macro", "f1_macro",
        "precision", "recall", "f1",
        "sensitivity", "specificity",
        "ppv", "npv",
        "balanced_accuracy", "mcc",
        "roc_auc", "pr_auc",
        "brier_score", "log_loss"
    ]

    bootstrap_results = {m: [] for m in metric_names}

    for _ in range(n_bootstrap):
        # Resample test indices with replacement
        idx = rng.integers(0, n, n)

        y_b = y_true[idx]
        y_pred_b = y_pred[idx]
        y_score_b = y_score[idx]

        # Need both classes present for ROC-AUC / some metrics
        if len(np.unique(y_b)) < 2:
            continue

        try:
            m = compute_metrics(y_b, y_pred_b, y_score_b)
            for name in metric_names:
                bootstrap_results[name].append(m[name])
        except Exception:
            continue

    return {
        "point_metrics": point_metrics,
        "bootstrap_metrics": bootstrap_results,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "threshold": threshold,
    }


def print_metrics_with_ci(point_metrics, bootstrap_results, confidence=0.95):
    """
    Print point estimates and bootstrap confidence intervals.

    Parameters
    ----------
    point_metrics : dict
        Metrics computed on the full test set.

    bootstrap_results : dict
        Dictionary mapping metric names to bootstrap metric values,
        as returned by bootstrap_metrics()["bootstrap_metrics"].

    confidence : float, default=0.95
        Confidence level for the percentile bootstrap CI.
    """

    alpha = 1.0 - confidence
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    def fmt(value, digits=4):
        """Safely format a numeric value."""
        if value is None:
            return "N/A"

        try:
            value = float(value)
        except (TypeError, ValueError):
            return "N/A"

        if not np.isfinite(value):
            return "N/A"

        return f"{value:.{digits}f}"

    def show(metric_name, label):
        """Print point estimate and bootstrap percentile CI."""
        point = point_metrics.get(metric_name, np.nan)

        values = bootstrap_results.get(metric_name, [])

        # Remove NaN / inf values
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            print(f"{label}: {fmt(point)} (95% CI: N/A)")
            return

        ci_lower = np.percentile(values, lower_pct)
        ci_upper = np.percentile(values, upper_pct)

        print(
            f"{label}: {fmt(point)} "
            f"({fmt(np.nanmean(values))})(95% CI: {fmt(ci_lower)} - {fmt(ci_upper)})"
        )

    print("\nConfusion Matrix Counts")
    print(
        f"TN: {point_metrics.get('tn', 'N/A')}, "
        f"FP: {point_metrics.get('fp', 'N/A')}, "
        f"FN: {point_metrics.get('fn', 'N/A')}, "
        f"TP: {point_metrics.get('tp', 'N/A')}"
    )

    print("\nClass-wise Metrics")
    show("precision_class0", "Precision (Class 0)")
    show("precision_class1", "Precision (Class 1)")
    show("recall_class0", "Recall (Class 0)")
    show("recall_class1", "Recall (Class 1)")
    show("f1_class0", "F1 Score (Class 0)")
    show("f1_class1", "F1 Score (Class 1)")

    print("\nMacro Average Metrics")
    show("precision_macro", "Precision (Macro)")
    show("recall_macro", "Recall (Macro)")
    show("f1_macro", "F1 Score (Macro)")

    print("\nPositive-Class / Threshold Metrics")
    show("precision", "Precision")
    show("recall", "Recall")
    show("f1", "F1 Score")
    show("ppv", "Positive Predictive Value (PPV)")
    show("npv", "Negative Predictive Value (NPV)")

    print("\nOverall Classification Metrics")
    show("balanced_accuracy", "Balanced Accuracy")
    show("mcc", "MCC")

    print("\nSensitivity / Specificity")
    show("sensitivity", "Sensitivity (TPR)")
    show("specificity", "Specificity (TNR)")

    print("\nRanking / Probability Metrics")
    show("roc_auc", "ROC AUC")
    show("pr_auc", "PR AUC")
    show("brier_score", "Brier Score")
    show("log_loss", "Log Loss")

import numpy as np
import pandas as pd


# def metrics_with_ci_df(results, confidence=0.95, digits=4):
#     """
#     Create a DataFrame of metrics with bootstrap confidence intervals
#     for multiple ablation/model types.

#     Parameters
#     ----------
#     results : dict
#         Dictionary where:
#             key = ablation/model type
#             value = {
#                 "point": point_metrics,
#                 "bootstrap": bootstrap_results
#             }

#     confidence : float, default=0.95
#         Confidence level for the percentile bootstrap CI.

#     digits : int, default=4
#         Number of decimal places.

#     Returns
#     -------
#     pd.DataFrame
#         Rows are ablation/model types.
#         Columns are metrics.
#         Each cell contains:
#             point [lower - upper]
#     """

#     alpha = 1.0 - confidence
#     lower_pct = 100 * alpha / 2
#     upper_pct = 100 * (1 - alpha / 2)

#     metric_labels = {
#         "ppv": "PPV",
#         "npv": "NPV",
#         "sensitivity": "Sensitivity",
#         "specificity": "Specificity",
#         "balanced_accuracy": "Balanced Accuracy",
#     }

#     def fmt(value):
#         """Safely format a value."""
#         if value is None:
#             return "N/A"

#         try:
#             value = float(value)
#         except (TypeError, ValueError):
#             return "N/A"

#         if not np.isfinite(value):
#             return "N/A"

#         return f"{value:.{digits}f}"

#     def format_metric(point, bootstrap):
#         """Format point estimate and bootstrap CI."""
#         if bootstrap is None:
#             return fmt(point)

#         values = np.asarray(bootstrap, dtype=float)
#         values = values[np.isfinite(values)]

#         if len(values) == 0:
#             return fmt(point)

#         ci_lower = np.percentile(values, lower_pct)
#         ci_upper = np.percentile(values, upper_pct)

#         return f"{fmt(point)} [{fmt(ci_lower)} - {fmt(ci_upper)}]"

#     # Each ablation/model is now a row
#     data = []

#     for result_type, result in results.items():

#         point_metrics = result["point"]
#         bootstrap_results = result["bootstrap"]

#         row = {
#             "Ablation": result_type,
#             "Threshold": result["threshold"]
#         }

#         for metric_name, label in metric_labels.items():
#             point = point_metrics.get(metric_name, np.nan)
#             bootstrap = bootstrap_results.get(metric_name, [])

#             row[label] = format_metric(point, bootstrap)

#         data.append(row)

#     df = pd.DataFrame(data)

#     return df

def metrics_with_ci_df(
    results,
    confidence=0.95,
    digits=4,
    compare_against=None,
    compare_metric=None,
):
    """
    Create a DataFrame of metrics with bootstrap confidence intervals,
    with an optional statistical comparison against a reference model
    for a selected metric.

    Parameters
    ----------
    results : dict
        Dictionary where:
            key = ablation/model type
            value = {
                "point": point_metrics,
                "bootstrap": bootstrap_results,
                "threshold": threshold,
            }

    confidence : float, default=0.95
        Confidence level for the percentile bootstrap CI.

    digits : int, default=4
        Number of decimal places.

    compare_against : str, optional
        Name of the model/result to use as the reference.
        Example: "All Features".

    compare_metric : str, optional
        Metric to statistically compare.

        Available metrics:
            "ppv"
            "npv"
            "sensitivity"
            "specificity"
            "balanced_accuracy"

    Returns
    -------
    pd.DataFrame
        Rows are ablation/model types.

        All metrics are reported with their bootstrap CIs.

        If compare_against and compare_metric are provided,
        additional columns are included:

            Comparison
            Difference
            Comparison CI
            Significant

        The difference is:

            reference - model

        Therefore:
            positive difference = reference is better
            negative difference = model is better
    """

    alpha = 1.0 - confidence
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    metric_labels = {
        "ppv": "PPV",
        "npv": "NPV",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "balanced_accuracy": "Balanced Accuracy",
    }

    # ---------------------------------------------------------
    # Validate parameters
    # ---------------------------------------------------------

    if compare_against is not None:
        if compare_against not in results:
            raise ValueError(
                f"'{compare_against}' not found in results. "
                f"Available results: {list(results.keys())}"
            )

    if compare_metric is not None:
        if compare_metric not in metric_labels:
            raise ValueError(
                f"'{compare_metric}' is not a valid metric. "
                f"Available metrics: {list(metric_labels.keys())}"
            )

    if (compare_against is None) != (compare_metric is None):
        raise ValueError(
            "compare_against and compare_metric must either both "
            "be provided or both be None."
        )

    # ---------------------------------------------------------
    # Formatting helpers
    # ---------------------------------------------------------

    def fmt(value):
        if value is None:
            return "N/A"

        try:
            value = float(value)
        except (TypeError, ValueError):
            return "N/A"

        if not np.isfinite(value):
            return "N/A"

        return f"{value:.{digits}f}"

    def get_bootstrap_values(bootstrap):
        if bootstrap is None:
            return np.array([])

        values = np.asarray(bootstrap, dtype=float)
        return values[np.isfinite(values)]

    def format_metric(point, bootstrap):
        values = get_bootstrap_values(bootstrap)

        if len(values) == 0:
            return fmt(point)

        ci_lower = np.percentile(values, lower_pct)
        ci_upper = np.percentile(values, upper_pct)

        return (
            f"{fmt(point)} "
            f"[{fmt(ci_lower)} - {fmt(ci_upper)}]"
        )

    # ---------------------------------------------------------
    # Bootstrap comparison
    # ---------------------------------------------------------

    def compare_bootstrap(model_bootstrap, reference_bootstrap):

        model_values = get_bootstrap_values(model_bootstrap)
        reference_values = get_bootstrap_values(reference_bootstrap)

        if len(model_values) == 0 or len(reference_values) == 0:
            return np.nan, np.nan, np.nan, False

        if len(model_values) != len(reference_values):
            raise ValueError(
                "Bootstrap samples must have the same number of "
                "resamples for paired comparison. "
                f"Got {len(model_values)} vs "
                f"{len(reference_values)}."
            )

        # Reference - model
        differences = reference_values - model_values

        difference = np.mean(differences)

        ci_lower = np.percentile(
            differences,
            lower_pct,
        )

        ci_upper = np.percentile(
            differences,
            upper_pct,
        )

        # Reference is significantly better if
        # the entire CI is above zero.
        significant = ci_lower > 0

        return (
            difference,
            ci_lower,
            ci_upper,
            significant,
        )

    # ---------------------------------------------------------
    # Reference model
    # ---------------------------------------------------------

    reference_result = None

    if compare_against is not None:
        reference_result = results[compare_against]

    # ---------------------------------------------------------
    # Build DataFrame
    # ---------------------------------------------------------

    data = []

    for result_type, result in results.items():

        point_metrics = result["point"]
        bootstrap_results = result["bootstrap"]

        row = {
            "Ablation": result_type,
            "Threshold": result["threshold"],
            "selected_features": len(result["selected_features"]),
        }

        # -----------------------------------------------------
        # Display all metrics
        # -----------------------------------------------------

        for metric_name, label in metric_labels.items():

            point = point_metrics.get(
                metric_name,
                np.nan,
            )

            bootstrap = bootstrap_results.get(
                metric_name,
                [],
            )

            row[label] = format_metric(
                point,
                bootstrap,
            )

        # -----------------------------------------------------
        # Statistical comparison for selected metric
        # -----------------------------------------------------

        if compare_against is not None:

            metric_label = metric_labels[compare_metric]

            if result_type == compare_against:

                row["Comparison"] = "Reference"
                row["Difference"] = "Reference"
                row["Comparison CI"] = "Reference"
                row["Significant"] = "Reference"

            else:

                model_bootstrap = bootstrap_results.get(
                    compare_metric,
                    [],
                )

                reference_bootstrap = (
                    reference_result["bootstrap"].get(
                        compare_metric,
                        [],
                    )
                )

                (
                    difference,
                    ci_lower,
                    ci_upper,
                    significant,
                ) = compare_bootstrap(
                    model_bootstrap,
                    reference_bootstrap,
                )

                row["Comparison"] = (
                    f"{compare_against} vs {result_type} "
                    f"({metric_label})"
                )

                row["Difference"] = fmt(difference)

                if np.isfinite(ci_lower) and np.isfinite(ci_upper):
                    row["Comparison CI"] = (
                        f"[{fmt(ci_lower)} - "
                        f"{fmt(ci_upper)}]"
                    )
                else:
                    row["Comparison CI"] = "N/A"

                row["Significant"] = significant

        data.append(row)

    return pd.DataFrame(data)

