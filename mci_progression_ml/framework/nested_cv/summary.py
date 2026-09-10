import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import t
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def mean_sd_ci(values, factor=1, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]*factor

    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    if n == 1:
        return float(values[0]), np.nan, np.nan, np.nan

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)

    alpha = 1 - confidence
    if SCIPY_AVAILABLE:
        crit = t.ppf(1 - alpha / 2, df=n - 1)
    else:
        crit = 1.96  # fallback approximation

    ci_low = mean - crit * se
    ci_high = mean + crit * se
    return mean, sd, float(ci_low), float(ci_high)


def summarize_family(name, results, confidence=0.95):
    def safe_get(metric):
        vals = []
        for r in results:
            v = r.get(metric)
            if v is not None and not np.isnan(v):
                vals.append(v)
        return vals

    metrics = {
        "f1_macro": ("outer_f1_macro", 1),
        "pr_auc": ("outer_pr_auc", 1),
        "roc_auc": ("outer_roc", 1),
        "balanced_acc": ("outer_balanced_acc", 100),
        "sensitivity": ("outer_sensitivity", 100),
        "specificity": ("outer_specificity", 100),
        "brier_score": ("calibrated_brier_score", 1),
        "threshold": ("best_threshold", 1)
    }

    summary = {"family": name, "n_folds": len(results)}

    for key, metric in metrics.items():
        name, factor = metric
        values = safe_get(name)
        mean, sd, ci_low, ci_high = mean_sd_ci(values, factor, confidence=confidence)

        summary[f"{key}_mean"] = mean
        summary[f"{key}_sd"] = sd
        summary[f"{key}_ci95_low"] = ci_low
        summary[f"{key}_ci95_high"] = ci_high
    
    time_6 = []
    time_12 = []
    time_24 = []

    for res in results:
        sensitivity_by_period = res.get("sensitivity_by_period", {})

        time_6.append(sensitivity_by_period.get("6", np.nan))
        time_12.append(sensitivity_by_period.get("12", np.nan))
        time_24.append(sensitivity_by_period.get("24", np.nan))


    def mean_ci(values):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]

        mean = np.mean(values)
        sd = np.std(values, ddof=1)
        n = len(values)

        # 95% CI using normal approximation
        se = sd / np.sqrt(n)
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se

        return mean, sd, ci_low, ci_high


    for name, values in [
        ("time_6", time_6),
        ("time_12", time_12),
        ("time_24", time_24),
    ]:
        mean, sd, ci_low, ci_high = mean_ci(values)

        summary[f"{name}"] = float(mean)
        summary[f"{name}_sd"] = float(sd)
        summary[f"{name}_ci95_low"] = float(ci_low)
        summary[f"{name}_ci95_high"] = float(ci_high)

    return summary
