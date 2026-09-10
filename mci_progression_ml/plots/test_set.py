import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

def plot(y_test, y_score, y_pred):

    # -----------------------------
    # Metrics
    # -----------------------------
    roc_fpr, roc_tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(roc_fpr, roc_tpr)

    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    prob_true, prob_pred = calibration_curve(
        y_test, y_score, n_bins=10, strategy="quantile"
    )
    brier = brier_score_loss(y_test, y_score)

    cm = confusion_matrix(y_test, y_pred)
    prevalence = np.mean(y_test)

    # -----------------------------
    # Plotting: 2x2 layout
    # -----------------------------
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    fig.suptitle("Comprehensive Model Performance Evaluation", fontsize=14)

    # 1. ROC Curve
    ax = axes[0, 0]
    sns.lineplot(x=roc_fpr, y=roc_tpr, ax=ax, linewidth=2,
                label=f"Model (AUC = {roc_auc:.3f})")
    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--",
                color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=True)

    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    sns.lineplot(x=pr_recall, y=pr_precision, ax=ax, linewidth=2,
                label=f"Model (AP = {pr_auc:.3f})")
    sns.lineplot(
        x=[0, 1],
        y=[prevalence, prevalence],
        ax=ax,
        linestyle="--",
        color="gray",
        label=f"Baseline = {prevalence:.3f}"
    )
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", frameon=True)

    # 3. Calibration Plot
    ax = axes[1, 0]
    sns.lineplot(x=prob_pred, y=prob_true, marker="o", ax=ax, linewidth=2,
                label=f"Model (Brier = {brier:.3f})")
    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--",
                color="gray", label="Perfect calibration")
    ax.set_title("Calibration Plot")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="upper left", frameon=True)

    # 4. Confusion Matrix
    ax = axes[1, 1]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",   # use "Greys" for grayscale journals
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        square=True,
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(["Class 0", "Class 1"])
    ax.set_yticklabels(["Class 0", "Class 1"], rotation=0)

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, label in zip(axes.flat, panel_labels):
        ax.text(
            0.5, -0.25, label,
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="top"
        )

    # -----------------------------
    # Final layout and save
    # -----------------------------
    plt.tight_layout()
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/plots/model_performance_2x2.png", dpi=300, bbox_inches="tight")
    plt.show()