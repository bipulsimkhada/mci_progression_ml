import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_mean_roc_pr_curves(model_results_dict, name = None):
    """
    Plots mean ROC and PR curves as subplots using seaborn styling.
    """

    # Seaborn style for research-quality plots
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("tab10", n_colors=len(model_results_dict))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(f"Comparison of {"Model Performance" if name is None else name} Across 25-Fold Cross-Validation", fontsize=14)

    # ---------------- ROC ----------------
    ax = axes[0]
    mean_fpr = np.linspace(0, 1, 200)

    for (model_name, results), color in zip(model_results_dict.items(), palette):
        tprs = []
        aucs = []

        for r in results:
            roc_curve_data = r.get("outer_roc_curve")
            auc_val = r.get("outer_roc")

            if roc_curve_data is None or auc_val is None or np.isnan(auc_val):
                continue

            fpr = roc_curve_data.get("fpr")
            tpr = roc_curve_data.get("tpr")

            if fpr is None or tpr is None:
                continue

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_val)

        if len(tprs) == 0:
            continue

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0

        ax.plot(
            mean_fpr,
            mean_tpr,
            label=f"{model_name} (AUC={mean_auc:.3f} ± {std_auc:.3f})",
            linewidth=2,
            color=color
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Mean Across Folds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---------------- PR ----------------
    ax = axes[1]
    mean_recall = np.linspace(0, 1, 200)

    for (model_name, results), color in zip(model_results_dict.items(), palette):
        precisions_interp = []
        pr_aucs = []

        for r in results:
            pr_curve_data = r.get("outer_pr_curve")
            pr_auc_val = r.get("outer_pr_auc")

            if pr_curve_data is None or pr_auc_val is None or np.isnan(pr_auc_val):
                continue

            recall = np.array(pr_curve_data.get("recall"))
            precision = np.array(pr_curve_data.get("precision"))

            if recall is None or precision is None:
                continue

            recall_rev = recall[::-1]
            precision_rev = precision[::-1]

            recall_unique, unique_idx = np.unique(recall_rev, return_index=True)
            precision_unique = precision_rev[unique_idx]

            interp_precision = np.interp(mean_recall, recall_unique, precision_unique)
            precisions_interp.append(interp_precision)
            pr_aucs.append(pr_auc_val)

        if len(precisions_interp) == 0:
            continue

        mean_precision = np.mean(precisions_interp, axis=0)
        mean_pr_auc = np.mean(pr_aucs)
        std_pr_auc = np.std(pr_aucs, ddof=1) if len(pr_aucs) > 1 else 0.0

        ax.plot(
            mean_recall,
            mean_precision,
            label=f"{model_name} (AUC={mean_pr_auc:.3f} ± {std_pr_auc:.3f})",
            linewidth=2,
            color=color
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Mean Across Folds)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

   

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # Panel labels below x-axis labels
    fig.text(0.25, 0.005, "(a)", ha="center", va="bottom",
            fontsize=14)
    fig.text(0.75, 0.005, "(b)", ha="center", va="bottom",
            fontsize=14)

    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/plots/roc_pr_curves.png", dpi=300, bbox_inches="tight")  # for paper
    plt.show()