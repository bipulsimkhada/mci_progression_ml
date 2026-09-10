import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Clean model names
def clean_model_name(x):
    x = x.lower()
    if "randomforest" in x:
        return "rf"
    elif "xgb" in x:
        return "xgb"
    elif "svc" in x:
        return "svc"
    elif "logistic" in x or x == "lr":
        return "lr"
    return x.title()

# df["model"] = df["model"].apply(clean_model_name)

def plot_heatmap(df, modality=None, name=None):

    # Filter for dem_cog_mri modality
    if modality is not None:
        df_plot = df[df["modality"] == modality].copy()
    else:
        df_plot = df.copy()

    # Aggregate in case there are duplicate rows
    df_plot = (
        df_plot.groupby(["model", "sampling"], as_index=False)
        .agg({
            "f1_macro_mean": "mean",
            "pr_auc_mean": "mean",
            "sensitivity_mean": "mean",
            "specificity_mean": "mean"
        })
    )

    # Order rows/columns
    model_order = ["RF", "XGB", "SVC", "LR"]
    sampling_order = ["None", "SMOTENC", "SMOTENC-Tomek"]

    df_plot["model"] = pd.Categorical(df_plot["model"], categories=model_order, ordered=True)
    df_plot["sampling"] = pd.Categorical(df_plot["sampling"], categories=sampling_order, ordered=True)
    df_plot = df_plot.sort_values(["model", "sampling"])

    # -----------------------------
    # 3. Create pivot tables
    # -----------------------------
    pivot_f1 = df_plot.pivot(index="model", columns="sampling", values="f1_macro_mean")
    pivot_pr = df_plot.pivot(index="model", columns="sampling", values="pr_auc_mean")
    pivot_sens = df_plot.pivot(index="model", columns="sampling", values="sensitivity_mean")
    pivot_spec = df_plot.pivot(index="model", columns="sampling", values="specificity_mean")

    # -----------------------------
    # 4. Create 2x2 heatmaps
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    cmap = "YlGnBu"

    sns.heatmap(
        pivot_f1,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.6,
        linecolor="white",
        cbar=True,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("F1 (Macro)", fontweight="bold")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")

    sns.heatmap(
        pivot_pr,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.6,
        linecolor="white",
        cbar=True,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("PR-AUC", fontweight="bold")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")

    sns.heatmap(
        pivot_sens,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.6,
        linecolor="white",
        cbar=True,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Sensitivity (%)", fontweight="bold")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")

    sns.heatmap(
        pivot_spec,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.6,
        linecolor="white",
        cbar=True,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Specificity (%)", fontweight="bold")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")

    # Clean subplot appearance
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for label, ax in zip(panel_labels, axes.flat):
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.5, -0.15, label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14
        )

    fig.suptitle(
        f"Heatmap of Model vs Sampling Performance {name}",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )

    # for ax in axes.flat:
    #     ax.tick_params(axis="x", rotation=45)
    #     ax.tick_params(axis="y", rotation=0)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)

    fig.tight_layout()

    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"results/plots/{name.lower().replace(' ', '_')}_model_sampling_heatmaps.png", dpi=300, bbox_inches="tight")

    plt.show()