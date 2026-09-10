import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

sns.set_theme(
    style="white",
    context="paper",
    font_scale=1.2
)

def sensitivity_by_conversion_point(df):
    """
    Plots sensitivity for pMCI by conversion point for the top 5 models.

    Parameters:
    - df: DataFrame containing model results with columns for sensitivity at different time points
          and their corresponding confidence intervals.
    """

    top5 = df.head(5).copy()

    timepoints = ["6 month", "12 month", "24 month"]
    time_cols = ["time_6", "time_12", "time_24"]
    ci_low_cols = [
        "time_6_ci95_low",
        "time_12_ci95_low",
        "time_24_ci95_low"
    ]
    ci_high_cols = [
        "time_6_ci95_high",
        "time_12_ci95_high",
        "time_24_ci95_high"
    ]

    # Create summary DataFrame: mean [95% CI] for each timepoint
    summary_df = top5[["model", "sampling", "modality"]].copy()

    for timepoint, mean_col, low_col, high_col in zip(
        timepoints,
        time_cols,
        ci_low_cols,
        ci_high_cols
    ):
        summary_df[timepoint] = [
            f"{mean * 100:.2f} [{low * 100:.2f}–{high * 100:.2f}]"
            for mean, low, high in zip(
                top5[mean_col],
                top5[low_col],
                top5[high_col]
            )
        ]

    print(summary_df)


    n_models = len(top5)
    n_times = len(timepoints)

    x = np.arange(n_times)

    # Width of each individual bar
    width = 0.15

    colors = sns.color_palette("tab10", n_models)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (_, row) in enumerate(top5.iterrows()):

        means = np.array([
            row[col] for col in time_cols
        ]) * 100

        ci_low = np.array([
            row[col] for col in ci_low_cols
        ]) * 100

        ci_high = np.array([
            row[col] for col in ci_high_cols
        ]) * 100

        # Convert CI bounds to error-bar distances
        yerr = np.vstack([
            means - ci_low,
            ci_high - means
        ])

        # Position bars around each timepoint
        positions = x + (i - (n_models - 1) / 2) * width

        label = (
            f"{str(row['model']).upper()} "
            f"({row['sampling']}) "
            f"({row['modality']})"
        )

        ax.bar(
            positions,
            means,
            width=width,
            color=colors[i],
            label=label,
            yerr=yerr,
            capsize=3,
            error_kw={
                "elinewidth": 1.2,
                "capthick": 1.2
            }
        )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(timepoints)

    ax.set_xlabel("Conversion Point")
    ax.set_ylabel("Sensitivity (%)")
    ax.set_title("Sensitivity for pMCI by Conversion Point")

    ax.set_ylim(70, 100)

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--"
    )

    # Legend below x-axis label
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3
    )

    sns.despine()

    plt.tight_layout()

    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/plots/conversion_point_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
