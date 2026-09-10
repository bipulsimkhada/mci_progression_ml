import matplotlib.pyplot as plt
from pathlib import Path

def plot_global_importance(
    df,
    class_col,
    top_k=15,
    title="Global Feature Importance",
    xlabel=None,
    importance_type="shap",
    ax=None,
):
    """
    Plot global feature importance as a horizontal bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'feature' and class_col.

    class_col : str
        Column containing the importance values.

    top_k : int, default=15
        Number of top features to display.

    title : str
        Plot title.

    xlabel : str, optional
        X-axis label. If None, inferred from importance_type.

    importance_type : {"shap", "pi"}, default="shap"
        Determines how features are sorted:
        - "shap": sort by absolute SHAP value.
        - "pi": sort by permutation importance.

    ax : matplotlib.axes.Axes, optional
        Existing axis. If None, a standalone figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axis.
    """

    # -----------------------------
    # Validate importance type
    # -----------------------------
    if importance_type not in {"shap", "pi"}:
        raise ValueError(
            "importance_type must be either 'shap' or 'pi'"
        )

    # -----------------------------
    # Sort and select top features
    # -----------------------------
    if importance_type == "shap":
        df_plot = (
            df.sort_values(
                class_col,
                ascending=False,
                key=lambda s: s.abs()
            )
            .head(top_k)
        )

        if xlabel is None:
            xlabel = "Mean |SHAP value|"

    else:  # permutation importance
        df_plot = (
            df.sort_values(
                class_col,
                ascending=False
            )
            .head(top_k)
        )

        if xlabel is None:
            xlabel = "Permutation Importance"

    # -----------------------------
    # Create axis if needed
    # -----------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # -----------------------------
    # Plot horizontal bars
    # -----------------------------
    ax.barh(
        df_plot["feature"],
        df_plot[class_col]
    )

    ax.invert_yaxis()
    ax.grid(False)

    # -----------------------------
    # Add selection/fold information
    # -----------------------------
    for i, (_, row) in enumerate(df_plot.iterrows()):

        if "selected_count" in df_plot.columns:
            sel_pct = row["selected_count"]

            label = (
                f"{int(sel_pct)}"
                if not isinstance(sel_pct, float)
                or sel_pct >= 1
                else f"{sel_pct:.0%}"
            )

            ax.text(
                row[class_col],
                i,
                f"  {label}",
                va="center",
                fontsize=9
            )

    # -----------------------------
    # Add right-side padding
    # -----------------------------
    xmax = df_plot[class_col].max()

    if xmax > 0:
        ax.set_xlim(
            0,
            xmax * 1.15
        )

    # -----------------------------
    # Axis formatting
    # -----------------------------
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        length=6,
        width=1,
        direction="out"
    )

    ax.set_xlabel(xlabel)

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold"
    )

    return ax


import matplotlib.pyplot as plt


def plot_importance_comparison(
    df_left,
    col_left,
    title_left,
    df_right,
    col_right,
    title_right,
    importance_type="shap",
    panel_labels=("a", "b"),
    figsize=(14, 6),
    top_k=15,
    output_name=None,
):
    """
    Create a two-panel comparison plot for SHAP or
    permutation importance.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        DataFrames containing feature importance results.

    col_left, col_right : str
        Importance-value columns for the left and right panels.

    title_left, title_right : str
        Titles for the two panels.

    importance_type : {"shap", "pi"}
        Type of importance being plotted.

    panel_labels : tuple
        Labels for the two panels, e.g. ("a", "b").

    figsize : tuple
        Figure size.

    top_k : int
        Number of features to display in each panel.

    output_name : str, optional
        Base filename for saving PNG and TIFF.
        Example: "shap_global_comparison"
    """

    if importance_type not in {"shap", "pi"}:
        raise ValueError(
            "importance_type must be either 'shap' or 'pi'"
        )

    # ---------------------------------
    # Create subplot
    # ---------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    # ---------------------------------
    # Plot left panel
    # ---------------------------------
    plot_global_importance(
        df=df_left,
        class_col=col_left,
        top_k=top_k,
        title=title_left,
        importance_type=importance_type,
        ax=axes[0],
    )

    # ---------------------------------
    # Plot right panel
    # ---------------------------------
    plot_global_importance(
        df=df_right,
        class_col=col_right,
        top_k=top_k,
        title=title_right,
        importance_type=importance_type,
        ax=axes[1],
    )

    # ---------------------------------
    # Panel labels
    # ---------------------------------
    for ax, label in zip(axes, panel_labels):
        ax.text(
            0.5,
            -0.15,
            f"({label})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
        )

    # ---------------------------------
    # Layout
    # ---------------------------------
    plt.tight_layout()

    # ---------------------------------
    # Save if requested
    # ---------------------------------
    if output_name is not None:
        Path("results/plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"results/plots/{output_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return fig, axes
