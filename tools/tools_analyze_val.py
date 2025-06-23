from matplotlib import pyplot as plt
import tol_colors as tc
import pandas as pd

"""
plotting tools for validation metrics
"""

# Set default color palette for matplotlib (use 'bright', 'muted', etc.)
tc.set_default_colors(cset="bright")
tc.Muted


def generate_df_from_csv(*, csv_path, max_steps=None):
    df = pd.read_csv(csv_path)

    # Detect the step column
    step_col_candidates = [col for col in df.columns if col.lower() == "step"]
    step_col = step_col_candidates[0]

    # Get all column names excluding Step
    value_cols = [col for col in df.columns if col != step_col]

    # Apply max_steps if provided to limit the number of steps plotted
    if max_steps is not None:
        df = df[df[step_col] <= max_steps]
        # Debugging: print the number of rows after applying max_steps
        print(
            f"Sliced DataFrame size for {csv_path} with max_steps={max_steps}: {df.shape[0]} rows"
        )

    return df


def plot_dfs(
    df_counts,
    df_titles,
    save_path,
    title=None,
    beta_step=None,
    beta_label="Increase in $\\beta$",
    y_label="Workpiece count",
):
    # Prepare full filename
    filename = save_path / f"{title}.pdf"

    # Plot settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Plot all curves
    for df, label in zip(df_counts, df_titles):
        step_col = df.columns[0]
        value_col = df.columns[1]

        df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        ax.plot(df[step_col], df[value_col], label=rf"${label}$", linewidth=2.0)

    # Add vertical line + centered label after plotting
    if beta_step is not None:
        ax.axvline(x=beta_step, color="gray", linestyle="--", linewidth=1)

        # Get y-limits and compute midpoint
        y_min, y_max = ax.get_ylim()
        y_mid = y_min + 0.5 * (y_max - y_min)

        # ax.text(beta_step + 3, y_mid+60, beta_label, rotation=90,
        #        verticalalignment='center', horizontalalignment='left',
        #        fontsize=13, color='gray')

        ax.text(
            beta_step + 3,
            y_mid - 20,
            beta_label,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="left",
            fontsize=13,
            color="gray",
        )

    ax.set_xlabel("Step", fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")

    if title:
        ax.set_title(title.replace("_", " ").title(), fontsize=16, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=12, frameon=False)
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout(pad=2.0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Save figure
    fig.savefig(filename, bbox_inches="tight")
    # plt.close(fig)

    return fig
