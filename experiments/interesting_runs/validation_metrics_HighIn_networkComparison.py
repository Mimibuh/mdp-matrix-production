from pathlib import Path
import pandas as pd

from matplotlib import pyplot as plt
import tol_colors as tc

from tools.global_config import custom_trained_models_path

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
    y_label="Workpiece Count",
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

        ax.text(
            beta_step + 3,
            y_mid,
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


def plot_dfs_dotted(
    df_counts,
    df_titles,
    save_path,
    title=None,
    beta_step=None,
    beta_label="Increase in $\\beta$",
):
    from itertools import cycle

    # Prepare full filename
    filename = save_path / f"{title}.pdf"

    # Plot settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Get default color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_iter = cycle(color_cycle)

    used_colors = []

    for idx, (df, label) in enumerate(zip(df_counts, df_titles)):
        step_col = df.columns[0]
        value_col = df.columns[1]

        df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        # Determine color
        if idx % 2 == 0:
            color = next(color_iter)
            used_colors.append(color)
            linestyle = "-"
        else:
            color = used_colors[-1]  # same color as previous
            linestyle = "--"  # dotted for second line

        ax.plot(
            df[step_col],
            df[value_col],
            label=rf"${label}$",
            linewidth=2.0,
            linestyle=linestyle,
            color=color,
        )

    # Add vertical beta line
    if beta_step is not None:
        ax.axvline(x=beta_step, color="gray", linestyle="--", linewidth=1)
        y_min, y_max = ax.get_ylim()
        y_mid = y_min + 0.5 * (y_max - y_min)

        ax.text(
            beta_step + 3,
            y_mid,
            beta_label,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="left",
            fontsize=13,
            color="gray",
        )

    # enlarge x space for legend
    x_min, _ = ax.get_xlim()
    ax.set_xlim(left=x_min, right=380)
    ax.set_xlabel("Step", fontsize=14, fontweight="bold")
    ax.set_ylabel("Workpiece count", fontsize=14, fontweight="bold")

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


# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
# model_folder_name_low = (
#    r"simple_3by3_oldarrival_run_arrival_change_201_007_conv2d_t_2025-06-16_17-42-48"
# )
model_folder_name_low = r"simple_3by3_oldarrival_conv2d"

# model_folder_name_high = r"3x3_HighIn_simple_3by3_oldarrival_run_arrival_change_201_007_transformer_t_2025-06-09_09-54-32"
model_folder_name_high = r"3x3_HighIn_simple_3by3_oldarrival_transformer"

# model_folder_name_good = r"3x3HighIn_simple_3by3_oldarrival_run_arrival_change_201_007_simple_linear_t_2025-06-15_18-52-58"
model_folder_name_good = r"3x3HighIn_simple_3by3_oldarrival_simple_linear"
# trainstep = 770
environment_name = r"simple_3by3_oldarrival"
# policy_name = "latest_stage"

csv_files = [
    r"val_mean_cor_finished.csv",
    r"val_mean_tot_finished.csv",
    r"val_mean_ord_cor_finished.csv",
    r"val_mean_plan_cor_finished.csv",
    r"val_mean_reward.csv",
    r"train_episode_mean_reward.csv",
]
csv_plot_titles = [
    "Validation: mean correctly finished workpieces",
    "Validation: mean episode reward",
    "Training: mean episode reward",
]

max_steps = 350


# automated settings
custom_path_interesting_runs = custom_trained_models_path

csv_path_general_low = (
    Path(custom_path_interesting_runs) / model_folder_name_low / r"exported_csvs"
)
csv_path_general_high = (
    Path(custom_path_interesting_runs) / model_folder_name_high / r"exported_csvs"
)
csv_path_general_good = (
    Path(custom_path_interesting_runs) / model_folder_name_good / r"exported_csvs"
)

df_Nfsco_low = generate_df_from_csv(
    csv_path=csv_path_general_low / csv_files[0], max_steps=max_steps
)
df_Ntot_low = generate_df_from_csv(
    csv_path=csv_path_general_low / csv_files[1], max_steps=max_steps
)
df_Nco_low = generate_df_from_csv(
    csv_path=csv_path_general_low / csv_files[2], max_steps=max_steps
)
df_Nfs_low = generate_df_from_csv(
    csv_path=csv_path_general_low / csv_files[3], max_steps=max_steps
)
df_val_reward_low = generate_df_from_csv(
    csv_path=csv_path_general_low / csv_files[4], max_steps=max_steps
)
df_train_reward_low = generate_df_from_csv(
    csv_path=csv_path_general_low / csv_files[5], max_steps=max_steps
)


df_Nfsco_high = generate_df_from_csv(
    csv_path=csv_path_general_high / csv_files[0], max_steps=max_steps
)
df_Ntot_high = generate_df_from_csv(
    csv_path=csv_path_general_high / csv_files[1], max_steps=max_steps
)
df_Nco_high = generate_df_from_csv(
    csv_path=csv_path_general_high / csv_files[2], max_steps=max_steps
)
df_Nfs_high = generate_df_from_csv(
    csv_path=csv_path_general_high / csv_files[3], max_steps=max_steps
)
df_val_reward_high = generate_df_from_csv(
    csv_path=csv_path_general_high / csv_files[4], max_steps=max_steps
)
df_train_reward_high = generate_df_from_csv(
    csv_path=csv_path_general_high / csv_files[5], max_steps=max_steps
)


df_Nfsco_good = generate_df_from_csv(
    csv_path=csv_path_general_good / csv_files[0], max_steps=max_steps
)
df_Ntot_good = generate_df_from_csv(
    csv_path=csv_path_general_good / csv_files[1], max_steps=max_steps
)
df_Nco_good = generate_df_from_csv(
    csv_path=csv_path_general_good / csv_files[2], max_steps=max_steps
)
df_Nfs_good = generate_df_from_csv(
    csv_path=csv_path_general_good / csv_files[3], max_steps=max_steps
)
df_val_reward_good = generate_df_from_csv(
    csv_path=csv_path_general_good / csv_files[4], max_steps=max_steps
)
df_train_reward_good = generate_df_from_csv(
    csv_path=csv_path_general_good / csv_files[5], max_steps=max_steps
)


# Assume all DataFrames have the same structure: [Step, Value]
step_col = df_Nco_low.columns[0]

# 1. Compute component-level counts
df_Nnfsco_low = df_Nco_low.copy()
df_Nnfsco_low.iloc[:, 1] = df_Nco_low.iloc[:, 1] - df_Nfsco_low.iloc[:, 1]

df_Nfswo_low = df_Nfs_low.copy()
df_Nfswo_low.iloc[:, 1] = df_Nfs_low.iloc[:, 1] - df_Nfsco_low.iloc[:, 1]

df_Nnfswo_low = df_Ntot_low.copy()
df_Nnfswo_low.iloc[:, 1] = (
    df_Ntot_low.iloc[:, 1]
    - df_Nfswo_low.iloc[:, 1]
    - df_Nnfsco_low.iloc[:, 1]
    - df_Nfsco_low.iloc[:, 1]
)

df_Nnfs_low = df_Ntot_low.copy()
df_Nnfs_low.iloc[:, 1] = df_Ntot_low.iloc[:, 1] - df_Nfs_low.iloc[:, 1]

df_Nwo_low = df_Ntot_low.copy()
df_Nwo_low.iloc[:, 1] = df_Ntot_low.iloc[:, 1] - df_Nco_low.iloc[:, 1]


df_counts_all_low = [
    df_Nfsco_low,
    df_Nfswo_low,
    df_Nnfsco_low,
    df_Nnfswo_low,
    df_Ntot_low,
    df_Nco_low,
    df_Nfs_low,
    df_Nwo_low,
    df_Nnfs_low,
]
df_titles_all_low = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
# fig = plot_dfs(df_counts_all_low, df_titles_all_low, csv_path_general_low, "all_low", beta_step=90)

df_counts_relevant_1_low = [df_Nfsco_low, df_Ntot_low, df_Nco_low, df_Nfs_low]
df_titles_relevant_1_low = [r"N_{FS, CO}", r"N_{out}", r"N_{CO}", r"N_{FS}"]
# fig = plot_dfs(df_counts_relevant_1_low, df_titles_relevant_1_low, csv_path_general_low, "relevant_1_low", beta_step=90)

df_counts_relevant_2_low = [df_Nfsco_low, df_Ntot_low, df_Nwo_low, df_Nnfs_low]
df_titles_relevant_2_low = [r"N_{FS, CO}", r"N_{out}", r"N_{WO}", r"N_{NFS}"]
# # fig = plot_dfs(df_counts_relevant_2_low, df_titles_relevant_2_low, csv_path_general_low, "relevant_2_low", beta_step=90)

df_counts_least_low = [df_Nfsco_low, df_Ntot_low]
df_titles_least_low = [r"N_{FS, CO}", r"N_{out}"]
# fig = plot_dfs(df_counts_least_low, df_titles_least_low, csv_path_general_low, "least_low", beta_step=90)


df_counts_least_low = [df_val_reward_low, df_train_reward_low]
df_titles_least_low = [
    r"\textrm{Validation mean reward}",
    r"\textrm{Training mean reward}",
]
# fig = plot_dfs(df_counts_least_low, df_titles_least_low, csv_path_general_low, "reward_low", beta_step=90)


# Assume all DataFrames have the same structure: [Step, Value]
step_col = df_Nco_high.columns[0]

# 1. Compute component-level counts
df_Nnfsco_high = df_Nco_high.copy()
df_Nnfsco_high.iloc[:, 1] = df_Nco_high.iloc[:, 1] - df_Nfsco_high.iloc[:, 1]

df_Nfswo_high = df_Nfs_high.copy()
df_Nfswo_high.iloc[:, 1] = df_Nfs_high.iloc[:, 1] - df_Nfsco_high.iloc[:, 1]

df_Nnfswo_high = df_Ntot_high.copy()
df_Nnfswo_high.iloc[:, 1] = (
    df_Ntot_high.iloc[:, 1]
    - df_Nfswo_high.iloc[:, 1]
    - df_Nnfsco_high.iloc[:, 1]
    - df_Nfsco_high.iloc[:, 1]
)

df_Nnfs_high = df_Ntot_high.copy()
df_Nnfs_high.iloc[:, 1] = df_Ntot_high.iloc[:, 1] - df_Nfs_high.iloc[:, 1]

df_Nwo_high = df_Ntot_high.copy()
df_Nwo_high.iloc[:, 1] = df_Ntot_high.iloc[:, 1] - df_Nco_high.iloc[:, 1]


df_counts_all_high = [
    df_Nfsco_high,
    df_Nfswo_high,
    df_Nnfsco_high,
    df_Nnfswo_high,
    df_Ntot_high,
    df_Nco_high,
    df_Nfs_high,
    df_Nwo_high,
    df_Nnfs_high,
]
df_titles_all_high = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
# fig = plot_dfs(df_counts_all_high, df_titles_all_high, csv_path_general_high, "all_high", beta_step=90)

df_counts_relevant_1_high = [df_Nfsco_high, df_Ntot_high, df_Nco_high, df_Nfs_high]
df_titles_relevant_1_high = [r"N_{FS, CO}", r"N_{out}", r"N_{CO}", r"N_{FS}"]
# fig = plot_dfs(df_counts_relevant_1_high, df_titles_relevant_1_high, csv_path_general_high, "relevant_1_high", beta_step=90)

df_counts_relevant_2_high = [df_Nfsco_high, df_Ntot_high, df_Nwo_high, df_Nnfs_high]
df_titles_relevant_2_high = [r"N_{FS, CO}", r"N_{out}", r"N_{WO}", r"N_{NFS}"]
# fig = plot_dfs(df_counts_relevant_2_high, df_titles_relevant_2_high, csv_path_general_high, "relevant_2_high", beta_step=90)

df_counts_least_high = [df_Nfsco_high, df_Ntot_high]
df_titles_least_high = [r"N_{FS, CO}", r"N_{out}"]
# fig = plot_dfs(df_counts_least_high, df_titles_least_high, csv_path_general_high, "least_high", beta_step=90)


df_counts_least_high = [df_val_reward_high, df_train_reward_high]
df_titles_least_high = [
    r"\textrm{Validation mean reward}",
    r"\textrm{Training mean reward}",
]
# fig = plot_dfs(df_counts_least_high, df_titles_least_high, csv_path_general_high, "reward_high", beta_step=90)


# Assume all DataFrames have the same structure: [Step, Value]
step_col = df_Nco_high.columns[0]

# 1. Compute component-level counts
df_Nnfsco_good = df_Nco_good.copy()
df_Nnfsco_good.iloc[:, 1] = df_Nco_good.iloc[:, 1] - df_Nfsco_good.iloc[:, 1]

df_Nfswo_good = df_Nfs_good.copy()
df_Nfswo_good.iloc[:, 1] = df_Nfs_good.iloc[:, 1] - df_Nfsco_good.iloc[:, 1]

df_Nnfswo_good = df_Ntot_good.copy()
df_Nnfswo_good.iloc[:, 1] = (
    df_Ntot_good.iloc[:, 1]
    - df_Nfswo_good.iloc[:, 1]
    - df_Nnfsco_good.iloc[:, 1]
    - df_Nfsco_good.iloc[:, 1]
)

df_Nnfs_good = df_Ntot_good.copy()
df_Nnfs_good.iloc[:, 1] = df_Ntot_good.iloc[:, 1] - df_Nfs_good.iloc[:, 1]

df_Nwo_good = df_Ntot_good.copy()
df_Nwo_good.iloc[:, 1] = df_Ntot_good.iloc[:, 1] - df_Nco_good.iloc[:, 1]


df_counts_all_good = [
    df_Nfsco_good,
    df_Nfswo_good,
    df_Nnfsco_good,
    df_Nnfswo_good,
    df_Ntot_good,
    df_Nco_good,
    df_Nfs_good,
    df_Nwo_good,
    df_Nnfs_good,
]
df_titles_all_good = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
fig = plot_dfs(
    df_counts_all_good,
    df_titles_all_good,
    csv_path_general_good,
    "all_good",
    beta_step=90,
)

df_counts_relevant_1_good = [df_Nfsco_good, df_Ntot_good, df_Nco_good, df_Nfs_good]
df_titles_relevant_1_good = [r"N_{FS, CO}", r"N_{out}", r"N_{CO}", r"N_{FS}"]
# fig = plot_dfs(df_counts_relevant_1_good, df_titles_relevant_1_good, csv_path_general_good, "relevant_1_good", beta_step=90)

df_counts_relevant_2_good = [df_Nfsco_good, df_Ntot_good, df_Nwo_good, df_Nnfs_good]
df_titles_relevant_2_good = [r"N_{FS, CO}", r"N_{out}", r"N_{WO}", r"N_{NFS}"]
# fig = plot_dfs(df_counts_relevant_2_good, df_titles_relevant_2_good, csv_path_general_good, "relevant_2_good", beta_step=90)

df_counts_least_good = [df_Nfsco_good, df_Ntot_good]
df_titles_least_good = [r"N_{FS, CO}", r"N_{out}"]
# fig = plot_dfs(df_counts_least_good, df_titles_least_good, csv_path_general_good, "least_good", beta_step=90)


df_counts_least_good = [df_val_reward_good, df_train_reward_good]
df_titles_least_good = [
    r"\textrm{Validation mean reward}",
    r"\textrm{Training mean reward}",
]
# fig = plot_dfs(df_counts_least_good, df_titles_least_good, csv_path_general_good, "reward_good", beta_step=90)


# plt.show()

# plot training reward comparison for low plateau, good and high plateau
all_train_rewards = [df_train_reward_low, df_train_reward_high, df_train_reward_good]
all_labels = [r"\textrm{CNN}", r"\textrm{Transformer}", r"\textrm{FNN}"]

plot_dfs(
    df_counts=all_train_rewards,
    df_titles=all_labels,
    save_path=csv_path_general_good,
    title="reward_comparison_networks",
    beta_step=None,
    beta_label="",
    y_label="Mean reward",
)


# plot validation metrics comparison for low plateau, good and high plateau
# all out and correct exits
all_val_metrics_relevant = [
    df_Ntot_low,
    df_Nfsco_low,
    df_Ntot_high,
    df_Nfsco_high,
    df_Ntot_good,
    df_Nfsco_good,
]
# all_labels_relevant = [r"\textrm{CNN}~N_{out}", r"\textrm{CNN}~N_{FS, CO}", r"\textrm{Transf.}~N_{out}", r"\textrm{Transf.}~N_{FS, CO}", r"\textrm{FNN}~N_{out}",   r"\textrm{FNN}~N_{FS, CO}"]
all_labels_relevant = [
    r"\textrm{CNN}~\overline{N}_{out}",
    r"\textrm{CNN}~\overline{N}_{FS, CO}",
    r"\textrm{Transf.}~\overline{N}_{out}",
    r"\textrm{Transf.}~\overline{N}_{FS, CO}",
    r"\textrm{FNN}~\overline{N}_{out}",
    r"\textrm{FNN}~\overline{N}_{FS, CO}",
]

plot_dfs_dotted(
    df_counts=all_val_metrics_relevant,
    df_titles=all_labels_relevant,
    save_path=csv_path_general_good,
    title="val_metrics_networks",
    beta_step=None,
    beta_label=r"",
)


plt.show()
