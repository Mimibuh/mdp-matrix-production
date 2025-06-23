from pathlib import Path
import pandas as pd

from matplotlib import pyplot as plt
import tol_colors as tc


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


def plot_dfs(df_counts, df_titles, save_path, title=None):
    # Prepare full filename
    filename = save_path / f"{title}.pdf"

    # Plot settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig, ax = plt.subplots(figsize=(8, 6))

    for df, label in zip(df_counts, df_titles):
        step_col = df.columns[0]
        value_col = df.columns[1]

        df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        ax.plot(df[step_col], df[value_col], label=rf"${label}$", linewidth=2.0)

    ax.set_xlabel("Step", fontsize=14, fontweight="bold")
    ax.set_ylabel("Workpiece Count", fontsize=14, fontweight="bold")

    if title:
        ax.set_title(
            title.replace("_", " ").title(), fontsize=16, fontweight="bold"
        )  # optional

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=12, frameon=False)
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout(pad=2.0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Save figure as high-quality vector graphic
    fig.savefig(filename, bbox_inches="tight")
    # plt.close(fig)

    return fig


# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
model_folder_name = r"3x3_oldLowIn_simple_3by3_01_03_run_arrival_change_401_006_transformer_t_2025-04-27_21-36-07"
trainstep = 330
environment_name = r"simple_3by3_01_03"
policy_name = "latest_stage"

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

max_steps = 440


# automated settings
custom_path_interesting_runs = Path(
    r"C:\Users\mimib\Desktop\Masterarbeit Produktionsmanagement\interesting_runs"
)
csv_path_general = (
    Path(custom_path_interesting_runs) / model_folder_name / r"exported_csvs"
)

df_Nfsco = generate_df_from_csv(
    csv_path=csv_path_general / csv_files[0], max_steps=max_steps
)

df_Ntot = generate_df_from_csv(
    csv_path=csv_path_general / csv_files[1], max_steps=max_steps
)

df_Nco = generate_df_from_csv(
    csv_path=csv_path_general / csv_files[2], max_steps=max_steps
)

df_Nfs = generate_df_from_csv(
    csv_path=csv_path_general / csv_files[3], max_steps=max_steps
)

df_val_reward = generate_df_from_csv(
    csv_path=csv_path_general / csv_files[4], max_steps=max_steps
)

df_train_reward = generate_df_from_csv(
    csv_path=csv_path_general / csv_files[5], max_steps=max_steps
)


# Assume all DataFrames have the same structure: [Step, Value]
step_col = df_Nco.columns[0]

# 1. Compute component-level counts
df_Nnfsco = df_Nco.copy()
df_Nnfsco.iloc[:, 1] = df_Nco.iloc[:, 1] - df_Nfsco.iloc[:, 1]

df_Nfswo = df_Nfs.copy()
df_Nfswo.iloc[:, 1] = df_Nfs.iloc[:, 1] - df_Nfsco.iloc[:, 1]

df_Nnfswo = df_Ntot.copy()
df_Nnfswo.iloc[:, 1] = (
    df_Ntot.iloc[:, 1]
    - df_Nfswo.iloc[:, 1]
    - df_Nnfsco.iloc[:, 1]
    - df_Nfsco.iloc[:, 1]
)

df_Nnfs = df_Ntot.copy()
df_Nnfs.iloc[:, 1] = df_Ntot.iloc[:, 1] - df_Nfs.iloc[:, 1]

df_Nwo = df_Ntot.copy()
df_Nwo.iloc[:, 1] = df_Ntot.iloc[:, 1] - df_Nco.iloc[:, 1]


df_counts_all = [
    df_Nfsco,
    df_Nfswo,
    df_Nnfsco,
    df_Nnfswo,
    df_Ntot,
    df_Nco,
    df_Nfs,
    df_Nwo,
    df_Nnfs,
]
df_titles_all = [
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
fig = plot_dfs(df_counts_all, df_titles_all, csv_path_general, "all")

df_counts_relevant_1 = [df_Nfsco, df_Ntot, df_Nco, df_Nfs]
df_titles_relevant_1 = [r"N_{FS, CO}", r"N_{out}", r"N_{CO}", r"N_{FS}"]
fig = plot_dfs(
    df_counts_relevant_1, df_titles_relevant_1, csv_path_general, "relevant_1"
)

df_counts_relevant_2 = [df_Nfsco, df_Ntot, df_Nwo, df_Nnfs]
df_titles_relevant_2 = [r"N_{FS, CO}", r"N_{out}", r"N_{WO}", r"N_{NFS}"]
fig = plot_dfs(
    df_counts_relevant_2, df_titles_relevant_2, csv_path_general, "relevant_2"
)

df_counts_least = [df_Nfsco, df_Ntot]
df_titles_least = [r"N_{FS, CO}", r"N_{out}"]
fig = plot_dfs(df_counts_least, df_titles_least, csv_path_general, "least")


df_counts_least = [df_val_reward, df_train_reward]
df_titles_least = [r"\textrm{Validation mean reward}", r"\textrm{Training mean reward}"]
fig = plot_dfs(df_counts_least, df_titles_least, csv_path_general, "reward")


plt.show()
