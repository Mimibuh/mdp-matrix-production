from pathlib import Path

from matplotlib import pyplot as plt
import tol_colors as tc

from tools.global_config import custom_trained_models_path
from tools.tools_analyze_val import plot_dfs, generate_df_from_csv

# Set default color palette for matplotlib (use 'bright', 'muted', etc.)
tc.set_default_colors(cset="bright")
tc.Muted


# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
# model_folder_name = r"3x3HighIn_simple_3by3_oldarrival_run_arrival_change_201_007_simple_linear_t_2025-06-15_18-52-58"
model_folder_name = r"3x3HighIn_simple_3by3_oldarrival_simple_linear"

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
fig = plot_dfs(df_counts_all, df_titles_all, csv_path_general, "all", beta_step=90)

df_counts_relevant_1 = [df_Nfsco, df_Ntot, df_Nco, df_Nfs]
df_titles_relevant_1 = [r"N_{FS, CO}", r"N_{out}", r"N_{CO}", r"N_{FS}"]
fig = plot_dfs(
    df_counts_relevant_1,
    df_titles_relevant_1,
    csv_path_general,
    "relevant_1",
    beta_step=90,
)

df_counts_relevant_2 = [df_Ntot, df_Nfsco, df_Nnfs, df_Nwo]
df_titles_relevant_2 = [
    r"\overline{N}_{out}",
    r"\overline{N}_{FS, CO}",
    r"\overline{N}_{NFS}",
    r"\overline{N}_{WO}",
]
fig = plot_dfs(
    df_counts_relevant_2,
    df_titles_relevant_2,
    csv_path_general,
    "relevant_2",
    beta_step=90,
)

df_counts_least = [df_Nfsco, df_Ntot]
df_titles_least = [r"N_{FS, CO}", r"N_{out}"]
fig = plot_dfs(
    df_counts_least, df_titles_least, csv_path_general, "least", beta_step=90
)


df_counts_least = [df_train_reward, df_val_reward]
df_titles_least = [r"\textrm{Training}", r"\textrm{Validation}"]
fig = plot_dfs(
    df_counts_least,
    df_titles_least,
    csv_path_general,
    "reward",
    beta_step=90,
    y_label="Mean reward",
)


plt.show()
