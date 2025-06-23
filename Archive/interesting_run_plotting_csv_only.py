from pathlib import Path

from matplotlib import pyplot as plt

from tools.tools_matplot import plot_selected_columns_from_csv

# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
model_folder_name = r"3x3_LowIn_simple_3by3_01_03_run_arrival_change_401_006_transformer_t_2025-04-27_21-36-07"
trainstep = 330
environment_name = r"simple_3by3_01_03"
policy_name = "latest_stage"

csv_files = [
    r"val_mean_cor_finished.csv",
    r"val_mean_reward.csv",
    r"train_episode_mean_reward.csv",
]
csv_plot_legend_labels = [
    "val_num_correctly_finished",
    "val_episode_mean_reward",
    "train_episode_mean_reward",
]
csv_plot_titles = [
    "Validation: mean correctly finished workpieces",
    "Validation: mean episode reward",
    "Training: mean episode reward",
]


# automated settings
custom_path_interesting_runs = Path(
    r"C:\Users\mimib\Desktop\Masterarbeit Produktionsmanagement\interesting_runs"
)

for i in range(len(csv_files)):
    csv_path = (
        Path(custom_path_interesting_runs)
        / model_folder_name
        / r"exported_csvs"
        / csv_files[i]
    )
    fig = plot_selected_columns_from_csv(
        csv_path=csv_path,
        column_indices=[0],  # relative to non-Step columns
        legend_labels=[csv_plot_legend_labels[i]],
        title=csv_plot_titles[i],
        max_steps=440,
    )
    fig.show()


# Set default color palette for matplotlib (use 'bright', 'muted', etc.)


plt.show()
