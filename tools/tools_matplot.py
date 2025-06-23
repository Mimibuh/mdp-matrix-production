import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ray.rllib.algorithms.utils import torch
from train.config import ENV_CONFIGS
from train.environment import Matrix_production
from matplotlib.offsetbox import AnchoredText


import tol_colors as tc

"""
general plotting tools for matrix production environment
"""

# Set default color palette for matplotlib (use 'bright', 'muted', etc.)
tc.set_default_colors(cset="muted")


def plot_system(
    *,
    ax,
    grid_size,
    arrival_interval,
    arrival_pattern,
    state_grid_before,
    state_grid_after,
    queue_before,
    queue_after,
    step_reward,
    machine_abilities_grid,
    env_name,
):

    border_width = 0.25  # Thickness of the border inside each cell
    color_buffer_or_failed = "darkred"
    color_available = "lightgrey"

    color_no_wp = "lightgrey"

    number_of_pgs = len(ENV_CONFIGS[env_name]["product_groups"])
    colors_pg = plt.cm.viridis(np.linspace(0.2, 0.7, number_of_pgs))

    rows, cols = grid_size
    ax.set_xlim(-2, cols + 3)
    ax.set_ylim(0, rows * 2 + 2.5)
    # axes[0].set_ylim(-1, grid_rows * 2 + 4)
    ax.set_xticks([])
    ax.set_yticks([])

    # Ensure the spines (borders) are visible
    for spine in ax.spines.values():
        spine.set_visible(True)
    # ax.set_aspect("equal")  # Ensure square cells

    states = [state_grid_after, state_grid_before]

    state_row_offset = rows + 1

    m = 0

    info_text = [
        "move workpieces\n no failure/arrival yet",
        "new state\n(after failure/arrival)",
    ]
    # plot grid
    for i in range(2):

        ax.text(
            -1,
            ((state_row_offset * i + 1) + (rows - 1 + state_row_offset * i + 1)) / 2,
            info_text[i],
            fontsize=8,
            ha="center",
            va="center",
        )

        for row in range(rows):
            row_reverse = abs(
                row - rows + 1
            )  # because plooting starts with bottom row, but state, action etc starts with top row
            for col in range(cols):
                edge_color = (
                    color_buffer_or_failed
                    if states[i][row_reverse, col, 0] == 1
                    else color_available
                )
                center_color = (
                    colors_pg[int(states[i][row_reverse, col, 3])]
                    if states[i][row_reverse, col, 1] == 1
                    else color_no_wp
                )

                marker_size = 45
                ax.plot(
                    col + 2,
                    row + state_row_offset * i + 1,
                    marker="s",
                    markersize=45,
                    markerfacecolor=center_color,
                    markeredgewidth=6,
                    markeredgecolor=edge_color,
                )

                text_positions = [
                    (
                        col + 2 - 0.004 * marker_size,
                        row + state_row_offset * i + 1 + 0.007 * marker_size,
                    ),  # Top-left
                    (
                        col + 2 + 0.004 * marker_size,
                        row + state_row_offset * i + 1 + 0.007 * marker_size,
                    ),  # Top-right
                    (
                        col + 2 - 0.004 * marker_size,
                        row + state_row_offset * i + 1 - 0.007 * marker_size,
                    ),  # Bottom-left
                    (
                        col + 2 + 0.004 * marker_size,
                        row + state_row_offset * i + 1 - 0.007 * marker_size,
                    ),  # Bottom-right
                    (
                        col + 2 - 0.004 * marker_size,
                        (row + state_row_offset * i + 1),
                    ),  # Middle-left
                ]

                text_content = [
                    (
                        f"id:{int(states[i][row_reverse, col, 2])}"
                        if states[i][row_reverse, col, 1] == 1
                        else ""
                    ),  # id
                    (
                        f"pg:{int(states[i][row_reverse, col, 3])}"
                        if states[i][row_reverse, col, 1] == 1
                        else ""
                    ),  # pg
                    (
                        f"d:{int(states[i][row_reverse, col, 4])}"
                        if states[i][row_reverse, col, 1] == 1
                        else ""
                    ),  # index done
                    (
                        f"rs:{int(states[i][row_reverse, col, 5])}"
                        if states[i][row_reverse, col, 1] == 1
                        else ""
                    ),  # remaining service time
                    (
                        f"ab:{int(machine_abilities_grid[row_reverse, col])}"
                    ),  # machine abilities
                ]

                for z, pos in enumerate(text_positions):
                    ax.text(
                        pos[0],
                        pos[1],
                        text_content[z],
                        fontsize=8,
                        ha="center",
                        va="center",
                    )
                m += 1

    # plot queue and finish area
    queues = [queue_after, queue_before]
    for i in range(2):
        edge_color = "darkgrey"
        center_color = (
            colors_pg[int(queues[i][1])] if queues[i][0] >= 1 else color_no_wp
        )

        x_queue = 1
        y_queue = (
            (state_row_offset * i + 1) + (rows - 1 + state_row_offset * i + 1)
        ) / 2

        # queue
        ax.plot(
            x_queue,
            y_queue,
            marker="h",
            markersize=marker_size,
            markerfacecolor=center_color,
            markeredgewidth=6,
            markeredgecolor=edge_color,
        )

        ax.text(
            x_queue,
            y_queue,
            # f"pg:{int(queues[i][1])}" if queues[i][0] == 1 else "",
            f"no:{int(queues[i][0])}, pg:{int(queues[i][1])}, t:{int(queues[i][2])}, i_n:{int(queues[i][3])}]",
            fontsize=8,
            ha="center",
            va="center",
        )

        # finish
        ax.plot(
            1 + cols + 1,
            ((state_row_offset * i + 1) + (rows - 1 + state_row_offset * i + 1)) / 2,
            marker="H",
            markersize=45,
            markerfacecolor=edge_color,
            markeredgewidth=6,
            markeredgecolor=edge_color,
        )

    current_timestep, index_next = int(queue_before[2]), int(queue_before[3])
    try_arrival = current_timestep == arrival_interval - 1
    anchored_text = AnchoredText(
        # f"try_arrival: {try_arrival}       next_pg: {arrival_pattern[index_next]}
        f"reward: {step_reward}",
        loc="upper right",
        prop=dict(size=8),
        frameon=True,
    )
    ax.add_artist(anchored_text)


def draw_action_arrows(
    *, ax, grid_size, numpy_action_decoding, action_grid, action_queue
):
    # print("action_grid", action_grid)

    rows, cols = grid_size
    state_row_offset = rows + 1

    finish_x_y_after = (
        1 + cols + 1,
        ((state_row_offset + 1) + (rows - 1 + state_row_offset + 1)) / 2
        - state_row_offset,
    )

    choose_color = {0: "grey", 1: "blue"}
    for row in range(action_grid.shape[0]):
        row_reverse = abs(
            row - rows + 1
        )  # because plotting starts with bottom row, but state, action etc starts with top row
        for col in range(action_grid.shape[1]):
            a = int(action_grid[row_reverse, col, 0])
            instr = numpy_action_decoding[a][0]
            if instr >= 0:

                next_loc_row_col = (
                    numpy_action_decoding[a][1],
                    numpy_action_decoding[a][2],
                )
                if next_loc_row_col == (-2, -2):  # finished
                    ax.annotate(
                        "",
                        xy=(finish_x_y_after[0], finish_x_y_after[1]),
                        xytext=(col + 2, row + state_row_offset + 1),
                        arrowprops=dict(
                            arrowstyle="->", color=choose_color[instr], lw=2
                        ),
                    )
                else:  # stay in grid
                    ax.annotate(
                        "",
                        xy=(
                            numpy_action_decoding[a][2] + 2,
                            abs(numpy_action_decoding[a][1] - rows + 1) + 1,
                        ),
                        xytext=(col + 2, row + state_row_offset + 1),
                        arrowprops=dict(
                            arrowstyle="->", color=choose_color[instr], lw=2
                        ),
                    )

    # Add arrow for the action_queue, using a special queue cell position
    queue_cell_x_y_before = (
        1,
        ((state_row_offset + 1) + (rows - 1 + state_row_offset + 1)) / 2,
    )

    a = int(action_queue)
    next_loc_row_col = (numpy_action_decoding[a][1], numpy_action_decoding[a][2])

    ax.text(
        queue_cell_x_y_before[0],
        queue_cell_x_y_before[1],
        # f"pg:{int(queues[i][1])}" if queues[i][0] == 1 else "",
        f"\n\naction: {numpy_action_decoding[a]}",
        fontsize=8,
        ha="center",
        va="center",
    )

    instr = numpy_action_decoding[a][0]
    if instr >= 0:
        if next_loc_row_col == (-1, -1):  # stay in queue
            ax.annotate(
                "",
                xy=(
                    queue_cell_x_y_before[0],
                    queue_cell_x_y_before[1] - state_row_offset,
                ),
                xytext=(queue_cell_x_y_before[0], queue_cell_x_y_before[1]),
                arrowprops=dict(arrowstyle="->", color=choose_color[instr], lw=2),
            )
        elif next_loc_row_col == (-2, -2):  # finished
            ax.annotate(
                "",
                xy=(finish_x_y_after[0], finish_x_y_after[1]),
                xytext=(queue_cell_x_y_before[0], queue_cell_x_y_before[1]),
                arrowprops=dict(arrowstyle="->", color=choose_color[instr], lw=2),
            )
        else:  # stay in grid
            ax.annotate(
                "",
                xy=(next_loc_row_col[1] + 2, abs(next_loc_row_col[0] - rows + 1) + 1),
                xytext=(queue_cell_x_y_before[0], queue_cell_x_y_before[1]),
                arrowprops=dict(arrowstyle="->", color=choose_color[instr], lw=2),
            )


def plot_random_steps_rl_module(
    *, num_plot_steps, environment_name, rl_module, start=None, seed=1
):
    env_config = ENV_CONFIGS[environment_name]

    dummy_env = Matrix_production(env_config)
    dummy_env.reset(seed=seed)  # resets with random state

    # choose random number between 0 and (env_config.max_steps-1)-num_plot_steps
    if start is None:
        start = np.random.randint(
            0, (env_config["max_steps"] - 2) - (num_plot_steps + 1)
        )
    end = start + num_plot_steps + 1

    obs = np.array(dummy_env.state)
    grid_rows = env_config["grid_size"][0]
    grid_cols = env_config["grid_size"][1]
    arrival_interval = env_config["arrival_interval"]
    arrival_pattern = env_config["arrival_pattern"]
    machine_abilities = env_config["machine_abilities"]
    machine_abilities_grid = dummy_env.list_to_matrix(machine_abilities, 1)

    collect_obs_grids = np.zeros((end + 1, grid_rows, grid_cols, 6))
    collect_obs_queues = np.zeros((end + 1, 4))
    collect_interstate_grids = np.zeros((end + 1, grid_rows, grid_cols, 6))
    collect_interstate_queues = np.zeros((end + 1, 4))

    collect_action_grids = np.zeros((end + 1, grid_rows, grid_cols, 1))
    collect_action_queues = np.zeros(end + 1)
    collect_rewards = np.zeros(end + 1)

    for step in range(end + 1):
        assert np.array_equal(obs[:-4], dummy_env.get_grid_state())

        collect_obs_grids[step] = dummy_env.list_to_matrix(dummy_env.get_grid_state())
        collect_obs_queues[step] = dummy_env.get_queue_state()
        obs_batch = np.array([obs], dtype=np.float32)
        results = rl_module._pi(
            torch.tensor(obs_batch, dtype=torch.float32), inference=False
        )

        action = np.array(results["actions"][0])
        env_action = dummy_env.decode_action_network_to_env(action)
        collect_action_grids[step] = dummy_env.list_to_matrix(list(env_action[:-1]), 1)
        collect_action_queues[step] = env_action[-1]

        obs, reward, done, trunc, info = dummy_env.step(action)
        interstate_grid = info["interstate_grid_before_arrival_failure"]
        interstate_queue = info["interstate_queue_before_arrival_failure"]
        collect_interstate_grids[step] = interstate_grid
        collect_interstate_queues[step] = interstate_queue

        collect_rewards[step] = reward

    chosen_obs_grids = collect_obs_grids[start:end]
    chosen_obs_queue = collect_obs_queues[start:end]
    chosen_interstate_grids = collect_interstate_grids[start:end]
    chosen_interstate_queues = collect_interstate_queues[start:end]
    chosen_act_grids = collect_action_grids[start:end]
    chosen_act_queues = collect_action_queues[start:end]
    chosen_rewards = collect_rewards[start:end]

    # fig, axes = plt.subplots(1, num_plot_steps, figsize=(9 * grid_cols, 3.5 * grid_rows))
    fig, axes = plt.subplots(
        num_plot_steps, 1, figsize=(2 * grid_cols + 2, 12 * grid_rows)
    )

    for plot_step in range(num_plot_steps):
        plot_system(
            ax=axes[plot_step],
            grid_size=(grid_rows, grid_cols),
            arrival_interval=arrival_interval,
            arrival_pattern=arrival_pattern,
            state_grid_before=chosen_obs_grids[plot_step],
            state_grid_after=chosen_interstate_grids[plot_step],
            queue_before=chosen_obs_queue[plot_step],
            queue_after=chosen_interstate_queues[plot_step],
            step_reward=chosen_rewards[plot_step],
            machine_abilities_grid=machine_abilities_grid,
            env_name=environment_name,
        )

        draw_action_arrows(
            ax=axes[plot_step],
            grid_size=(grid_rows, grid_cols),
            numpy_action_decoding=dummy_env.numpy_action_decoding,
            action_grid=chosen_act_grids[plot_step],
            action_queue=chosen_act_queues[plot_step],
        )

    fig.suptitle(f"RLModule Trajectory in {environment_name} (Steps {start}–{end - 1})")
    plt.tight_layout()

    return fig


def plot_random_steps_own_policy(
    *, num_plot_steps, environment_name, policy, start=None, seed=1
):
    env_config = ENV_CONFIGS[environment_name]

    dummy_env = Matrix_production(env_config)
    dummy_env.reset(seed=seed)  # resets with random state

    # choose random number between 0 and (env_config.max_steps-1)-num_plot_steps
    if start is None:
        start = np.random.randint(
            0, (env_config["max_steps"] - 2) - (num_plot_steps + 1)
        )
    end = start + num_plot_steps + 1

    obs = np.array(dummy_env.state)
    grid_rows = env_config["grid_size"][0]
    grid_cols = env_config["grid_size"][1]
    arrival_interval = env_config["arrival_interval"]
    arrival_pattern = env_config["arrival_pattern"]
    machine_abilities = env_config["machine_abilities"]
    machine_abilities_grid = dummy_env.list_to_matrix(machine_abilities, 1)

    collect_obs_grids = np.zeros((end + 1, grid_rows, grid_cols, 6))
    collect_obs_queues = np.zeros((end + 1, 4))
    collect_interstate_grids = np.zeros((end + 1, grid_rows, grid_cols, 6))
    collect_interstate_queues = np.zeros((end + 1, 4))

    collect_action_grids = np.zeros((end + 1, grid_rows, grid_cols, 1))
    collect_action_queues = np.zeros(end + 1)
    collect_rewards = np.zeros(end + 1)

    for step in range(end + 1):
        assert np.array_equal(obs[:-4], dummy_env.get_grid_state())

        collect_obs_grids[step] = dummy_env.list_to_matrix(dummy_env.get_grid_state())
        collect_obs_queues[step] = dummy_env.get_queue_state()
        # obs_batch = np.array([obs], dtype=np.float32)

        action = policy.compute_action(obs)
        env_action = dummy_env.decode_action_network_to_env(action)

        collect_action_grids[step] = dummy_env.list_to_matrix(list(env_action[:-1]), 1)
        collect_action_queues[step] = env_action[-1]

        obs, reward, done, trunc, info = dummy_env.step(action)
        interstate_grid = info["interstate_grid_before_arrival_failure"]
        interstate_queue = info["interstate_queue_before_arrival_failure"]
        collect_interstate_grids[step] = interstate_grid
        collect_interstate_queues[step] = interstate_queue

        collect_rewards[step] = reward

    chosen_obs_grids = collect_obs_grids[start:end]
    chosen_obs_queue = collect_obs_queues[start:end]
    chosen_interstate_grids = collect_interstate_grids[start:end]
    chosen_interstate_queues = collect_interstate_queues[start:end]
    chosen_act_grids = collect_action_grids[start:end]
    chosen_act_queues = collect_action_queues[start:end]
    chosen_rewards = collect_rewards[start:end]

    # fig, axes = plt.subplots(1, num_plot_steps, figsize=(9 * grid_cols, 3.5 * grid_rows))
    fig, axes = plt.subplots(
        num_plot_steps, 1, figsize=(2 * grid_cols + 2, 12 * grid_rows)
    )

    for plot_step in range(num_plot_steps):
        plot_system(
            ax=axes[plot_step],
            grid_size=(grid_rows, grid_cols),
            arrival_interval=arrival_interval,
            arrival_pattern=arrival_pattern,
            state_grid_before=chosen_obs_grids[plot_step],
            state_grid_after=chosen_interstate_grids[plot_step],
            queue_before=chosen_obs_queue[plot_step],
            queue_after=chosen_interstate_queues[plot_step],
            step_reward=chosen_rewards[plot_step],
            machine_abilities_grid=machine_abilities_grid,
            env_name=environment_name,
        )

        draw_action_arrows(
            ax=axes[plot_step],
            grid_size=(grid_rows, grid_cols),
            numpy_action_decoding=dummy_env.numpy_action_decoding,
            action_grid=chosen_act_grids[plot_step],
            action_queue=chosen_act_queues[plot_step],
        )

    # axes[0].set_xlim(0, 10)

    # Adjust layout
    fig.suptitle(f"Policy Rollout in {environment_name} (Steps {start}–{end - 1})")
    plt.tight_layout()

    return fig


def compare_steps_rl_module_policy(
    *, start, num_plot_steps, environment_name, rl_module, own_policy, seed=1
):

    fig_rl = plot_random_steps_rl_module(
        num_plot_steps=num_plot_steps,
        environment_name=environment_name,
        rl_module=rl_module,
        start=start,
        seed=seed,
    )
    fig_pol = plot_random_steps_own_policy(
        num_plot_steps=num_plot_steps,
        environment_name=environment_name,
        policy=own_policy,
        start=start,
        seed=seed,
    )

    return fig_rl, fig_pol


def plot_selected_columns_from_csv(
    csv_path,
    column_indices,
    legend_labels,
    title="Plot of Selected Metrics",
    max_steps=None,
):
    df = pd.read_csv(csv_path)

    # Debugging: print the original number of rows
    print(f"Original DataFrame size for {csv_path}: {df.shape[0]} rows")

    # Detect the step column
    step_col_candidates = [col for col in df.columns if col.lower() == "step"]
    if not step_col_candidates:
        raise ValueError("No column named 'Step' found (case insensitive).")
    step_col = step_col_candidates[0]

    # Get all column names excluding Step
    value_cols = [col for col in df.columns if col != step_col]

    # Validate input
    if len(column_indices) != len(legend_labels):
        raise ValueError("column_indices and legend_labels must have the same length.")

    # Apply max_steps if provided to limit the number of steps plotted
    if max_steps is not None:
        df = df[df[step_col] <= max_steps]
        # Debugging: print the number of rows after applying max_steps
        print(
            f"Sliced DataFrame size for {csv_path} with max_steps={max_steps}: {df.shape[0]} rows"
        )

    # Set Matplotlib to use LaTeX rendering
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig, ax = plt.subplots(figsize=(8, 6))

    # scientific line styles
    for idx, label in zip(column_indices, legend_labels):
        if idx < 0 or idx >= len(value_cols):
            raise IndexError(
                f"Column index {idx} out of range (0 to {len(value_cols)-1})"
            )

        col_name = value_cols[idx]
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        ax.plot(
            df[step_col],
            df[col_name],
            label=label,
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=5,
        )

    # Styling for a more scientific look
    ax.set_xlabel(
        "Step", fontsize=14, fontweight="bold", family="serif"
    )  # Bold, readable labels
    ax.set_ylabel("Value", fontsize=14, fontweight="bold", family="serif")
    ax.set_title(
        title, fontsize=16, fontweight="bold", family="serif"
    )  # Title font size increased for readability
    ax.grid(
        True, linestyle="--", linewidth=0.5, alpha=0.6
    )  # Lighter grid lines for clarity
    ax.legend(
        loc="best", fontsize=12, frameon=False
    )  # Legend outside the plot if needed
    ax.tick_params(
        axis="both", which="major", labelsize=12
    )  # Larger tick labels for readability
    fig.tight_layout(pad=2.0)  # Ensure tight layout

    # Additional minor adjustments for scientific publication
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right"
    )  # Rotate x-tick labels for better readability

    # Optionally save the figure as .png or .pdf
    # fig.savefig("plot.png", dpi=300)

    return fig


def plot_data_per_seed(
    *, baseline, datas, data_names, used_seeds, ylims=None, plot_title=None
):

    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a colormap with as many colors as there are data series.
    colors = plt.cm.viridis(np.linspace(0.2, 0.7, len(datas)))

    for i, data in enumerate(datas):
        ax.scatter(
            used_seeds, data, label=f"{data_names[i]}", alpha=1, color=colors[i], s=20
        )

    ax.plot(
        used_seeds,
        [baseline] * len(used_seeds),
        label="number_of_workpieces_arrived",
        linewidth=1.5,
        color="orange",
    )

    ax.set_xlabel("Seed")
    ax.set_ylabel("Value")

    if ylims:
        ax.set_ylim(ylims[0], ylims[1])

    ax.set_xlim(0, len(used_seeds) - 1)

    if plot_title:
        ax.set_title(plot_title)
    # else:
    # ax.set_title(f"{data_names} per seed")

    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend()
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    return fig
