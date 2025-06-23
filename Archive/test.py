# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:44:26 2025

@author: mimib
"""

from mimi_env import Desp
import numpy as np

daisy = Desp()

ad = daisy.action_decoding

grid_size = daisy.grid_size

print(daisy.action_space.nvec[0])

mb = daisy.machine_or_buffer

action_space = daisy.action_space

dic_valid_actions_by_remaining_stages = daisy.dic_valid_actions_by_remaining_stages

final_indices_by_pgs = daisy.final_indices_by_pgs

# print(ad[6][1])


def remap_orders(current_order, outgoings):
    descending_outgoings = sorted(outgoings, reverse=True)
    # Find elements in list1 that are also in list2 and remove them
    current_order = np.array(current_order)
    outgoings = np.array(outgoings)
    current_order = current_order[
        ~np.isin(current_order, descending_outgoings)
    ]  # Keep only elements NOT in list2
    print(current_order)

    remapping = {}
    i = 0
    for element in current_order:
        remapping[element] = i
        i += 1

    return remapping


print("remapping: ", remap_orders([0, 1, 2, 3, 4, 5, 8, 10], [3, 5]))


def downstream_masking():
    downstream_masks = []
    for y in range(grid_size[1] - 1, -1, -1):
        for x in range(grid_size[0] - 1, -1, -1):

            current = (x, y)

            possible_new = []
            for x_ in range(grid_size[0]):
                for y_ in range(y, grid_size[1], 1):
                    possible_new.append((x_, y_))

            possible_new.append((-2, -2))
            print(possible_new)

            single_head_mask = []

            keys_except_last = list(ad.keys())[:-1]

            for (
                key
            ) in keys_except_last:  # last action key stands for "noWP" not a location!
                if ad[key][1] in possible_new:
                    single_head_mask.append(0)
                else:
                    single_head_mask.append(-float("inf"))

            # append "okay" (=0) for last action as it indicates that no WP is here, has nothing to do with downstream constraint
            single_head_mask.append(0)

            # append head mask to all head masks (in correct order by for loops garantueed)
            downstream_masks.append(single_head_mask)

    # queue can do every action, append to downstream_masks, is the last head
    downstream_masks.append([0] * len(ad))

    for i in range(len(downstream_masks)):
        print("\nHead i=", i)
        for action in range(len(downstream_masks[i])):
            print(
                "Action: ",
                action,
                "   yes" if downstream_masks[i][action] == 0 else "   no",
            )


def wait_service_masking():
    number_of_heads = 5
    one_mask_for_all_heads = [0] * daisy.action_space.nvec[0]
    for i in range(len(mb)):
        if mb[i] == 0:
            one_mask_for_all_heads[(grid_size[0] * grid_size[1] + i)] = -float("inf")

    return [one_mask_for_all_heads] * number_of_heads


def machine_failure_masking(state):
    number_of_heads = 5
    one_mask_for_all_heads = [0] * daisy.action_space.nvec[0]
    machine_buffer_status = []

    i = 0
    while i <= grid_size[0] * grid_size[1] * 6:
        machine_buffer_status.append(state[i])
        i += 6

    for i in range(len(machine_buffer_status)):
        if machine_buffer_status[i] == 1:  # machine failure
            one_mask_for_all_heads[(grid_size[0] * grid_size[1] + i)] = -float("inf")

    return [one_mask_for_all_heads] * number_of_heads


import torch


def machine_failure_batch(observations: torch.Tensor) -> torch.Tensor:
    batch_size = observations.shape[0]
    obs_dim = observations.shape[1]

    # Determine machine buffer status: Extract every 6th value in the observation
    indices = torch.arange(
        0, grid_size[0] * grid_size[1] * 6, 6, device=observations.device
    )
    print("hier: ", indices)
    machine_buffer_status = observations[
        :, indices
    ]  # Shape: [batch_size, num_machines]
    print(machine_buffer_status)

    # Initialize mask for all actions
    general_mask = torch.zeros(
        batch_size, action_space.nvec[0], device=observations.device
    )

    # Identify failed machines and update mask
    machine_failure_indices = (machine_buffer_status == 1).nonzero(
        as_tuple=True
    )  # (batch_idx, machine_idx)
    grid_offset = grid_size[0] * grid_size[1]
    general_mask[
        machine_failure_indices[0], grid_offset + machine_failure_indices[1]
    ] = -float("inf")

    number_of_heads = 5
    # Duplicate the mask for all heads
    mask = general_mask.unsqueeze(1).expand(
        -1, number_of_heads, -1
    )  # Shape: [batch_size, number_of_heads, nvec]

    return mask


def previous_actions_mask(prev_actions):
    prev_act_mask = [0] * daisy.action_space.nvec[0]

    number_of_cells = grid_size[0] * grid_size[1]

    for action in prev_actions:
        prev_act_mask[action] = -float("inf")
        if action < number_of_cells:
            prev_act_mask[action + number_of_cells] = -float("inf")
        elif action < number_of_cells * 2:
            prev_act_mask[action - number_of_cells] = -float("inf")

    return prev_act_mask


def prev_actions_mask_batch(prev_actions) -> torch.Tensor:
    """
    Generate a mask to invalidate actions based on a list of previous actions for all heads in a batch.

    Args:
        prev_actions (List[torch.Tensor]): A list of tensors, where each tensor has shape [batch_size]
            and contains the previous actions sampled for a specific head across the batch.

    Returns:
        torch.Tensor: A mask of shape [batch_size, num_actions] with invalid actions set to -inf.
    """
    batch_size = prev_actions[0].shape[0]  # Get the batch size
    num_actions = action_space.nvec[0]  # Total number of actions
    number_of_cells = grid_size[0] * grid_size[1]  # Number of cells in the grid

    # Initialize the mask for all batch elements with zeros
    prev_act_mask = torch.zeros(
        batch_size, num_actions, dtype=torch.float32, device=prev_actions[0].device
    )

    # Stack all previous actions into a single tensor of shape [batch_size, num_heads]
    actions_tensor = torch.stack(prev_actions, dim=1)  # Shape: [batch_size, num_heads]

    # Flatten the batch indices and actions for advanced indexing
    batch_indices = (
        torch.arange(batch_size, device=actions_tensor.device)
        .unsqueeze(1)
        .expand_as(actions_tensor)
    )
    flat_batch_indices = batch_indices.flatten()
    flat_actions = actions_tensor.flatten()

    # Mark the sampled actions as invalid
    prev_act_mask[flat_batch_indices, flat_actions] = -float("inf")

    # Handle below-cell actions
    below_cell_mask = flat_actions < number_of_cells  # True / False
    below_cell_actions = flat_actions[below_cell_mask]
    below_batch_indices = flat_batch_indices[below_cell_mask]
    prev_act_mask[below_batch_indices, below_cell_actions + number_of_cells] = -float(
        "inf"
    )

    # Handle above-cell actions
    above_cell_mask = (flat_actions >= number_of_cells) & (
        flat_actions < number_of_cells * 2
    )
    above_cell_actions = flat_actions[above_cell_mask]
    above_batch_indices = flat_batch_indices[above_cell_mask]
    prev_act_mask[above_batch_indices, above_cell_actions - number_of_cells] = -float(
        "inf"
    )

    return prev_act_mask


def nowp_mask_batch(head: int, observations: torch.Tensor) -> torch.Tensor:
    """
    Generate a batch-wise action mask based on whether a workpiece is present for the given head.

    Args:
        head (int): The current head index.
        observations (torch.Tensor): Tensor of shape [batch_size, observation_size] containing the batch of observations.

    Returns:
        torch.Tensor: A mask of shape [batch_size, num_actions], where invalid actions are set to -inf.
    """
    batch_size = observations.shape[0]
    num_actions = action_space.nvec[0]  # Total number of actions

    # Extract WP status for the current head across the batch
    wp_indices = observations[:, head * 6 + 1]  # Shape: [batch_size]

    # Initialize the mask for all actions with zeros
    mask = torch.zeros(
        batch_size, num_actions, dtype=torch.float32, device=observations.device
    )

    # Vectorized masking logic
    # For WP absent (wp_indices == 0): Mask all actions except the last one
    no_wp_mask = wp_indices == 0  # Shape: [batch_size]
    mask[no_wp_mask] = -float("inf")  # Mask all actions for batches with no WP
    mask[no_wp_mask, -1] = 0  # Unmask the last action for batches with no WP

    # For WP present (wp_indices == 1): Mask the last action
    wp_present_mask = wp_indices == 1  # Shape: [batch_size]
    mask[wp_present_mask, -1] = -float(
        "inf"
    )  # Mask the last action for batches with WP

    return mask


def get_remaining_stages_masks(head: int, observations: torch.Tensor) -> torch.Tensor:

    print("dic: ", dic_valid_actions_by_remaining_stages)

    batch_size = observations.shape[0]
    num_actions = action_space.nvec[0]  # Total number of actions

    # Extract pg and index_done for the current head across the batch
    pg_values = observations[:, head * 6 + 3]  # Shape: [batch_size]
    index_done_values = observations[:, head * 6 + 4]  # Shape: [batch_size]

    # Convert (pg, index_done) pairs into a tensor of shape [batch_size, 2]
    state_tuples = torch.stack(
        [pg_values, index_done_values], dim=1
    )  # Shape: [batch_size, 2]

    # Convert to a list of tuples to use as dictionary keys
    state_tuples_list = [tuple(map(int, state.tolist())) for state in state_tuples]

    # Retrieve valid actions for all batch elements
    allowed_action_lists = [
        dic_valid_actions_by_remaining_stages.get(state, [])
        for state in state_tuples_list
    ]

    # Create a mask initialized to -inf (invalid actions)
    mask = torch.full(
        (batch_size, num_actions),
        -float("inf"),
        dtype=torch.float32,
        device=observations.device,
    )

    # Convert valid actions into a tensor
    batch_indices = torch.arange(
        batch_size, device=observations.device
    ).repeat_interleave(
        torch.tensor(
            [len(lst) for lst in allowed_action_lists], device=observations.device
        )
    )
    action_indices = torch.tensor(
        sum(allowed_action_lists, []), device=observations.device
    )

    # Set valid actions to 0 (unmasked)
    if len(batch_indices) > 0:  # Avoid errors if all lists are empty
        mask[batch_indices, action_indices] = 0
    else:
        print("With these observations: ", observations, "  for head: ", head)
        print(
            "Found no possible action for remaining stages here! Deadlock. How could this happen?"
        )

    return mask


def get_finished_wp_mask(head: int, observations: torch.Tensor) -> torch.Tensor:
    batch_size = observations.shape[0]
    num_actions = action_space.nvec[0]  # Total number of actions

    # Initialize the mask for all actions with zeros
    mask = torch.zeros(
        batch_size, num_actions, dtype=torch.float32, device=observations.device
    )

    num_heads = num_actions
    if head < num_heads - 1:  # Only for non-queue heads
        # Extract pg and index_done for all observations

        pg = observations[:, head * 6 + 3].long()
        print("pg: ", pg)
        index_done = observations[:, head * 6 + 4].long().unsqueeze(-1)
        print("index_done: ", index_done)

        # Convert final_indices to tensor if not already
        final_indices_tensor = torch.tensor(
            final_indices_by_pgs, device=observations.device
        )
        print("final_indices: ", final_indices_tensor[pg])

        # Determine which workpieces are finished
        is_finished = index_done == final_indices_tensor[pg]

        # Service action mask for finished workpieces
        service_action_mask = torch.tensor(
            [0] * (grid_size[0] * grid_size[1])
            + [-float("inf")] * (grid_size[0] * grid_size[1])
            + [0] * 3,
            dtype=torch.float32,
            device=observations.device,
        )

        # Debugging shapes
        print("\n\nhead: ", head)
        print("observations: ", observations)
        print(f"observations shape: {observations.shape}")
        print(f"final_indices shape: {final_indices_tensor.shape}")
        print("final indices: ", final_indices_tensor)
        print(f"Mask shape: {mask.shape}")
        print(
            f"Is finished shape: {is_finished.shape}, Sum: {is_finished.sum().item()}"
        )
        print(f"Service action mask shape: {service_action_mask.shape}")

        # Ensure the mask length matches num_actions
        if service_action_mask.shape[0] != num_actions:
            raise ValueError(
                f"Service action mask length ({service_action_mask.shape[0]}) does not match num_actions ({num_actions})."
            )

        num_finished = is_finished.sum().item()  # Number of finished workpieces

        if num_finished > 0:
            print("num finished: ", num_finished)
            print("is_finished: ", is_finished)
            print("mask: ", mask)
            # Expand the service_action_mask to match the batch of finished observations
            expanded_mask = service_action_mask.unsqueeze(0).expand(num_finished, -1)
            print("mask[finished]", mask[is_finished.squeeze(-1)])

            # More debug prints
            # print(f"Expanded mask shape: {expanded_mask.shape}")
            # print(f"Mask[is_finished] shape: {mask[is_finished].shape}")

            # Apply the service_action_mask to each finished observation
            mask[is_finished.squeeze(-1)] = expanded_mask
            print("DONE THAT THING")
            # time.sleep(3)

        print("final mask: ", mask)

    return mask


def matrix_to_list(matrix):
    lst = []
    for y in range(matrix.shape[1] - 1, -1, -1):
        for x in range(matrix.shape[0] - 1, -1, -1):
            lst.append(matrix[x, y])

    return lst


actions = [
    torch.tensor([1, 5]),  # Actions for Head 0
    torch.tensor([4, 4]),  # Actions for Head 1
    torch.tensor([6, 6]),  # Actions for Head 2
]

print("hier ", prev_actions_mask_batch(actions))


# print(mb)
# print(wait_service_masking())


# a = [[1, 2, 3, 4], [1, 2, 3, 45]]
# b = [[3, 4, 6, 2], [3, 2, 1, 4]]


# general_masks = []
# for head in range(len(a)):
#    tmp = []
#    for action_index in range(len(a[head])):
#        tmp.append(min(a[head][action_index], b[head][action_index]))
#    general_masks.append(tmp)

# print(general_masks)

# test_obs = daisy.observation_space.sample()
# print(test_obs)
# print(machine_failure_masking(test_obs))

import torch

# print(previous_actions_mask([11]))

##gen = [1, 2, 3]
# ob = [3, 4, 5]
# acm = [2, 5, 1]

# combined_mask = torch.min(torch.stack([torch.tensor(gen), torch.tensor(ob), torch.tensor(acm)]), dim=0).values

# print(combined_mask)
o1 = daisy.observation_space.sample()
o2 = daisy.observation_space.sample()
print(o1)
print(o2)


obs = torch.tensor([o1, o2])
print("new: ", get_remaining_stages_masks(0, obs))

obs_mask = torch.tensor(
    [[0.0, 0.0, 0.0, -float("inf"), -float("inf"), 0.0, 0.0, 0.0, 0.0]]
)
general_mask = torch.tensor(
    [
        [
            0.0,
            -float("inf"),
            -float("inf"),
            0.0,
            -float("inf"),
            -float("inf"),
            -float("inf"),
            0.0,
            0.0,
        ]
    ]
)
action_mask = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
remaining_stages_mask = torch.tensor(
    [[-float("inf"), 0.0, 0.0, -float("inf"), 0.0, 0.0, 0.0, -float("inf"), 0.0]]
)

import pandas as pd


def generate_index_map(rows, cols):
    """
    Generate an index map of shape (rows, cols) filling from bottom-right
    to top-left (i.e., each column bottom-to-top, then move left).
    """
    index_map = np.zeros((rows, cols), dtype=int)
    index = 0
    # Go from rightmost column down to 0
    for c in reversed(range(cols)):
        # Fill this column from bottom row up to row=0
        for r in reversed(range(rows)):
            index_map[r, c] = index
            index += 1
    return index_map


def list_to_matrix(lst, rows, cols, values_per_cell=3):
    """
    Converts a flat list stored in bottom-right zigzag order into a 3D matrix.

    :param lst: The input flat list (each cell contains multiple values).
    :param rows: Number of rows in the target matrix.
    :param cols: Number of columns in the target matrix.
    :param values_per_cell: Number of values per matrix cell.
    :return: A 3D numpy array with shape (rows, cols, values_per_cell).
    """

    # Step 1: Reshape into (rows, cols, values_per_cell)
    data_2d = np.array(lst).reshape(-1, values_per_cell)

    index_map = generate_index_map(2, 3)
    # Use fancy indexing to reorder sub-lists into the desired 2×2 layout
    matrix = data_2d[index_map]

    return matrix


# Example: 3x3 matrix where each cell contains 6 values
lst = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

rows, cols = 2, 2  # Define matrix size

# Convert flat list to structured 3D matrix
# matrix = list_to_matrix(lst, rows, cols)
# print(matrix)
data = [
    [1, 2, 3],  # first sub-list
    [4, 5, 6],  # second sub-list
    [7, 8, 9],  # third sub-list
    [10, 11, 12],  # fourth sub-list
    [13, 14, 15],
    [16, 17, 18],
]
data_2d = np.array(data)  # shape: (4, 3)

# Define where each sub-list should go in the 2×2 matrix
#   (row, col) -> index in data_2d
# Top-left    -> 3
# Top-right   -> 1
# Bottom-left -> 2
# Bottom-right-> 0
index_map = np.array([[3, 1], [2, 0]], dtype=int)

# Use fancy indexing to reorder sub-lists into the desired 2×2 layout
matrix = data_2d[index_map]

print(matrix)


print(generate_index_map(2, 3))


data_2d = np.array(data)  # shape: (4, 3)

# Define where each sub-list should go in the 2×2 matrix
#   (row, col) -> index in data_2d
# Top-left    -> 3
# Top-right   -> 1
# Bottom-left -> 2
# Bottom-right-> 0
index_map = generate_index_map(2, 3)
# Use fancy indexing to reorder sub-lists into the desired 2×2 layout
matrix = data_2d[index_map]
print(matrix)

df = pd.DataFrame(matrix.tolist())  # Convert NumPy to list first
print(df.applymap(str).to_string(index=False, header=False))

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix)
print(matrix_to_list(matrix))


print(get_finished_wp_mask(0, obs))

import torch
import gym


def scale_observations(
    obs: torch.Tensor, observation_space: gym.spaces.Space
) -> torch.Tensor:
    """
    Scale a batch of observations so that each feature is between 0 and 1,
    using the predefined minimum and maximum values for each feature from the observation_space.

    Args:
        obs (torch.Tensor): A batch of observations with shape [batch_size, observation_size].
        observation_space (gym.spaces.Space): The observation space that defines the min/max values for each feature.

    Returns:
        torch.Tensor: Scaled observations with the same shape as the input, with each feature in [0, 1].
    """

    # Extract min and max values from the observation space
    min_vals = torch.zeros(
        len(daisy.observation_space.nvec)
    )  # Min values for each component (starts at 0)
    max_vals = torch.tensor(daisy.observation_space.nvec) - 1
    print("min:  ", min_vals.shape)
    print("max: ", max_vals.shape)
    print("obs: ", obs.shape)

    # Ensure the input is a float tensor
    obs = obs.float()

    # Scale each feature (dimension) based on the predefined min and max values
    scaled_obs = (obs - min_vals) / (max_vals - min_vals)

    # Handle any division by zero (if max == min, the feature is constant across the batch)
    # scaled_obs = torch.where(max_vals - min_vals == 0, torch.zeros_like(scaled_obs), scaled_obs)

    return scaled_obs


# Scale the batch of observations using the space's theoretical min/max values
scaled_batch = scale_observations(obs, daisy.observation_space)

print("Original Observations:\n", obs)
print("\nScaled Observations:\n", scaled_batch)
