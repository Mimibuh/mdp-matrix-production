import numpy as np
import torch


def numpy_softmax(values, dim):
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=dim, keepdims=True)


def build_downstream_mask(
    *,
    num_cells: int,
    num_actions_per_cell: int,
    downstream_relations: dict[int, list[int]],
) -> torch.Tensor:
    downstream_mask = torch.zeros((num_cells, num_actions_per_cell), dtype=torch.bool)

    for cell_index in range(num_cells):
        valid_actions = downstream_relations[cell_index]  # Default: All valid
        downstream_mask[cell_index, valid_actions] = 1  # Set valid actions to True
    return downstream_mask
