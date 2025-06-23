import numpy as np
import torch
from numba import jit
from torch import Tensor

"""
here is where the action sampling and masking logic is implemented.
jit is used for performance optimization, especially for the action masking logic, as it is sequential and can be computationally intensive.
"""


@jit(nopython=True)
def get_prev_actions_action_mask_numba_for(
    out_mask: np.array, num_cells: int, num_actions: int, prev_actions: np.array
):
    """
    Generate a mask to invalidate actions based on a list of previous actions for all heads in a batch.

    Args:
        prev_actions (List[torch.Tensor]): A list of tensors, where each tensor has shape [batch_size]
        and contains the previous actions sampled for a specific head across the batch.

    Returns:
        torch.Tensor: A mask of shape [batch_size, num_actions] with invalid actions set to -inf.
    """

    batch_size = prev_actions.shape[0]  # Get the batch size
    for b in range(batch_size):
        for act in prev_actions[b]:
            if act == num_actions - 1:
                # last actions can always be taken
                continue

            out_mask[b, act] = False

            # if act >= num_actions - 3:
            #     # last 3 actions have no mirror action
            #     continue
            #
            # if act >= (num_cells - 1):
            #     out_mask[b, act - (num_cells - 1)] = False
            # else:
            #     out_mask[b, act + (num_cells - 1)] = False


@jit(nopython=True)
def get_nowp_mask_numba_for(
    out_mask: np.array, num_cells: int, cell: int, observations: np.array
):
    """
    Generate a batch-wise action mask based on whether a workpiece is present for the given head.

    Args:
        observations (torch.Tensor): Tensor of shape [batch_size, observation_size] containing the batch of observations.

    Returns:
        torch.Tensor: A mask of shape [batch_size, num_actions], where invalid actions are set to -inf.
    """
    if cell < num_cells - 1:  # all heads excep queue
        # Extract WP status for the current head across the batch
        wp_indices = observations[:, cell * 6 + 1]  # Shape: [batch_size]
    else:  # queue head
        wp_indices = observations[:, cell * 6]

    # For WP absent (wp_indices == 0): Mask all actions except the last one
    no_wp_mask = wp_indices == 0  # Shape: [batch_size]
    out_mask[no_wp_mask, :-1] = (
        False  # Mask all actions for batches with no WP, except the last action for batches with no WP
    )

    # For WP present (wp_indices == 1): Mask the last action
    wp_present_mask = wp_indices >= 1  # Shape: [batch_size]
    out_mask[wp_present_mask, -1] = False  # Mask the last action for batches with WP


@jit(nopython=True)
def compute_mask_new(
    cell: int,
    *,
    out_mask: np.array,
    num_actions: int,
    num_cells: int,
    downstream_mask: np.array,
    obs_unscaled: np.array,
    prev_actions: np.array,
):

    ds_mask_ = downstream_mask[cell, :]
    out_mask[:, ~ds_mask_] = False

    get_nowp_mask_numba_for(out_mask, int(num_cells), cell, obs_unscaled)

    if cell > 0:
        # get previous actions mask
        get_prev_actions_action_mask_numba_for(
            out_mask, num_cells, num_actions, prev_actions
        )


def sample_and_fill_logits_mask(
    *,
    logits: torch.Tensor,  # batch x cells x actions
    out_actions: np.array,
    out_logits_mask: np.array,
    inference: bool,
    downstream_mask: Tensor,
    obs_unscaled: Tensor,
) -> None:
    """
    Handles sequential sampling with action masking.

    Returns:
        A dictionary with SampleBatch.ACTIONS and SampleBatch.ACTION_LOGP.
    """

    batch_size, num_cells, num_actions = logits.shape

    prev_actions = np.zeros((batch_size, 0), dtype=int)
    obs_unscaled_ = obs_unscaled.detach().cpu().numpy()
    downstream_mask_ = downstream_mask.detach().cpu().numpy()
    logits_ = logits.detach().cpu().numpy()

    for cell in range(num_cells):
        cell_logits = logits_[:, cell, :]

        mask2 = np.ones((batch_size, num_actions), dtype=np.bool)
        compute_mask_new(
            out_mask=mask2,
            cell=cell,
            num_actions=num_actions,
            num_cells=num_cells,
            downstream_mask=downstream_mask_,
            obs_unscaled=obs_unscaled_,
            prev_actions=prev_actions,
        )

        out_logits_mask[:, cell, :] = mask2

        # apply masks on logits
        masked_logits = np.copy(cell_logits)
        masked_logits[~mask2] = -1000

        # Check for NaN or Inf values in the masked_logits
        if np.any(np.isnan(masked_logits)):
            print("Debug: masked_logits contains NaN values.")
            print("NaN values found in masked_logits:", masked_logits)

        if np.any(np.isinf(masked_logits)):
            print("Debug: masked_logits contains Inf values.")
            print("Inf values found in masked_logits:", masked_logits)

        # if inference:
        #     # Deterministic action selection (argmax)
        #     actions = masked_logits.argmax(axis=-1)  # Returns the most probable action
        # else:
        #     # Stochastic action selection (for training)
        #     prob = numpy_softmax(masked_logits, dim=-1)

        if inference:
            # Deterministic action selection (argmax)
            actions = masked_logits.argmax(axis=-1)  # Returns the most probable action
        else:
            # Stochastic action selection (for training)
            # Apply Log-Sum-Exp for numerical stability in softmax
            max_logits = np.max(
                masked_logits, axis=-1, keepdims=True
            )  # Find max logit for numerical stability
            stabilized_logits = (
                masked_logits - max_logits
            )  # Subtract max logit to prevent overflow
            exp_logits = np.exp(stabilized_logits)
            prob = exp_logits / np.sum(
                exp_logits, axis=-1, keepdims=True
            )  # Normalize to get valid probabilities

            actions = np.zeros(batch_size, dtype=np.int64)
            for i in range(batch_size):
                actions[i] = np.random.choice(num_actions, p=prob[i])

        out_actions[:, cell] = actions
        prev_actions = np.concatenate(
            (prev_actions, np.expand_dims(actions, axis=-1)), axis=1
        )


def _sample_actions(
    *,
    logits: torch.Tensor,  # batch x cells x actions
    inference: bool,
    downstream_mask: Tensor,
    obs_unscaled: Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Handles sequential sampling with action masking.

    :param logits: The raw logits computed by _pi().
    :param inference: If True, uses deterministic sampling.

    Returns:
        A dictionary with SampleBatch.ACTIONS and SampleBatch.ACTION_LOGP.
    """

    batch_size, num_cells, num_actions = logits.shape
    out_logits_mask_ = np.ones((batch_size, num_cells, num_actions), dtype=np.bool)
    out_actions_ = np.zeros((batch_size, num_cells), dtype=np.int64)

    # fast sampling of actions and computation of masks
    sample_and_fill_logits_mask(
        logits=logits,
        out_actions=out_actions_,
        out_logits_mask=out_logits_mask_,
        inference=inference,
        downstream_mask=downstream_mask,
        obs_unscaled=obs_unscaled,
    )

    # compute log probabilities
    actions = torch.from_numpy(out_actions_)
    logits_mask = torch.from_numpy(out_logits_mask_)

    logits_masked = torch.where(
        logits_mask,
        logits,
        torch.tensor(-1000, dtype=torch.float32, device=logits.device),
    )

    logps = []
    for i in range(num_cells):
        dist = torch.distributions.Categorical(logits=logits_masked[:, i, :])
        logps.append(dist.log_prob(actions[:, i]))  # shape: [batch_size]

    logp = torch.stack(logps, dim=1).sum(dim=1)  # final shape: [batch_size]

    return actions, logp
