# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:01:19 2025

@author: mimib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override
from ray.rllib.core import Columns
from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical
from mimi_env import Desp


from typing import Dict


class MultiHeadActionsRLM(TorchRLModule, ValueFunctionAPI):
    """
    An RLModule for multi-head autoregressive policies.

    This module uses multiple heads to sample actions sequentially, where each
    head conditions on the observations and the actions sampled by previous heads.
    """

    @override(RLModule)
    def setup(self):
        super().setup()

        self.grid_size = self.model_config["grid_size"]
        self.action_decoding = self.model_config["action_decoding"]
        self.machine_or_buffer = self.model_config["machine_or_buffer"]
        self.dic_valid_actions_by_remaining_stages = self.model_config[
            "dic_valid_actions_by_remaining_stages"
        ]
        self.final_indices_by_pgs = self.model_config["final_indices_by_pgs"]

        # Define the action distribution class
        self.action_dist_cls = TorchMultiCategorical

        # Shared encoder for the observations.
        self._shared_encoder = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Policy heads for each action component.
        self.num_heads = self.action_space.nvec.size
        self.heads = nn.ModuleList()

        for i in range(self.num_heads):
            self.heads.append(
                nn.Linear(
                    256 + i * self.action_space.nvec.max(), self.action_space.nvec[i]
                )
            )

        # Value function head.
        self._value_net = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Store the nvec (input lengths) of the MultiDiscrete action space
        self.action_dist_input_lens = self.action_space.nvec.tolist()
        # print("\n\nactions lengths for logits: ", self.action_dist_input_lens)#

        # generate downstream masks, first head is cell rightest and downest, then move up before moving left
        # [6 3]
        # [5 2]
        # [4 1]

        downstream_masks = self.get_downstream_action_masks()
        wait_service_masks = self.get_wait_service_action_masks()

        self.general_masks = []
        for head in range(len(downstream_masks)):
            tmp = []
            for action_index in range(len(downstream_masks[head])):
                tmp.append(
                    min(
                        downstream_masks[head][action_index],
                        wait_service_masks[head][action_index],
                    )
                )
            self.general_masks.append(tmp)

        # for scaling later in pi method
        # Extract min and max values from the observation space
        self.min_vals = torch.zeros(
            len(self.observation_space.nvec)
        )  # Min values for each component (starts at 0)
        self.max_vals = torch.tensor(self.observation_space.nvec) - 1

    def get_inference_action_dist_cls(self):
        return TorchMultiCategorical.get_partial_dist_cls(
            input_lens=self.action_dist_input_lens
        )

    def _pi(self, obs: torch.Tensor, inference: bool) -> Dict[str, torch.Tensor]:
        """Compute actions autoregressively using the heads."""
        batch_size = obs.shape[0]

        # for unscaled version, just use this base_embed and obs
        # base_embed = self._shared_encoder(obs.float())  # Shared embedding

        # scaled version
        # 1) Raw obs for integer checks
        raw_obs = obs
        # 2) Scale obs for the neural network
        scaled_obs = self.scale_observations(obs.float())
        # Shared encoder for scaled observations
        base_embed = self._shared_encoder(scaled_obs)

        prev_actions = torch.zeros(
            batch_size, 0, device=obs.device
        )  # Start with no previous actions

        actions = []
        log_probs = []
        logits_list = []

        # get action masks depending on the observation --> plan must still be feasible, no service at failed machines
        obs_mask = self.get_obs_action_mask(
            raw_obs
        )  # tensor (batch_size, num_actions) --> same for all heads in one batch

        for i, head in enumerate(self.heads):
            # Concatenate base embedding with previous actions
            input_to_head = torch.cat([base_embed, prev_actions], dim=-1)

            # Compute logits and distribution for the current head
            logits = head(input_to_head)

            # apply masking
            # general mask (downstream and wait_service)
            general_mask = torch.tensor(self.general_masks[i])  # get mask for head i
            # Duplicate for the whole batch size
            general_mask = general_mask.unsqueeze(0).expand(
                batch_size, -1
            )  # shape [batchsize, num_actions]

            nowp_mask = self.get_nowp_mask(i, raw_obs)  # get mask for head i
            rem_stages_mask = self.get_remaining_stages_masks(i, raw_obs)
            finished_wp_mask = self.get_finished_wp_mask(i, raw_obs)

            if not actions:
                num_actions = self.action_space.nvec[0]
                action_mask = torch.zeros(
                    batch_size, num_actions, dtype=torch.float32, device=obs.device
                )
            else:
                action_mask = self.get_prev_actions_action_mask(
                    actions
                )  # shape [batchsize, num_actions] mask for next head over all batches

            # combine masks
            combined_mask = torch.min(
                torch.stack(
                    [
                        obs_mask,
                        general_mask,
                        action_mask,
                        nowp_mask,
                        rem_stages_mask,
                        finished_wp_mask,
                    ],
                    dim=0,
                ),
                dim=0,
            ).values  # sahpe[batchsize, num_actions]
            logits += combined_mask

            # print("\n\n head: " , i)
            # print("combined_mask: ", combined_mask)
            # print("logits: ", logits)
            # print("prev actions: ", actions)
            # print("obs: ", obs)
            # #print("At head: ", i)
            # print("obs_mask: ", obs_mask)
            # print("general_mask: ", general_mask)
            # print("action_mask: ", action_mask)
            # print("remaining_stages_mask: ", rem_stages_mask)
            # print("nowp_mask: ", nowp_mask)
            # print("finished_wp_mask: ", finished_wp_mask)

            # print("combined masks looks like: ", combined_mask)

            if logits.max().item() == -float("inf"):
                print("combined_mask: ", combined_mask)
                print("logits: ", logits)
                print("prev actions: ", actions)
                print("obs: ", obs)
                print("At head: ", i)
                print("obs_mask: ", obs_mask)
                print("general_mask: ", general_mask)
                print("action_mask: ", action_mask)
                print("remaining_stages_mask: ", rem_stages_mask)
                print("nowp_mask: ", nowp_mask)
                raise ValueError("All actions are invalid due to masking")

            logits_list.append(logits)
            # print("current logits: ", logits)
            dist = TorchCategorical(logits=logits)
            # print("dist: ")

            # Sample action
            if inference:
                dist = dist.to_deterministic()

            action = dist.sample()
            actions.append(action)

            # Save log-probability
            log_probs.append(dist.logp(action))

            # One-hot encode the action and append to prev_actions
            one_hot_action = F.one_hot(
                action, num_classes=self.action_space.nvec[i]
            ).float()
            prev_actions = torch.cat([prev_actions, one_hot_action], dim=-1)

        actions = torch.stack(actions, dim=1)  # Combine actions into a single tensor
        log_probs = torch.sum(
            torch.stack(log_probs, dim=1), dim=1
        )  # Sum log-probabilities
        fin_logits = torch.cat(logits_list, dim=1)

        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: log_probs,
            Columns.ACTION_DIST_INPUTS: fin_logits,
        }

    def scale_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Scale a batch of observations so that each feature is between 0 and 1,
        using the predefined minimum and maximum values for each feature from the observation_space.

        Args:
            obs (torch.Tensor): A batch of observations with shape [batch_size, observation_size].
            observation_space (gym.spaces.Space): The observation space that defines the min/max values for each feature.

        Returns:
            torch.Tensor: Scaled observations with the same shape as the input, with each feature in [0, 1].
        """

        # Ensure the input is a float tensor
        obs = obs.float()

        # Scale each feature (dimension) based on the predefined min and max values
        scaled_obs = (obs - self.min_vals) / (self.max_vals - self.min_vals)

        # Handle any division by zero (if max == min, the feature is constant across the batch)
        scaled_obs = torch.where(
            self.max_vals - self.min_vals == 0, torch.zeros_like(scaled_obs), scaled_obs
        )

        return scaled_obs

    def get_downstream_action_masks(self):
        downstream_masks = []
        for y in range(self.grid_size[1] - 1, -1, -1):
            for x in range(self.grid_size[0] - 1, -1, -1):

                # current head gird position : (x, y)
                possible_new = []
                for x_ in range(self.grid_size[0]):
                    for y_ in range(y, self.grid_size[1], 1):
                        possible_new.append((x_, y_))

                possible_new.append((-2, -2))
                single_head_mask = []
                keys_except_last = list(self.action_decoding.keys())[:-1]

                for (
                    key
                ) in (
                    keys_except_last
                ):  # last action key stands for "noWP" not a location!
                    if self.action_decoding[key][1] in possible_new:
                        single_head_mask.append(0)
                    else:
                        single_head_mask.append(-float("inf"))

                # append "okay" (=0) for last action as it indicates that no WP is here, has nothing to do with downstream constraint
                single_head_mask.append(0)

                # append head mask to all head masks (in correct order by for loops garantueed)
                downstream_masks.append(single_head_mask)

        # queue can do every action, append to downstream_masks, is the last head
        downstream_masks.append([0] * len(self.action_decoding))

        return downstream_masks

    def get_wait_service_action_masks(self):
        one_mask_for_all_heads = [0] * self.action_space.nvec[0]
        for i in range(len(self.machine_or_buffer)):
            if self.machine_or_buffer[i] == 0:
                one_mask_for_all_heads[(self.grid_size[0] * self.grid_size[1] + i)] = (
                    -float("inf")
                )

        return [one_mask_for_all_heads] * self.num_heads

    def get_obs_action_mask(self, observations: torch.Tensor):

        # machine failure depending on status, calculates for complete batch
        batch_size = observations.shape[0]
        # obs_dim = observations.shape[1]

        # Determine machine buffer status: Extract every 6th value in the observation
        indices = torch.arange(
            0, self.grid_size[0] * self.grid_size[1] * 6, 6, device=observations.device
        )
        machine_buffer_status = observations[
            :, indices
        ]  # Shape: [batch_size, num_machines]

        # Initialize mask for all actions
        general_mask = torch.zeros(
            batch_size, self.action_space.nvec[0], device=observations.device
        )

        # Identify failed machines and update mask
        machine_failure_indices = (machine_buffer_status == 1).nonzero(
            as_tuple=True
        )  # (batch_idx, machine_idx)
        grid_offset = self.grid_size[0] * self.grid_size[1]
        general_mask[
            machine_failure_indices[0], grid_offset + machine_failure_indices[1]
        ] = -float("inf")

        return general_mask  # Shape: [batch_size, num_actions]

    def get_nowp_mask(self, head: int, observations: torch.Tensor) -> torch.Tensor:
        """
        Generate a batch-wise action mask based on whether a workpiece is present for the given head.

        Args:
            head (int): The current head index.
            observations (torch.Tensor): Tensor of shape [batch_size, observation_size] containing the batch of observations.

        Returns:
            torch.Tensor: A mask of shape [batch_size, num_actions], where invalid actions are set to -inf.
        """

        batch_size = observations.shape[0]
        num_actions = self.action_space.nvec[0]  # Total number of actions

        if head < self.num_heads - 1:  # all heads excep queue
            # Extract WP status for the current head across the batch
            wp_indices = observations[:, head * 6 + 1]  # Shape: [batch_size]
        else:  # queue head
            wp_indices = observations[:, head * 6]

        # Initialize the mask for all actions with zeros
        mask = torch.zeros(
            batch_size, num_actions, dtype=torch.float32, device=observations.device
        )

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

    def get_finished_wp_mask(
        self, head: int, observations: torch.Tensor
    ) -> torch.Tensor:
        batch_size = observations.shape[0]
        num_actions = self.action_space.nvec[0]  # Total number of actions

        # Initialize the mask for all actions with zeros
        mask = torch.zeros(
            batch_size, num_actions, dtype=torch.float32, device=observations.device
        )

        if head < self.num_heads - 1:  # Only for non-queue heads
            # Extract pg and index_done for all observations
            pg = observations[:, head * 6 + 3].long()
            index_done = observations[:, head * 6 + 4].long().unsqueeze(-1)

            # Convert final_indices to tensor if not already
            final_indices_tensor = torch.tensor(
                self.final_indices_by_pgs, device=observations.device
            )

            # Determine which workpieces are finished
            is_finished = index_done == final_indices_tensor[pg]

            # Service action mask for finished workpieces
            service_action_mask = torch.tensor(
                [0] * (self.grid_size[0] * self.grid_size[1])
                + [-float("inf")] * (self.grid_size[0] * self.grid_size[1])
                + [0] * 3,
                dtype=torch.float32,
                device=observations.device,
            )

            # Debugging shapes
            # print("\n\nhead: ", head)
            # print("observations: ", observations)
            # print(f"observations shape: {observations.shape}")
            # print(f"final_indices shape: {final_indices_tensor.shape}")
            # print("final indices: ", final_indices_tensor)
            # print(f"Mask shape: {mask.shape}")
            # print(f"Is finished shape: {is_finished.shape}, Sum: {is_finished.sum().item()}")
            # print(f"Service action mask shape: {service_action_mask.shape}")

            # Ensure the mask length matches num_actions
            if service_action_mask.shape[0] != num_actions:
                raise ValueError(
                    f"Service action mask length ({service_action_mask.shape[0]}) does not match num_actions ({num_actions})."
                )

            num_finished = is_finished.sum().item()  # Number of finished workpieces

            if num_finished > 0:
                # Expand the service_action_mask to match the batch of finished observations
                expanded_mask = service_action_mask.unsqueeze(0).expand(
                    num_finished, -1
                )

                # More debug prints
                # print(f"Expanded mask shape: {expanded_mask.shape}")
                # print(f"Mask[is_finished] shape: {mask[is_finished].shape}")

                # Apply the service_action_mask to each finished observation
                mask[is_finished.squeeze(-1)] = expanded_mask
                #    time.sleep(1)

            # print("final mask: ", mask)

        return mask

    def get_prev_actions_action_mask(self, prev_actions):
        """
        Generate a mask to invalidate actions based on a list of previous actions for all heads in a batch.

        Args:
            prev_actions (List[torch.Tensor]): A list of tensors, where each tensor has shape [batch_size]
            and contains the previous actions sampled for a specific head across the batch.

        Returns:
            torch.Tensor: A mask of shape [batch_size, num_actions] with invalid actions set to -inf.
        """

        # Handle the case where prev_actions is empty (first iteration)
        if not prev_actions:  # Check if the list is empty
            batch_size = prev_actions[0].shape[0]  # Get the batch size
            num_actions = self.action_space.nvec[0]  # Total number of actions
            return torch.zeros(
                batch_size,
                num_actions,
                dtype=torch.float32,
                device=prev_actions[0].device,
            )

        batch_size = prev_actions[0].shape[0]  # Get the batch size
        num_actions = self.action_space.nvec[0]  # Total number of actions
        number_of_cells = (
            self.grid_size[0] * self.grid_size[1]
        )  # Number of cells in the grid

        # Initialize the mask for all batch elements with zeros
        prev_act_mask = torch.zeros(
            batch_size, num_actions, dtype=torch.float32, device=prev_actions[0].device
        )

        # Stack all previous actions into a single tensor of shape [batch_size, num_heads]
        actions_tensor = torch.stack(
            prev_actions, dim=1
        )  # Shape: [batch_size, num_heads]

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

        # caution: "noWP" action can be chosen twice! dont mask
        prev_act_mask[flat_batch_indices, -1] = 0

        # Handle below-cell actions
        below_cell_mask = flat_actions < number_of_cells  # True / False
        below_cell_actions = flat_actions[below_cell_mask]
        below_batch_indices = flat_batch_indices[below_cell_mask]
        prev_act_mask[below_batch_indices, below_cell_actions + number_of_cells] = (
            -float("inf")
        )

        # Handle above-cell actions
        above_cell_mask = (flat_actions >= number_of_cells) & (
            flat_actions < number_of_cells * 2
        )
        above_cell_actions = flat_actions[above_cell_mask]
        above_batch_indices = flat_batch_indices[above_cell_mask]
        prev_act_mask[above_batch_indices, above_cell_actions - number_of_cells] = (
            -float("inf")
        )

        return prev_act_mask  # shape [batchsize, num_actions] gives mask for next head

    def get_remaining_stages_masks(
        self, head: int, observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action masks based on remaining stages constraints.
        If this is the last head (head == self.num_heads - 1), the product group (`pg`)
        is extracted from head * 6 + 1, and index_done is always 0.

        Args:
            head (int): Current head index.
            observations (torch.Tensor): Batch of observations with shape [batch_size, obs_dim].

        Returns:
            torch.Tensor: Mask tensor of shape [batch_size, num_actions] with -inf for invalid actions.
        """

        batch_size = observations.shape[0]
        num_actions = self.action_space.nvec[0]  # Number of available actions

        # Check if this is the last head
        if head == self.num_heads - 1:
            # print("current head is the queue head")
            pg_values = observations[:, head * 6 + 1]  # Product group from head * 6 + 1
            index_done_values = torch.zeros(
                batch_size, dtype=torch.int64, device=observations.device
            )  # Always 0
        else:
            pg_values = observations[
                :, head * 6 + 3
            ]  # Normal case: get pg from head * 6 + 3
            index_done_values = observations[
                :, head * 6 + 4
            ]  # Get index_done from head * 6 + 4

        # Convert (pg, index_done) pairs into a tensor of shape [batch_size, 2]
        state_tuples = torch.stack(
            [pg_values, index_done_values], dim=1
        )  # Shape: [batch_size, 2]

        # Convert to a list of tuples to use as dictionary keys
        state_tuples_list = [tuple(map(int, state.tolist())) for state in state_tuples]

        # Retrieve valid actions for all batch elements
        allowed_action_lists = [
            self.dic_valid_actions_by_remaining_stages.get(state, [])
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

    @override(TorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self._pi(batch[Columns.OBS], inference=True)

    @override(TorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self._pi(batch[Columns.OBS], inference=False)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._forward_exploration(batch)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, torch.Tensor], embeddings=None):
        """Value function forward pass."""
        obs = batch[Columns.OBS].float()
        vf_out = self._value_net(obs)
        return vf_out.squeeze(-1)


if __file__ == "__main__":
    dummy_env = Desp()
    obs_space = dummy_env.observation_space
    print(type(obs_space))
    act_space = dummy_env.action_space
    ad = dummy_env.action_decoding
    gs = dummy_env.grid_size
    mb = dummy_env.machine_or_buffer
    vabrs = dummy_env.dic_valid_actions_by_remaining_stages
    final_indices = dummy_env.final_indices_by_pgs

    model_config = {
        "grid_size": gs,
        "action_decoding": ad,
        "machine_or_buffer": mb,
        "dic_valid_actions_by_remaining_stages": vabrs,
        "final_indices_by_pgs": final_indices,
        # "fcnet_hiddens": [256, 256],
        # "action_digits": 4,
        # "total_number_action_components": 10
        # Add other configurations as needed
    }

    test_model = MultiHeadActionsRLM(
        observation_space=obs_space, action_space=act_space, model_config=model_config
    )

    test_tensor = torch.tensor(
        [
            [
                0,
                1,
                0,
                0,
                2,
                2,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            ]
        ]
    )

    print("test: ", test_model.get_remaining_stages_masks(0, test_tensor))
