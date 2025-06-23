# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:33:39 2025

@author: mimib
"""


import gymnasium

# from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_distributions import TorchDistribution

# from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core import Columns


class MultiHeadDictRLModule(TorchRLModule):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        inference_only: bool,
        model_config: dict,
        catalog_class=None,
        **kwargs,
    ):
        """
        Args:
            observation_space (gymnasium.Space): The observation space.
            action_space (gymnasium.Space): The action space.
            model_config (dict): Custom config dict provided via RLModuleSpec(model_config=...).
            **kwargs: Capture any extra arguments RLlib might add in the future.
        """
        super().__init__()

        # Optionally, store these if you need them later:
        self.observation_space = observation_space
        self.action_space = action_space

        print("RLModule Observation Space Type:", type(self.observation_space))
        print("RLModule Action Space Type:", type(self.action_space))

        # Total input size for the policy network
        self.total_obs_size = len(self.observation_space)
        print(self.total_obs_size)

        # Extract custom configs (similar to your original code).
        # Rename from `model_config_dict` to `model_config`.
        config = model_config
        self.grid_size = config["grid_size"]
        # self.obs_sizes = config["obs_sizes"]
        # self.total_obs_size = sum(self.obs_sizes.values())
        self.hidden_size = config["fcnet_hiddens"]
        # self.action_digits = config["action_digits"]
        self.grid_action_dim = config["total_number_action_components"]

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.total_obs_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
        )

        # Policy heads
        self.num_heads = self.grid_size[0] * self.grid_size[1] + 1

        self.heads = nn.ModuleList()

        # Grid heads
        self.heads = nn.ModuleList(
            [
                nn.Linear(
                    self.hidden_size[1] + i * self.grid_action_dim, self.grid_action_dim
                )
                for i in range(1, self.num_heads + 1)
            ]
        )

        # Value head
        self.value_head = nn.Linear(self.hidden_size[1], 1)
        self.value_out = None

    def forward(self, input_dic, state, seq_lens):

        print("RLModule Observation Space Type:", type(self.observation_space))
        print("RLModule Action Space Type:", type(self.action_space))

        """
        1) Encode the observation into base_embed
        2) The heads themselves won't be fully computed here. We'll do that in the custom dist.
        3) Store value function output for later.
        """

        obs = input_dic["obs"].float()
        base_embed = self.shared_encoder(obs)  # shape [B, hidden_dim]

        # Value function
        self._value_out = self.value_head(base_embed)  # Save value function output

        # Use your custom distribution to sample actions
        action_distribution = MultiHeadChainDistribution(base_embed, self)
        sampled_actions = action_distribution.sample(base_embed)

        # Return actions and other outputs
        return {
            Columns.ACTIONS: sampled_actions,  # Include actions under the expected key
            "action_dist": action_distribution,  # Optional: pass the action distribution
        }

    def value_function(self):
        # Return the estimated value, reshaped to [B]
        return torch.reshape(self._value_out, [-1])

    def get_head_logits(self, base_embed, prev_actions, head_index):

        print("RLModule Observation Space Type:", type(self.observation_space))
        print("RLModule Action Space Type:", type(self.action_space))

        """
        Produce logits for the specified grid cell head.

        Args:
            base_embed (torch.Tensor): Shared embedding [B, hidden_dim].
            prev_actions (torch.Tensor): All previous actions [B, head_index * grid_action_dim].
            head_index (int): Index of the current grid cell.

        Returns:
            torch.Tensor: Logits for the current grid cell [B, grid_action_dim].
        """
        combined_input = torch.cat(
            [base_embed, prev_actions], dim=-1
        )  # [B, hidden_dim + head_index * grid_action_dim]
        logits = self.heads[head_index](combined_input)  # [B, grid_action_dim]
        return logits


# ---------------------------
# 2) The Custom Distribution
# ---------------------------
# import torch
import torch.nn.functional as F

# from ray.rllib.models.torch.torch_distributions import TorchDistribution

FLOAT_MIN = float("-inf")  # Define globally


class MultiHeadChainDistribution(TorchDistribution):
    """
    Autoregressive distribution that samples from each head i in order,
    passing the chosen action from head i to head i+1.

    Also demonstrates optional masking.
    """

    def __init__(self, inputs, model):
        """
        Args:
            inputs (torch.Tensor): The model outputs from forward() -- in this case, [B, hidden_dim].
            model: The custom RLModule.
        """
        super().__init__(inputs, model)
        self.base_embed = inputs  # Base embedding, shape [B, hidden_dim]
        self.model = model
        self.grid_size = model.grid_size
        self.num_heads = model.num_heads
        self.num_actions_per_cell = model.grid_action_dim

    @staticmethod
    def required_input_dim(action_space, model_config):
        """
        Define the required input dimension for the distribution.

        Args:
            action_space: The action space of the environment.
            model_config (dict): Model configuration dictionary.

        Returns:
            int: The expected input dimension for the distribution.
        """
        return model_config["fcnet_hiddens"][-1]  # Typically the hidden layer size

    def _get_torch_distribution(self, dist_inputs):
        """
        Get the PyTorch distribution object.

        Args:
            dist_inputs (torch.Tensor): The logits for the distribution.

        Returns:
            torch.distributions.Categorical: A Categorical distribution for sampling.
        """
        return torch.distributions.Categorical(logits=dist_inputs)

    def sample(self, base_embed):
        """
        Sample a complete multi-cell action by iterating from head 0 to head (num_cells - 1).
        """
        batch_size = base_embed.shape[0]
        grid_actions = []
        prev_actions = torch.zeros(batch_size, 0, device=base_embed.device)

        for i in range(self.num_heads):
            logits_i = self.model.get_head_logits(base_embed, prev_actions, i)
            valid_mask_i = self.compute_valid_mask(i, batch_size)
            masked_logits = logits_i + (1 - valid_mask_i) * FLOAT_MIN
            dist_i = self._get_torch_distribution(masked_logits)
            action_i = dist_i.sample()  # [B]
            grid_actions.append(action_i)

            action_one_hot = F.one_hot(
                action_i, num_classes=self.num_actions_per_cell
            ).float()
            prev_actions = torch.cat([prev_actions, action_one_hot], dim=-1)

        env_actions = torch.stack(grid_actions, dim=1)
        return env_actions

    def logp(self, env_actions):
        """
        Compute the log-probabilities of a batch of given actions.
        """
        batch_size = env_actions.shape[0]
        sampled_actions = self.to_sampled_actions(env_actions)  # [B, num_heads]
        logp_parts = []
        prev_actions = torch.zeros(batch_size, 0, device=self.base_embed.device)

        for i in range(self.num_heads):
            logits_i = self.model.get_head_logits(self.base_embed, prev_actions, i)
            valid_mask_i = self.compute_valid_mask(i, batch_size)
            masked_logits = logits_i + (1 - valid_mask_i) * FLOAT_MIN
            dist_i = self._get_torch_distribution(masked_logits)
            action_i = sampled_actions[:, i]
            logp_i = dist_i.log_prob(action_i)  # [B]
            logp_parts.append(logp_i)

            action_one_hot = F.one_hot(
                action_i, num_classes=self.num_actions_per_cell
            ).float()
            prev_actions = torch.cat([prev_actions, action_one_hot], dim=-1)

        return torch.stack(logp_parts, dim=1).sum(dim=1)

    def compute_valid_mask(self, i, batch_size):
        """
        Example valid mask: all ones (no invalid actions).
        """
        return torch.ones(
            (batch_size, self.num_actions_per_cell), device=self.base_embed.device
        )
