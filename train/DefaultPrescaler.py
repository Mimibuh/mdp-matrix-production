import torch
from torch import nn as nn

"""
Prescaler for state inputs, as further processed by other networks. Scale by bounded max, min values ot get data in [0,1]
"""


class DefaultPrescaler(nn.Module):
    max_vals: torch.Tensor
    min_vals: torch.Tensor

    def __init__(self, model):
        super().__init__()
        self.model = model

    def setup(self, action_space, observation_space, model_config):
        # for scaling later in pi method
        # Extract min and max values from the observation space
        self.min_vals = torch.zeros(
            len(observation_space.nvec)
        )  # Min values for each component (starts at 0)
        self.max_vals = torch.tensor(observation_space.nvec) - 1

        self.model.setup(
            action_space=action_space,
            observation_space=observation_space,
            model_config=model_config,
        )

    def scale_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Scale a batch of observations so that each feature is between 0 and 1,
        using the predefined minimum and maximum values for each feature from the observation_space.

        Args:
            obs (torch.Tensor): A batch of observations with shape [batch_size, observation_size].

        Returns:
            torch.Tensor: Scaled observations with the same shape as the input, with each feature in [0, 1].
        """

        # Ensure the input is a float tensor
        obs = obs.float()

        # Scale each feature (dimension) based on the predefined min and max values
        scaled_obs = (obs - self.min_vals) / (self.max_vals - self.min_vals)

        # Handle any division by zero (if max == min, the feature is constant across the batch)
        scaled_obs = torch.where(self.max_vals - self.min_vals == 0, obs, scaled_obs)

        return scaled_obs

    def forward(self, x):
        return self.model(self.scale_observations(x))


# redecide networks provide longer input
class DefaultPrescaler_Redecide(nn.Module):
    max_vals: torch.Tensor
    min_vals: torch.Tensor

    def __init__(self, model):
        super().__init__()
        self.model = model

    def setup(self, action_space, observation_space, model_config):
        # for scaling later in pi method
        # Extract min and max values from the observation space
        self.min_vals = torch.zeros(
            len(observation_space.nvec) + len(action_space.nvec) * action_space.nvec[0]
        )  # Min values for each component (starts at 0)
        obs_max = torch.tensor(observation_space.nvec) - 1
        act_max = torch.ones(len(action_space.nvec) * action_space.nvec[0]) * (
            action_space.nvec[0] - 1
        )
        self.max_vals = torch.cat([obs_max, act_max], dim=0)

        self.model.setup(
            action_space=action_space,
            observation_space=observation_space,
            model_config=model_config,
        )

    def scale_observations(self, obs: torch.Tensor) -> torch.Tensor:

        # Ensure the input is a float tensor
        obs = obs.float()

        # Scale each feature (dimension) based on the predefined min and max values
        scaled_obs = (obs - self.min_vals) / (self.max_vals - self.min_vals)

        # Handle any division by zero (if max == min, the feature is constant across the batch)
        scaled_obs = torch.where(self.max_vals - self.min_vals == 0, obs, scaled_obs)

        return scaled_obs

    def forward(self, x):
        return self.model(self.scale_observations(x))
