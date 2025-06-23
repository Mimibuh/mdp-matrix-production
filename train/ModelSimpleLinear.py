import torch
from torch import nn as nn

""" SHARED ENCODER
Simple linear mdoel
"""


class ModelSimpleLinear(nn.Module):
    def __init__(
        self,
        *,
        grid_size: tuple[int, int],
        out_features: int,
        inner_features: int = 256,
    ) -> None:
        super().__init__()

        self.grid_size = grid_size
        self.inner_features = inner_features
        self.out_features = out_features

    def setup(self, action_space, observation_space, model_config):
        self.layer = nn.Sequential(
            nn.Linear(observation_space.shape[0], self.inner_features),
            nn.ReLU(),
            nn.Linear(self.inner_features, self.out_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.grid_size[0] * self.grid_size[1] * 6 + 4

        return self.layer(x)
