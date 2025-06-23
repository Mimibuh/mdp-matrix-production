from torch import nn as nn

"""
Policy value network (critic), outputs single scalar state value used for loss calculation.
"""


class DefaultValueHead(nn.Module):
    def __init__(self):
        super().__init__()

        self._value_net = nn.Linear(256, 1)

    def setup(self, action_space, observation_space, model_config):
        pass

    def forward(self, base_embed):
        return self._value_net(base_embed)
