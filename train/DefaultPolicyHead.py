from torch import nn as nn
from tools.global_config import custom_pretrained_models_path
import torch

"""
Policy head network (actor), outputs logits (later probability distribution through softmax) for each subaction to be chosen
"""


class DefaultPolicyHead(nn.Module):
    def __init__(self):
        super().__init__()

    def setup(self, action_space, observation_space, model_config):
        total_actions = sum(action_space.nvec)

        self._policy_head = nn.Linear(256, total_actions)

        if "load_pretraining" in model_config:
            if model_config["load_pretraining"]:
                path = (
                    custom_pretrained_models_path
                    / f"{model_config['pretrain_path']}_policy_head.pth"
                )
                self.load_state_dict(torch.load(path))

    def forward(self, base_embed):
        return self._policy_head(base_embed)
