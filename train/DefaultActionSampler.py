import torch
from tools.tools_numpy import build_downstream_mask
from train.sample_actions import _sample_actions

"""
Samples actions based on a downstream constraint, and other actions --> sequential sampling in _sample_actions
"""


class DefaultActionSampler:
    def __init__(self, downstream_relations):
        # a dictionary that maps cell index to possible next locations under the downstream constraint
        # cell 0 is the most right and bottom cell, then first go up before going left
        # same indexing like state/actions
        self.downstream_relations = downstream_relations

    def setup(self, action_space, observation_space, model_config):
        self.num_cells = action_space.nvec.size  # Anzahl der Zellen
        self.num_actions = action_space.nvec[0]

        # with downstream relations create a mask tensor, this is not batchsize specific! extend to batchsize later!
        # Initialize a False (0) mask tensor without batch dimension
        self.downstream_mask = build_downstream_mask(
            num_cells=self.num_cells,
            num_actions_per_cell=self.num_actions,
            downstream_relations=self.downstream_relations,
        )

    def sample(
        self, logits, inference, obs_unscaled
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return _sample_actions(
            downstream_mask=self.downstream_mask,
            logits=logits,
            inference=inference,
            obs_unscaled=obs_unscaled,
        )
