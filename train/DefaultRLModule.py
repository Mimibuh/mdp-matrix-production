from typing import Dict

import hydra
import torch
from einops import rearrange
from gymnasium.spaces import MultiDiscrete
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils import override
from torch.nn import Module

from train.DefaultActionSampler import DefaultActionSampler
from train.DefaultAutoregressiveDistribution import DefaultAutoregressiveDistribution


class DefaultRLModule(TorchRLModule, ValueFunctionAPI):
    """
    An RLModule for multi-head autoregressive policies. Core component of training pipeline

    This module predicts all subactions based only on the input state.

    Sampling method ensures hard constraints:
        - No duplicate action locations
        - workpieces can only move downstream
        - unoccupied choose none

    Soft constraints need to be handled by the environment:
        - No service at failed machines
        - action invalid if no workpieces is present
        - production plan reward penalty
        - not finihed reward penalty
        - order reward penalty

    includes
    - shared encoder (backbone network)
    - policy head (actor)
    - value head (critic)
    - action sampler (sampling method for actions)

    """

    action_space: MultiDiscrete
    observation_space: MultiDiscrete

    action_sampler: DefaultActionSampler
    shared_encoder: Module
    value_net: Module
    policy_head: Module
    num_actions: int
    num_cells: int
    action_dist_input_lens: list[int]

    @override(RLModule)
    def setup(self):
        super().setup()

        self.num_cells = self.action_space.nvec.size  # Anzahl der Zellen
        self.num_actions = self.action_space.nvec[0]

        model_config = self.model_config
        self.action_sampler = hydra.utils.instantiate(model_config["action_sampler"])
        self.shared_encoder = hydra.utils.instantiate(model_config["shared_encoder"])
        self.policy_head = hydra.utils.instantiate(model_config["policy_head"])
        self.value_net = hydra.utils.instantiate(model_config["value_head"])

        self.action_sampler.setup(
            action_space=self.action_space,
            observation_space=self.observation_space,
            model_config=model_config,
        )
        self.shared_encoder.setup(
            action_space=self.action_space,
            observation_space=self.observation_space,
            model_config=model_config,
        )
        self.policy_head.setup(
            action_space=self.action_space,
            observation_space=self.observation_space,
            model_config=model_config,
        )
        self.value_net.setup(
            action_space=self.action_space,
            observation_space=self.observation_space,
            model_config=model_config,
        )

        self.action_dist_input_lens = list(self.action_space.nvec)

    def get_inference_action_dist_cls(self):
        self0 = self

        # workaround for from_logits function, requirement for RLlib
        class my_distribution(DefaultAutoregressiveDistribution):
            def __init__(self, logits):
                super().__init__(logits, model=self0)

            def from_logits(logits, model=None):
                return my_distribution(logits)

        return my_distribution

    def _pi(self, obs: torch.Tensor, inference: bool) -> Dict[str, torch.Tensor]:
        # process state through shared encoder
        base_embed = self.shared_encoder(obs)

        # process base embedding through policy head, returns predicted logits
        logits = self.policy_head(base_embed)

        logits_ = rearrange(
            logits, "b (c a)->b c a", c=self.num_cells, a=self.num_actions
        )

        # actions sampler returns complete action, sampled sequentially, also returns logp for PPO
        actions, logp = self.action_sampler.sample(
            logits=logits_, inference=inference, obs_unscaled=obs
        )

        self.last_obs_unscaled = obs

        # return actions, logp and logits for further processing
        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: logp,
            Columns.ACTION_DIST_INPUTS: logits_,
        }

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

        # scale observations
        obs = batch[Columns.OBS].float()
        base_embed = self.shared_encoder(obs)

        vf_out = self.value_net(base_embed)
        return vf_out.squeeze(-1)


class DefaultRLModule_Redecision(TorchRLModule, ValueFunctionAPI):
    """
    An RLModule for multi-head autoregressive policies.

    This module predicts all subactions based only on the input state.

    Sampling method ensures hard constraints:
        - No duplicate action locations
        - workpieces can only move downstream

    Soft constraints need to be handled by the environment:
        - No service at failed machines
        - action invalid if no workpieces is present
        - production plan reward penalty
        - not finihed reward penalty
        - order reward penalty


    same logic as DefaultRLModule, but with redecision mechanism
    --> redecision shared encoder makes a logit decision, receives the state again concatenated with the logits and makes a new logit prediction



    """

    action_space: MultiDiscrete
    observation_space: MultiDiscrete

    action_sampler: DefaultActionSampler
    shared_encoder: Module
    value_net: Module
    policy_head: Module
    num_actions: int
    num_cells: int
    action_dist_input_lens: list[int]

    @override(RLModule)
    def setup(self):
        super().setup()

        self.num_cells = self.action_space.nvec.size  # Anzahl der Zellen
        self.num_actions = self.action_space.nvec[0]

        model_config = self.model_config
        self.action_sampler = hydra.utils.instantiate(model_config["action_sampler"])
        self.shared_encoder = hydra.utils.instantiate(model_config["shared_encoder"])
        self.policy_head = hydra.utils.instantiate(model_config["policy_head"])
        self.value_net = hydra.utils.instantiate(model_config["value_head"])

        self.action_sampler.setup(
            action_space=self.action_space,
            observation_space=self.observation_space,
            model_config=model_config,
        )
        self.shared_encoder.setup(
            action_space=self.action_space,
            observation_space=self.observation_space,
            model_config=model_config,
        )
        self.policy_head.setup(
            action_space=self.action_space,
            observation_space=self.observation_space,
            model_config=model_config,
        )
        self.value_net.setup(
            action_space=self.action_space,
            observation_space=self.observation_space,
            model_config=model_config,
        )

        self.action_dist_input_lens = list(self.action_space.nvec)

    def get_inference_action_dist_cls(self):
        self0 = self

        class my_distribution(DefaultAutoregressiveDistribution):
            def __init__(self, logits):
                super().__init__(logits, model=self0)

            def from_logits(logits, model=None):
                return my_distribution(logits)

        return my_distribution

    def _pi(self, obs: torch.Tensor, inference: bool) -> Dict[str, torch.Tensor]:
        """
        Compute all action logits at once using only the state.
        """
        batch_size = obs.shape[0]
        redecide = torch.zeros(batch_size, self.num_actions * self.num_cells)

        input = torch.cat((obs, redecide), dim=1)

        base_embed = self.shared_encoder(input)
        # base_embed = self.shared_encoder(obs)

        logits = self.policy_head(base_embed)

        base_embed_redecide = self.shared_encoder(torch.cat((obs, logits), dim=1))

        logits_redecide = self.policy_head(base_embed_redecide)

        logits_ = rearrange(
            logits_redecide, "b (c a)->b c a", c=self.num_cells, a=self.num_actions
        )
        actions, logp = self.action_sampler.sample(
            logits=logits_, inference=inference, obs_unscaled=obs
        )

        self.last_obs_unscaled = obs

        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: logp,
            Columns.ACTION_DIST_INPUTS: logits_,
        }

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

        # scale observations
        obs = batch[Columns.OBS].float()

        batch_size = obs.shape[0]

        redecide = torch.zeros(batch_size, self.num_actions * self.num_cells)

        input = torch.cat((obs, redecide), dim=1)
        base_embed = self.shared_encoder(input)

        vf_out = self.value_net(base_embed)
        return vf_out.squeeze(-1)
