import numpy as np
import torch
from ray.rllib.models.torch.torch_distributions import (
    TorchCategorical,
    TorchDistribution,
)
from torch import Tensor

from train.sample_actions import compute_mask_new, sample_and_fill_logits_mask

"""
Custom probability dsitribution, needed for RLlib. Can sample actions and calc logp fro training.
"""


class DefaultAutoregressiveDistribution(TorchDistribution):
    def __init__(self, logits: Tensor, model=None):
        super().__init__(logits)
        self.logits = logits  # (B, C, A) -> logits
        self.model = model
        if model:
            self.obs_unscaled = model.last_obs_unscaled
            self.downstream_mask = model.action_sampler.downstream_mask

    def set_model(self, model):
        """
        A method to set the model data after distribution creation.
        This allows accessing the model in distribution methods.
        """
        self.model = model
        self.obs_unscaled = model.last_obs_unscaled
        self.downstream_mask = model.action_sampler.downstream_mask

    def sample(self):
        """
        Sample actions sequentially conditioned on previous actions.
        """
        actions, logp = self.model.action_sampler.sample(
            logits=self.logits, inference=False, obs_unscaled=self.obs_unscaled
        )
        self._sampled_actions = actions
        self._logp = logp
        return actions

    def deterministic_sample(self):
        """
        Sample actions sequentially conditioned on previous actions (deterministic).
        """
        actions, logp = self.model.action_sampler.sample(
            logits=self.logits, inference=True, obs_unscaled=self.obs_unscaled
        )
        self._sampled_actions = actions
        self._logp = logp
        return actions

    def sampled_action_logp(self):
        """
        Return the log probability of the sampled actions.
        """
        return self._logp

    def logp(self, actions: Tensor):
        """
        Calculate log probability of actions sequentially, conditioned on previous actions.
        """

        batch_size, num_cells, num_actions = self.logits.shape
        total_logp = torch.zeros(batch_size, device=self.logits.device)
        prev_actions = torch.zeros(
            (batch_size, 0), dtype=torch.int64, device=self.logits.device
        )

        for cell in range(num_cells):
            logits_i = self.logits[:, cell, :]

            mask_i_np = np.ones((batch_size, num_actions), dtype=np.bool)

            # Compute mask based on previous actions, downstream_mask, and observation
            compute_mask_new(
                cell=cell,
                out_mask=mask_i_np,
                num_cells=num_cells,
                num_actions=num_actions,
                downstream_mask=self.downstream_mask.detach().cpu().numpy(),
                obs_unscaled=self.obs_unscaled.detach().cpu().numpy(),
                prev_actions=prev_actions.cpu().numpy(),
            )

            mask_i = torch.from_numpy(mask_i_np).to(self.logits.device)
            logits_i_masked = torch.where(
                mask_i, logits_i, torch.tensor(-1000.0, device=self.logits.device)
            )

            dist = TorchCategorical(logits=logits_i_masked)
            logp_i = dist.logp(actions[:, cell])
            total_logp += logp_i

            prev_actions = torch.cat((prev_actions, actions[:, cell : cell + 1]), dim=1)

        return total_logp

    def entropy(self):
        """
        Calculate the entropy of the distribution sequentially.
        """
        batch_size, num_cells, num_actions = self.logits.shape
        out_logits_mask = np.ones((batch_size, num_cells, num_actions), dtype=np.bool)

        sample_and_fill_logits_mask(
            logits=self.logits,
            out_actions=np.zeros((batch_size, num_cells), dtype=np.int64),
            out_logits_mask=out_logits_mask,
            inference=False,
            downstream_mask=self.downstream_mask,
            obs_unscaled=self.obs_unscaled,
        )

        logits_masked = torch.where(
            torch.from_numpy(out_logits_mask).to(self.logits.device),
            self.logits,
            torch.tensor(-1000.0, dtype=torch.float32, device=self.logits.device),
        )

        dist = TorchCategorical(logits=logits_masked)
        entropies = dist.entropy()
        return entropies.sum(dim=-1)

    def kl(self, other):
        """
        Calculate Kullback-Leibler divergence between two distributions.
        Calculates only a rough approximation of the KL divergence, not used in final training.
        """
        assert isinstance(other, DefaultAutoregressiveDistribution)
        assert hasattr(
            self, "_sampled_actions"
        ), "KL requires that sample() or deterministic_sample() has been called first."

        batch_size, num_cells, num_actions = self.logits.shape
        total_kl = torch.zeros(batch_size, device=self.logits.device)
        prev_actions = torch.zeros(
            (batch_size, 0), dtype=torch.int64, device=self.logits.device
        )
        actions = self._sampled_actions

        for cell in range(num_cells):
            logits_p = self.logits[:, cell, :]
            logits_q = other.logits[:, cell, :]

            mask_i_np = np.ones((batch_size, num_actions), dtype=np.bool)

            compute_mask_new(
                cell=cell,
                out_mask=mask_i_np,
                num_cells=num_cells,
                num_actions=num_actions,
                downstream_mask=self.downstream_mask.detach().cpu().numpy(),
                obs_unscaled=self.obs_unscaled.detach().cpu().numpy(),
                prev_actions=prev_actions.cpu().numpy(),
            )

            mask_i = torch.from_numpy(mask_i_np).to(self.logits.device)
            logits_p_masked = torch.where(
                mask_i, logits_p, torch.tensor(-1000.0, device=self.logits.device)
            )
            logits_q_masked = torch.where(
                mask_i, logits_q, torch.tensor(-1000.0, device=self.logits.device)
            )

            dist_p = TorchCategorical(logits=logits_p_masked)
            dist_q = TorchCategorical(logits=logits_q_masked)
            kl_i = dist_p.kl(dist_q)
            total_kl += kl_i

            prev_actions = torch.cat((prev_actions, actions[:, cell : cell + 1]), dim=1)

        return total_kl

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """
        Return the required shape of the model output (e.g., number of actions).
        """
        return np.prod(action_space.shape)

    @staticmethod
    def required_input_dim(space, **kwargs):
        """
        Returns the required input dimension for this distribution.
        The input dimension depends on the shape of the action space.
        For autoregressive distributions, this typically relates to the number of cells (steps in the sequence).
        """
        return len(space.nvec)

    def _get_torch_distribution(self, *args, **kwargs):
        # print("here logits", self.logits)
        """
        Create and return the torch distribution. In this case, it's a Categorical distribution.
        The logits are used to define the probabilities over the actions.
        """
        return TorchCategorical(logits=args[0])

    @staticmethod
    def from_logits(logits, model=None):
        """
        This is where the distribution is initialized from the logits.
        If the model is available, we store it.
        """
        # Initialize the distribution with logits and optionally pass the model
        dist = DefaultAutoregressiveDistribution(logits=logits, model=model)

        # Set the model data if it's available
        if model:
            dist.set_model(model)

        return dist
