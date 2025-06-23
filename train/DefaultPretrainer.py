from datetime import datetime
from tools.global_config import (
    custom_bench_policy_data_path,
    custom_pretrained_models_path,
)
from train.config import ENV_CONFIGS
from train.environment import Matrix_production
from test.DefaultPolicyMaker import DefaultPolicyMaker
from torch import nn as nn
import torch
import torch.optim as optim
import hydra
import numpy as np
import torch.nn.functional as F

"""
Basic version of pretraining on a benchmark policy. Extend and tune for further use
"""


class DefaultPretrainer(nn.Module):
    """
    Collects data from an environment and pretrains a model with a shared encoder and policy head.

    Args:
        model_config (dict): Configuration for the model. Expected keys:
            - "shared_encoder"
            - "policy_head"
            - "environment_name" (optional)
        pretrain_config (dict): Pretraining settings. Expected keys:
            - "environment_name"
            - "policy_name"
            - "model_architecture"
            - "action_dims"
    """

    def __init__(self, model_config, pretrain_config):
        super().__init__()
        self.shared_encoder = hydra.utils.instantiate(model_config["shared_encoder"])
        self.policy_head = hydra.utils.instantiate(model_config["policy_head"])
        self.model_config = model_config

        self.environment_name = pretrain_config.get("environment_name", None)
        self.policy_name = pretrain_config.get("policy_name", None)
        self.model_architecture = pretrain_config.get("model_architecture", None)
        self.action_dims = pretrain_config.get("action_dims", None)

    def setup(self, observation_space, action_space):
        self.shared_encoder.setup(
            observation_space=observation_space,
            action_space=action_space,
            model_config=self.model_config,
        )

        self.policy_head.setup(
            observation_space=observation_space,
            action_space=action_space,
            model_config=self.model_config,
        )

    def forward(self, obs):
        x = self.shared_encoder(obs)
        logits = self.policy_head(x)
        return logits  # list of [logits_i] for each head

    def collect_data(self, n_steps=1000, version_number="overwritten"):
        print("\nStart data collection ...")

        obs_list, action_list = [], []

        selected_config = ENV_CONFIGS[self.environment_name]
        temp_env = Matrix_production(selected_config)  # Make sure this works properly
        pol_mk = DefaultPolicyMaker(self.policy_name, self.environment_name)
        bench_policy = pol_mk.policy

        for _ in range(n_steps):  # number of steps
            if _ % 1000 == 0:
                print("num datapoints ", _)
            obs, info = temp_env.reset()
            truncated = False
            while not truncated:
                action = bench_policy.compute_action(obs)
                obs_list.append(obs)
                action_list.append(action)
                obs, reward, terminated, truncated, info = temp_env.step(action)

        # Convert to tensors
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
        action_tensor = torch.tensor(np.array(action_list), dtype=torch.long)

        print("Data collection completed.")

        # save collected data
        custom_bench_policy_data_path.mkdir(parents=True, exist_ok=True)

        data_save_path = (
            custom_bench_policy_data_path
            / f"{self.environment_name}_{self.policy_name}_{self.model_architecture}_pretrain_data_V{version_number}.pth"
        )
        torch.save(
            {"obs_tensor": obs_tensor, "action_tensor": action_tensor}, data_save_path
        )

        print(f"Collected data saved to {data_save_path}")

        return obs_tensor, action_tensor

    def load_data(self, version_number="overwritten"):
        data_save_path = (
            custom_bench_policy_data_path
            / f"{self.environment_name}_{self.policy_name}_{self.model_architecture}_pretrain_data_V{version_number}.pth"
        )
        loaded_data = torch.load(data_save_path)
        obs_tensor = loaded_data["obs_tensor"]
        action_tensor = loaded_data["action_tensor"]
        print("\nCollected data reloaded successfully.")

        return obs_tensor, action_tensor

    def pretrain(
        self,
        *,
        epochs=100,
        entropy_coeff=0.01,
        batch_size=32,
        version_number="overwritten",
    ):

        # load data
        obs_tensor, action_tensor = self.load_data(version_number=version_number)

        print("\nStart pretraining ...")
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        num_samples = obs_tensor.size(0)

        for epoch in range(epochs):
            # Shuffle the data indices at the beginning of each epoch.
            permutation = torch.randperm(num_samples)
            epoch_loss = 0.0
            epoch_entropy = 0.0

            for i in range(0, num_samples, batch_size):
                indices = permutation[i : i + batch_size]
                obs_batch = obs_tensor[indices]
                action_batch = action_tensor[indices]

                logits = self(obs_batch)  # [batch_size, total_actions]
                loss = 0.0
                entropy_total = 0.0

                start = 0
                for j, dim in enumerate(self.action_dims):
                    end = start + dim
                    logit_head = logits[:, start:end]  # [batch_size, dim]
                    action_head = action_batch[:, j]  # [batch_size]

                    # Standard Cross Entropy loss
                    loss += loss_fn(logit_head, action_head)

                    # Entropy regularization
                    probs = F.softmax(logit_head, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                    entropy_total += entropy

                    start = end

                # Apply entropy bonus
                loss -= entropy_coeff * entropy_total

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_entropy += entropy_total.item()

            avg_loss = epoch_loss / ((num_samples + batch_size - 1) // batch_size)
            avg_entropy = epoch_entropy / ((num_samples + batch_size - 1) // batch_size)
            print(
                f"Epoch {epoch + 1}, Avg Loss = {avg_loss:.4f}, Avg Entropy = {avg_entropy:.4f}"
            )

            # Save weights after each epoch
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            encoder_path = (
                custom_pretrained_models_path
                / f"{self.environment_name}_{self.policy_name}_{self.model_architecture}_pretrain_data_V{version_number}_encoder_epoch_{epoch + 1}_{timestamp}.pth"
            )
            policy_path = (
                custom_pretrained_models_path
                / f"{self.environment_name}_{self.policy_name}_{self.model_architecture}_pretrain_data_V{version_number}_policy_head_epoch_{epoch + 1}_{timestamp}.pth"
            )
            torch.save(self.shared_encoder.state_dict(), encoder_path)
            torch.save(self.policy_head.state_dict(), policy_path)
            print(f"Saved encoder weights to {encoder_path}")
            print(f"Saved policy head weights to {policy_path}")
