# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:35:27 2025

@author: mimib
"""

from tools.global_config import custom_checkpoint_path
from mimi_env import Desp
from custom_model_2 import MultiHeadActionsRLM
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from datetime import datetime
from train.config import ENV_CONFIGS

ray.shutdown()
ray.init(log_to_driver=True, logging_level="INFO")


environment_name = "X1_stochastic"


def env_creator(env_config):
    print("Creating environment with env_config:", env_config)
    return Desp(env_config)  # Ensure Desp can retrieve env_config internally


register_env(environment_name, lambda config: env_creator(config))

# Use the default or specified config
selected_config = ENV_CONFIGS[environment_name]

# Create a temporary environment to extract params
temp_env = Desp(selected_config)  # Make sure this works properly

# Extract required parameters from the environment
gs = temp_env.grid_size  # Assuming your env exposes these attributes
ad = temp_env.action_decoding
mb = temp_env.machine_or_buffer
vabrs = temp_env.dic_valid_actions_by_remaining_stages
final_indices = temp_env.final_indices_by_pgs


ModelCatalog.register_custom_model("multi_head_dict_model", MultiHeadActionsRLM)

# Define the RLModuleSpec with correct parameters
module_spec = RLModuleSpec(
    module_class=MultiHeadActionsRLM,
    model_config={
        "grid_size": gs,
        "action_decoding": ad,
        "machine_or_buffer": mb,
        "dic_valid_actions_by_remaining_stages": vabrs,
        "final_indices_by_pgs": final_indices,
    },
    catalog_class=None,
)

config = (
    PPOConfig()
    .environment(env=environment_name, env_config=selected_config)
    .framework("torch")
    .rl_module(rl_module_spec=module_spec)
    .env_runners(num_env_runners=0)
)

trainer = config.build()

# Make sure the path exists
custom_checkpoint_path.mkdir(parents=True, exist_ok=True)
# Create a timestamp for unique checkpoint naming
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Define the checkpoint filename with timestamp
checkpoint_filename = f"OuterOrder_{environment_name}_t_{timestamp}"
checkpoint_path = custom_checkpoint_path / checkpoint_filename


# Start training the PPO agent
for iteration in range(1):  # Set the number of iterations you want to train for
    print("train iteration: ", iteration)
    result = trainer.train()
    checkpoint_path = trainer.save_to_path(str(checkpoint_path))
    print("Checkpoint saved to:", checkpoint_path)


# Reload and test trained model
# rl_module_path = pathlib.Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
# rl_module = RLModule.from_checkpoint(str(rl_module_path))["default_policy"]
# print("RLModule loaded successfully.")

# Test environment to generate an observation
# test_env = gymnasium.make("Matrix_Prod_v0")
# obs, info = test_env.reset()  # Reset the environment to get an initial state
# Convert observation to a batch (batch size = 1)
# obs_batch = torch.from_numpy(np.array([obs]))
# Compute action logits
# output = rl_module.forward_inference({"obs": obs_batch})
# print(output["actions"])
