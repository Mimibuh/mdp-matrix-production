# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:22:38 2025

@author: mimib
"""
import torch
import numpy as np
from ray.rllib.core.rl_module import RLModule
import pathlib
import gymnasium
from gymnasium.envs.registration import register

from tools.global_config import custom_checkpoint_path

# Define the custom checkpoint directory

# Make sure the path exists
custom_checkpoint_path.mkdir(parents=True, exist_ok=True)

register(
    id="Matrix_Prod_v0",  # Environment name
    entry_point="mimi_env:Desp",  # Path to your environment class  # Custom arguments for your environment
)

checkpoint_filename = "X1_stochastic"
checkpoint_path = custom_checkpoint_path / checkpoint_filename
rl_module_path = (
    pathlib.Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
)

# Load the RLModule

rl_module = RLModule.from_checkpoint(str(rl_module_path))["default_policy"]
print("RLModule loaded successfully.")

# Test environment to generate an observation
test_env = gymnasium.make("Matrix_Prod_v0")
obs, info = test_env.reset()  # Reset the environment to get an initial state

total_reward = 0


for t in range(100):
    print("\ntime = ", t)

    # Convert observation to a batch (batch size = 1)
    obs_batch = torch.from_numpy(np.array([obs]))
    output = rl_module.forward_inference({"obs": obs_batch})
    logits = rl_module._pi(obs_batch, False)
    print("interm: inference False: ", logits["actions"][0])
    print("final: inference True : ", output["actions"][0])

    # obs, reward, terminated, truncated, info_dic = test_env.step(logits["actions"][0])
    obs, reward, terminated, truncated, info_dic = test_env.step(output["actions"][0])

    total_reward += reward

    if terminated:
        break

# print last state
print(obs)
