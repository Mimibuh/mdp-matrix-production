# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:10:10 2025

@author: mimib
"""
from custom_model_2 import MultiHeadActionsRLM
from mimi_env import Desp


from ray.tune.registry import register_env
import torch


def env_creator(env_config):
    # from path.to.custom.env import MyEnv  # your custom env class
    return Desp(env_config)


register_env("MyEnv-v0", env_creator)


dummy_env = Desp()
obs_space = dummy_env.observation_space
print(type(obs_space))
act_space = dummy_env.action_space


model_config = {
    "grid_size": (1, 2),
    "fcnet_hiddens": [256, 256],
    "action_digits": 4,
    "total_number_action_components": 20,
    # Add other configurations as needed
}


model = MultiHeadActionsRLM(
    observation_space=obs_space,
    action_space=act_space,
    model_config=model_config,
    inference_only=False,
)
model.setup()


# Assume `env` is your real environment
observation_space = dummy_env.observation_space

# Create a mock batch of observations (batch size = 4, for example)
batch_size = 4
mock_obs = torch.tensor(
    [observation_space.sample() for _ in range(batch_size)], dtype=torch.float32
)

# Create the mock batch in RLlib format
mock_batch = {"obs": mock_obs}

# Print the mock batch
print("Mock Batch:", mock_batch)

# Test inference mode
inference_output = model._forward_inference(mock_batch)
print("Inference Output:", inference_output)

# Test exploration mode
exploration_output = model._forward_exploration(mock_batch)
print("Exploration Output:", exploration_output)

# Test value function computation
value_function_output = model.compute_values(mock_batch)
print("Value Function Output:", value_function_output)
