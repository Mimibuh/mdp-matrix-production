# RL Control in Matrix Production Systems

This repository contains code for training and evaluating reinforcement learning (RL) policies in matrix production systems. It includes training utilities, rule-based baselines, and scripts for testing and analysis.

---

## Directory Structure

### `train/`
This folder contains all components related to training:

- Configuration files defining matrix system settings and parameters .
- All classes required for environment setup, agent definition, and model training.
- The reward function is defined in `tools/tool_reward_shaping.py`.  
  To use a specific reward formulation, its name must be specified in the corresponding configuration file.

### `tools/`
Utility modules used throughout the project:

- Includes supporting functions for training, logging, reward shaping, and plotting.
- The module `tool_reward_shaping.py` is especially important, as it defines the reward functions used during training.

### `test/`
Provides tools for model evaluation:

- Contains validation routines and rule-based benchmark policies used for comparison.
- The `Validator` class handles evaluation of trained policies on test scenarios.

---

## Executable Scripts

### `experiments/training_runs/`
Main entry point for training experiments:

- Each script defines a specific training run, including:
  - The matrix configuration used
  - The encoder architecture (e.g., transformer or CNN)
  - Key hyperparameters (e.g., learning rate, batch size)
- These scripts are used to train RL models via PPO.

### `experiments/interesting_runs/`
Contains scripts for evaluation and comparison:

- These scripts load trained models and print test results.
- Includes comparisons against rule-based baselines on selected scenarios.

### `experiments/trained_models/`
Directory for the trained models used in evaluation and benchmarking.

---

## Archive

### `Archive/`
Contains archived files, mainly training runs.  
These are not actively used but are kept for documentation and reproducibility.
