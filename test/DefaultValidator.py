# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 18:44:19 2025

@author: mimib
"""
from train.config import ENV_CONFIGS
from train.environment import Matrix_production
import numpy as np
import torch
import random
import copy

"""
Validator to compare runs, do not change the configs of the validator.

Only thing to change is seeds:  0-99 for validation, 100-199 for test set.

For meaningful rlmodel vs hybrid comparison (random seed issue) use the corresponding "no_batch" functions
"""


# create validation instances


class DefaultValidator:
    def __init__(
        self,
        environment_name,
        test_seeds=list(range(100)),
        test_steps=200,
        reset_strategy=None,
    ):
        self.test_seeds = test_seeds
        self.selected_config = ENV_CONFIGS[environment_name]

        self.test_steps = test_steps

        # overwrite for reproducability
        self.selected_config["max_steps"] = self.test_steps
        self.selected_config["verbosity_env"] = 0

        if reset_strategy:
            self.selected_config["reset_strategy"] = reset_strategy
        else:
            self.selected_config["reset_strategy"] = {
                "start_zero": False,
                "utilization": 0.6,
                "control_done_index": True,
            }
        reward_params = dict(
            finished_wps_reward_base=1.5,
            beta=0.9,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        )

        self.test_envs = []

        for s in self.test_seeds:
            new_env = Matrix_production(self.selected_config)
            new_env.reset(seed=s)
            new_env.observation_space.seed(s)

            self.test_envs.append(new_env)

    def reset_envs(self):
        for index, seed in enumerate(self.test_seeds):
            self.test_envs[index].reset(seed=seed)
            # self.test_envs[index].observation_space.seed(seed)

    def test_rl_model(self, rl_module):
        self.reset_envs()

        print("Test model on validation set...")
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)  # need this when using the model!
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rl_module.eval()

        # Initialize reward tracker for each environment
        num_envs = len(self.test_envs)
        reward_tracker = np.zeros(
            num_envs, dtype=np.float32
        )  # Tracks cumulative rewards

        actions_per_env = [[] for _ in range(len(self.test_envs))]

        obs = [env.state for env in self.test_envs]
        for step in range(self.test_steps):
            obs_batch = np.array(obs, dtype=np.float32)
            results = rl_module._pi(
                torch.tensor(obs_batch, dtype=torch.float32), inference=True
            )
            # Execute each action in its corresponding environment
            batch_actions = results["actions"]
            batch_results = [
                env.step(action) for env, action in zip(self.test_envs, batch_actions)
            ]

            # Store each action per environment
            for i, action in enumerate(batch_actions):
                actions_per_env[i].append(action.cpu().numpy().copy())

            # Unpack batch results
            obs, rewards, dones, truncs, infos = zip(*batch_results)

            # Convert to NumPy for further processing
            obs = np.array(obs, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=bool)
            truncs = np.array(truncs, dtype=bool)

            # Accumulate rewards for each environment
            reward_tracker += rewards  # Adds step rewards to total rewards

        buffer_wps_tracker = [env.in_buffer_waiting for env in self.test_envs]

        results = {
            "no_cor_both_finishes": np.array(
                [env.correct_both_finishes for env in self.test_envs]
            ),
            "no_cor_order_finishes": np.array(
                [env.correct_order_finishes for env in self.test_envs]
            ),
            "no_cor_plan_finishes": np.array(
                [env.correct_plan_finishes for env in self.test_envs]
            ),
            "no_total_finishes": np.array(
                [env.number_of_finished_wps for env in self.test_envs]
            ),
            "rewards": reward_tracker,
        }

        # Add pieces in buffer if new arrival process activated
        if not self.selected_config.get("old_arrival_process"):
            results["in_buffer_waiting_over_time"] = np.array(buffer_wps_tracker)

        print("Completed validation test.\n")

        return self.test_seeds, results, actions_per_env

    def test_rl_model_nobatch(self, rl_module):
        self.reset_envs()

        print("Test model on validation set...")
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rl_module.eval()

        num_envs = len(self.test_envs)
        all_env_rewards = []
        actions_per_env = [[] for _ in range(num_envs)]
        all_env_in_buffer_waiting_over_time = []

        for i, env in enumerate(self.test_envs):
            obs = env.state
            reward_tracker = 0

            for step in range(self.test_steps):
                obs_batch = np.array([obs], dtype=np.float32)
                results = rl_module._pi(
                    torch.tensor(obs_batch, dtype=torch.float32), inference=True
                )
                action = results["actions"].flatten()

                # Store the action
                actions_per_env[i].append(action.cpu().numpy().copy())

                new_obs, reward, done, trunc, info = env.step(action)
                obs = new_obs
                reward_tracker += reward

                if done or trunc:
                    break  # Stop stepping this environment early

            all_env_rewards.append(reward_tracker)
            all_env_in_buffer_waiting_over_time.append(env.in_buffer_waiting)

        results = {
            "no_cor_both_finishes": np.array(
                [env.correct_both_finishes for env in self.test_envs]
            ),
            "no_cor_order_finishes": np.array(
                [env.correct_order_finishes for env in self.test_envs]
            ),
            "no_cor_plan_finishes": np.array(
                [env.correct_plan_finishes for env in self.test_envs]
            ),
            "no_total_finishes": np.array(
                [env.number_of_finished_wps for env in self.test_envs]
            ),
            "rewards": np.array(all_env_rewards),
        }

        if not self.selected_config.get("old_arrival_process"):
            results["in_buffer_waiting_over_time"] = np.array(
                all_env_in_buffer_waiting_over_time
            )

        print("Completed validation test.\n")
        return self.test_seeds, results, actions_per_env

    def test_own_policy(self, policy):
        self.reset_envs()

        print("Test own policy on validation set...")
        random.seed(1)
        np.random.seed(1)

        # Initialize reward tracker for each environment
        num_envs = len(self.test_envs)
        reward_tracker = np.zeros(
            num_envs, dtype=np.float32
        )  # Tracks cumulative rewards

        obs = [env.state for env in self.test_envs]

        all_env_rewards = []
        all_env_in_buffer_waiting_over_time = []

        for i, env in enumerate(self.test_envs):
            reward_tracker = 0
            for step in range(self.test_steps):
                action = policy.compute_action(obs[i])
                new_obs, reward, done, trunc, info = env.step(action)
                obs[i] = new_obs
                reward_tracker += reward
                if done or trunc:
                    break  # Exit early if the episode is over.

            all_env_in_buffer_waiting_over_time.append(env.in_buffer_waiting)
            all_env_rewards.append(reward_tracker)

        print("Completed validation test.\n")

        results = {
            "no_cor_both_finishes": np.array(
                [env.correct_both_finishes for env in self.test_envs]
            ),
            "no_cor_order_finishes": np.array(
                [env.correct_order_finishes for env in self.test_envs]
            ),
            "no_cor_plan_finishes": np.array(
                [env.correct_plan_finishes for env in self.test_envs]
            ),
            "no_total_finishes": np.array(
                [env.number_of_finished_wps for env in self.test_envs]
            ),
            "rewards": np.array(all_env_rewards),
        }

        # Add pieces in buffer if new arrival process activated
        if not self.selected_config.get("old_arrival_process"):
            results["in_buffer_waiting_over_time"] = all_env_in_buffer_waiting_over_time

        return self.test_seeds, results

    def test_hybrid_nobatch(self, rl_module, policy):
        self.reset_envs()

        print("Test model on validation set...")
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rl_module.eval()

        # Initialize reward tracker for each environment
        num_envs = len(self.test_envs)
        all_env_rewards = []
        count_invalid_model_actions = 0
        all_env_in_buffer_waiting_over_time = []
        all_env_count_invalid_model_actions = []

        # Save RNG state before creating dummy
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        py_state = random.getstate()

        dummy_test_env = Matrix_production(self.selected_config)

        # Restore RNG state immediately
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        random.setstate(py_state)

        all_env_in_buffer_waiting_over_time = []

        for i, env in enumerate(self.test_envs):
            obs = env.state
            reward_tracker = 0
            count_invalid_model_actions = 0

            for step in range(self.test_steps):
                obs_batch = np.array([obs], dtype=np.float32)
                results = rl_module._pi(
                    torch.tensor(obs_batch, dtype=torch.float32), inference=True
                )
                action = results["actions"].flatten()

                # Save RNG state *after* RL action
                np_state = np.random.get_state()
                torch_state = torch.get_rng_state()
                random_state = random.getstate()

                # Check if the action is valid
                dummy_test_env.state = copy.deepcopy(obs)
                no_wrong_plan_before = (
                    dummy_test_env.number_of_finished_wps
                    - dummy_test_env.correct_plan_finishes
                )

                # Perform dummy step (may consume RNG)
                # Save global RNG state

                # Perform the dummy step (consumes RNG)
                new_obs, reward, done, trunc, info = dummy_test_env.step(action)

                # Restore RNG state so the rest of the test behaves as if nothing happened

                no_wrong_plan_after = (
                    dummy_test_env.number_of_finished_wps
                    - dummy_test_env.correct_plan_finishes
                )
                s, count = dummy_test_env.count_too_far_right_wps(action[-1])

                if count > 0 or no_wrong_plan_after > no_wrong_plan_before:
                    print(
                        f"Environment {i} step {step}: Model action invalid, using policy instead"
                    )
                    # print("action: ", action)
                    # print("obs: ", obs[i])

                    # If the action is invalid, use the policy to compute a new action

                    # Perform the dummy step (consumes RNG)
                    action = policy.compute_action(obs)

                    # Restore RNG state so the rest of the test behaves as if nothing happened

                    count_invalid_model_actions += 1

                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                random.setstate(random_state)
                # conduct step with chosen action
                new_obs, reward, done, trunc, info = env.step(action)

                obs = new_obs
                reward_tracker += reward
                if done or trunc:
                    break  # Exit early if the episode is over.

                # print("number of unfinished exits: ", )

            all_env_rewards.append(reward_tracker)
            all_env_in_buffer_waiting_over_time.append(env.in_buffer_waiting)

        results = {
            "no_cor_both_finishes": np.array(
                [env.correct_both_finishes for env in self.test_envs]
            ),
            "no_cor_order_finishes": np.array(
                [env.correct_order_finishes for env in self.test_envs]
            ),
            "no_cor_plan_finishes": np.array(
                [env.correct_plan_finishes for env in self.test_envs]
            ),
            "no_total_finishes": np.array(
                [env.number_of_finished_wps for env in self.test_envs]
            ),
            "rewards": np.array(all_env_rewards),
        }

        if not self.selected_config.get("old_arrival_process"):
            results["in_buffer_waiting_over_time"] = np.array(
                all_env_in_buffer_waiting_over_time
            )

        print("Completed validation test.\n")
        return self.test_seeds, results

    def test_hybrid(self, *, rl_module, policy):
        self.reset_envs()

        print("Test model on validation set...")
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)  # need this when using the model!
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rl_module.eval()

        # Initialize reward tracker for each environment
        num_envs = len(self.test_envs)
        reward_tracker = np.zeros(
            num_envs, dtype=np.float32
        )  # Tracks cumulative rewards

        obs = [env.state for env in self.test_envs]

        all_env_rewards = []
        count_invalid_model_actions = 0
        all_env_in_buffer_waiting_over_time = []
        all_env_count_invalid_model_actions = []

        dummy_test_env = Matrix_production(self.selected_config)

        for i, env in enumerate(self.test_envs):
            reward_tracker = 0
            count_invalid_model_actions = 0
            for step in range(self.test_steps):
                # print("current queue: ", obs[i][-4:])
                obs_batch = np.array([obs[i]], dtype=np.float32)
                results = rl_module._pi(
                    torch.tensor(obs_batch, dtype=torch.float32), inference=True
                )
                action = results["actions"].flatten()

                # Check if the action is valid
                # dummy_test_env.state = copy.deepcopy(obs[i])
                # no_wrong_plan_before = (
                #    dummy_test_env.number_of_finished_wps
                #    - dummy_test_env.correct_plan_finishes
                # )

                # Save RNG state before dummy step
                np_state = np.random.get_state()
                before = np.random.get_state()
                before_rng = torch.get_rng_state()

                # Perform dummy step (may consume RNG)
                # new_obs, reward, done, trunc, info = dummy_test_env.step(action)

                # Restore RNG state immediately after
                np.random.set_state(np_state)
                after = np.random.get_state()
                after_rng = torch.get_rng_state()

                if (np.array_equal(before[1], after[1])) == False:
                    print("randomness changed")

                if (np.array_equal(before_rng[1], after_rng[1])) == False:
                    print("RNG randomness changed")
                # print("torch RNG: ", (np.array_equal(before[1], after[1])))  # Should be True if no mutation

                # no_wrong_plan_after = (
                #    dummy_test_env.number_of_finished_wps
                #    - dummy_test_env.correct_plan_finishes
                # )
                # s, count = dummy_test_env.count_too_far_right_wps(action[-1])

                if True:  # count > 0: # or no_wrong_plan_after > no_wrong_plan_before:
                    print(
                        f"Environment {i} step {step}: Model action invalid, using policy instead"
                    )
                    # print("action: ", action)
                    # print("obs: ", obs[i])

                    # If the action is invalid, use the policy to compute a new action
                    # action = policy.compute_action(obs[i])
                    count_invalid_model_actions += 1

                # conduct step with chosen action
                new_obs, reward, done, trunc, info = env.step(action)

                obs[i] = new_obs
                reward_tracker += reward
                if done or trunc:
                    break  # Exit early if the episode is over.

                # print("number of unfinished exits: ", )

            all_env_in_buffer_waiting_over_time.append(env.in_buffer_waiting)
            all_env_rewards.append(reward_tracker)
            all_env_count_invalid_model_actions.append(count_invalid_model_actions)

        print("Completed hybrid validation test.\n")

        results = {
            "no_cor_both_finishes": np.array(
                [env.correct_both_finishes for env in self.test_envs]
            ),
            "no_cor_order_finishes": np.array(
                [env.correct_order_finishes for env in self.test_envs]
            ),
            "no_cor_plan_finishes": np.array(
                [env.correct_plan_finishes for env in self.test_envs]
            ),
            "no_total_finishes": np.array(
                [env.number_of_finished_wps for env in self.test_envs]
            ),
            "rewards": np.array(all_env_rewards),
            "counts_invalid_model_actions": np.array(
                all_env_count_invalid_model_actions
            ),
        }

        # Add pieces in buffer if new arrival process activated
        if not self.selected_config.get("old_arrival_process"):
            results["in_buffer_waiting_over_time"] = all_env_in_buffer_waiting_over_time

        return self.test_seeds, results

    def test_policy_determinstic_behavior(self, policy):
        self.reset_envs()

        print(
            "Test whether policy works deterministically (always same action for one obs)..."
        )
        random.seed(1)
        np.random.seed(1)

        # Initialize reward tracker for each environment
        num_envs = len(self.test_envs)
        reward_tracker = np.zeros(
            num_envs, dtype=np.float32
        )  # Tracks cumulative rewards

        obs = [env.state for env in self.test_envs]
        compare_action = policy.compute_action(obs)

        all_env_rewards = []

        for i, env in enumerate(self.test_envs):
            print(f"Test environment {i}")
            for _ in range(1000):
                action = policy.compute_action(obs)
                assert np.array_equal(action, compare_action)

        print("Policy works deterministically.\n")


if __name__ == "__main__":

    def main_test():
        vals = DefaultValidator("X1_stochastic")

        states = []
        for val in vals.test_envs:
            states.append(val.state)

        return states

    def test_2():
        # test if seeds are working
        states = main_test()
        for i in range(50):
            states_ = main_test()
            if states == states_:
                states = states_
                print("True in ", i)
            else:
                print("False")
                break

    def test_val():
        checkpoint_filename = "InnerOrder_X1_stochastic_olli_mimi_t_2025-03-14_09-32-21"
        val_test = DefaultValidator("X1_stochastic")

        tot_rewards, mean_reward = val_test.test_rl_model(checkpoint_filename)

        print("mean = ", mean_reward)
        print("all rewards: ", tot_rewards)

    test_val()
