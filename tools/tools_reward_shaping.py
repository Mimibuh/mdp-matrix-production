import numpy as np

from train.config import ENV_CONFIGS

"""
All rewards tested, set the name of the reward in the config file under "reward_type" to use it.
"""


def get_reward(
    *,
    num_fin_wps,
    inner_order,
    ratio_finished,
    num_proc_service,
    rejected,
    share_wp_too_far_right,
    count_wp_too_far_right,
    reward_type,
    reward_params,
    observation,
    actual_action,
    env_name,
):

    match reward_type:
        case "simple0":  # simple
            if num_fin_wps > 0:
                if (
                    inner_order[0] == 0 and ratio_finished[0] == 1
                ):  # wp finished correctly
                    reward = reward_params["bonus"]
                else:
                    reward = -0.1  # workpiece finished, but not correctly
            else:
                reward = 0  # no workpiece finished

        case "simple0_1":  # simple
            if num_fin_wps > 0:
                if (
                    inner_order[0] == 0 and ratio_finished[0] == 1
                ):  # wp finished correctly
                    reward = reward_params["bonus"]
                else:
                    reward = -0.1  # workpiece finished, but not correctly
            else:
                reward = -0.01  # no workpiece finished

            # extra processing reward:
            reward += 0.01 * num_proc_service

        case "simple0_2":  # simple without processing reward
            if num_fin_wps > 0:
                if (
                    inner_order[0] == 0 and ratio_finished[0] == 1
                ):  # wp finished correctly
                    reward = reward_params["bonus"]
                else:
                    reward = -0.1  # workpiece finished, but not correctly
            else:
                reward = -0.01  # no workpiece finished

        case "simple0_3":  # simple
            if num_fin_wps > 0:
                if (
                    inner_order[0] == 0 and ratio_finished[0] == 1
                ):  # wp finished correctly
                    reward = reward_params["bonus"]
                else:
                    reward = -0.1  # workpiece finished, but not correctly
            else:
                reward = -0.1  # no workpiece finished

            # extra processing reward:
            reward += 0.01 * num_proc_service

        case "simple0_4":  # simple
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward += reward_params["bonus"]
                else:
                    reward += -1  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1

        case "simple0_5":  # simple
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = reward_params["bonus"]
                else:
                    reward = -0.11  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1

        case "simple0_6":  # simple
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -1  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -1.05

        case "simple0_7":  # simple
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.5  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.55

        case "simple0_8":  # simple
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.55  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "simple0_9":  # simple
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -1.05  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -1

        case "simple0_10":  # simple
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -2.05  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -2

        case "simple0_maxcorplan":  # simple
            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward = reward_params["bonus"]
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.5
                else:
                    reward = -0.1  # workpiece finished, but not correct plan
            else:
                reward = -0.01  # no workpiece finished

            # extra processing reward:
            reward += 0.01 * num_proc_service

        case "simple0_maxcorplan_02":  # simple
            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward = 0.8
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.2
                else:
                    reward = -0.2  # workpiece finished, but not correct plan
            else:
                reward = -0.05  # no workpiece finished

        case "simple1":  # simple, separately reward/pen inner_order and ratio_finished
            if num_fin_wps > 0:
                if inner_order[0] == 0:  # wp finished in correct order
                    reward = 1
                else:
                    reward = -1

                if ratio_finished[0] == 1:  # finished wp has all production steps
                    reward += 5
                else:
                    reward += -2

            else:
                reward = 0  # no workpiece finished

        case "simple2":  # simple, separately reward/pen inner_order and ratio_finished
            if num_fin_wps > 0:
                if inner_order[0] == 0:  # wp finished in correct order
                    reward = 0.2
                else:
                    reward = -0.2

                if ratio_finished[0] == 1:  # finished wp has all production steps
                    reward += 1
                else:
                    reward += -0.4

            else:
                reward = 0  # no workpiece finished

            # extra processing reward:
            reward += 0.02 * num_proc_service

        case "simple3":  # simple, separately reward/pen inner_order and ratio_finished
            if num_fin_wps > 0:
                if inner_order[0] == 0:  # wp finished in correct order
                    reward = 0.2
                else:
                    reward = -0.2

                if ratio_finished[0] == 1:  # finished wp has all production steps
                    reward += 1
                else:
                    reward += -0.4

            else:
                reward = -0.1  # no workpiece finished

            # extra processing reward:
            reward += 0.02 * num_proc_service

        case "simple4":  # simple, separately reward/pen inner_order and ratio_finished
            if num_fin_wps > 0:
                if inner_order[0] == 0:  # wp finished in correct order
                    reward = 0.2
                else:
                    reward = -0.2

                if ratio_finished[0] == 1:  # finished wp has all production steps
                    reward += 1
                else:
                    reward += -0.4

            else:
                reward = -0.5  # no workpiece finished

            # extra processing reward:
            reward += 0.02 * num_proc_service

        case "simple5":  # simple, separately reward/pen inner_order and ratio_finished
            if num_fin_wps > 0:
                if inner_order[0] == 0:  # wp finished in correct order
                    reward = 0.1
                else:
                    reward = -0.1

                if ratio_finished[0] == 1:  # finished wp has all production steps
                    reward += 1
                else:
                    reward += -0.4

            else:
                reward = -0.5  # no workpiece finished

            # extra processing reward:
            reward += 0.02 * num_proc_service

        case "simple6":  # simple, separately reward/pen inner_order and ratio_finished
            if num_fin_wps > 0:
                if inner_order[0] == 0:  # wp finished in correct order
                    reward = 0.2
                else:
                    reward = 0

                if ratio_finished[0] == 1:  # finished wp has all production steps
                    reward += 1
                else:
                    reward += -(1 - ratio_finished[0])

            else:
                reward = -0.1  # no workpiece finished

            if num_fin_wps > 0:
                if inner_order[0] == 0 and ratio_finished[0] == 1:
                    reward += reward_params["bonus"]

            # extra processing reward:
            reward += 0.1 * num_proc_service

        case "simple7":  # simple, separately reward/pen inner_order and ratio_finished
            reward = 0

            if num_fin_wps > 0:
                if inner_order[0] == 0:  # wp finished in correct order
                    reward += 0.5

                if ratio_finished[0] == 1:  # finished wp has all production steps
                    reward += 1

                if ratio_finished != 1:
                    reward += -10

        case "exponential0":
            if num_fin_wps > 0:
                add = 0
                if (
                    inner_order[0] == 0 and ratio_finished[0] == 1
                ):  # wp finished correctly
                    add = reward_params["bonus"]

                reward = (
                    reward_params["finished_wps_reward_base"] ** (-inner_order[0])
                    - (1 - ratio_finished[0]) ** reward_params["beta"]
                    + add
                )

                if reward < 0:
                    reward = 0
            else:
                reward = 0

        case "simple0_maxcorplan_benchpolicy":  # simple

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward = reward_params["bonus"]
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.5
                else:
                    reward = -0.1  # workpiece finished, but not correct plan
            else:
                reward = -0.01  # no workpiece finished

            # extra processing reward:
            reward += 0.01 * num_proc_service

            if reward_params["beta"] > 0:
                from test.DefaultPolicyMaker import DefaultPolicyMaker

                # calculate action suggested by benchmark policy
                pol_mk = DefaultPolicyMaker("latest_stage", env_name)
                own_policy = pol_mk.policy
                policy_suggested_action = own_policy.compute_action(observation)

                heuristic_bonus_share = compare_actions(
                    actual_action, policy_suggested_action
                )

                reward += (
                    heuristic_bonus_share * reward_params["beta"]
                )  # reward share of aligned actions with benchmark policy

        case "simple0_maxcorplan_benchpolicy_02":  # simple

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward = reward_params["bonus"]
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.5
                else:
                    reward = -0.1  # workpiece finished, but not correct plan
            else:
                reward = -0.01  # no workpiece finished

            # extra processing reward:
            reward += 0.01 * num_proc_service

            if reward_params["beta"] > 0:
                from test.DefaultPolicyMaker import DefaultPolicyMaker

                # calculate action suggested by benchmark policy
                pol_mk = DefaultPolicyMaker("latest_stage", env_name)
                own_policy = pol_mk.policy
                policy_suggested_action = own_policy.compute_action(observation)

                heuristic_bonus_share = compare_actions(
                    actual_action, policy_suggested_action
                )

                if (
                    heuristic_bonus_share == 1
                ):  # reward if exactly the same as benchmark policy
                    reward += reward_params["beta"]

        case "benchpolicy_replication":  # simple

            if reward_params["beta"] > 0:
                from test.DefaultPolicyMaker import DefaultPolicyMaker

                # calculate action suggested by benchmark policy
                pol_mk = DefaultPolicyMaker("latest_stage", env_name)
                own_policy = pol_mk.policy
                policy_suggested_action = own_policy.compute_action(observation)

                heuristic_bonus_share = compare_actions(
                    actual_action, policy_suggested_action
                )

                if heuristic_bonus_share == 1:
                    reward = 1
                else:
                    reward = 0

        case "bench_replication_02":  # simple

            if reward_params["beta"] > 0:
                from test.DefaultPolicyMaker import DefaultPolicyMaker

                # calculate action suggested by benchmark policy
                pol_mk = DefaultPolicyMaker("latest_stage", env_name)
                own_policy = pol_mk.policy
                policy_suggested_action = own_policy.compute_action(observation)

                heuristic_bonus_share = compare_actions(
                    actual_action, policy_suggested_action
                )

                reward = heuristic_bonus_share

        case (
            "pen_too_far_right_01"  # works very bad, use "pen_too_far_right_02" instead
        ):  # penalize all wps in the system that are too far right, so they cannot be finished anymore
            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward = 0.8
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.2
                else:
                    reward = -0.2  # workpiece finished, but not correct plan
            else:
                reward = 0  # no workpiece finished

            # rejection reward
            if rejected > 0:
                reward += -0.05

            # penalize too far moved pieces
            reward += share_wp_too_far_right * (-0.75)

            # extra processing reward:
            reward += 0.02 * num_proc_service

        case (
            "pen_too_far_right_02"
        ):  # penalize all wps in the system that are too far right, so they cannot be finished anymore
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward += 0.8
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.2

            # rejection reward
            if rejected > 0:
                reward += -0.1

            # penalize too far moved pieces
            x, y = ENV_CONFIGS[env_name]["grid_size"]
            reward += count_wp_too_far_right * (-0.6) / (x * y)

            # extra processing reward:
            reward += num_proc_service * 0.4 / (x * y)

        case (
            "pen_too_far_right_03"
        ):  # penalize all wps in the system that are too far right, so they cannot be finished anymore
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward += 0.8
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.2

            # rejection reward
            if rejected > 0:
                reward += -0.1

            # penalize too far moved pieces
            x, y = ENV_CONFIGS[env_name]["grid_size"]
            reward += count_wp_too_far_right * (-0.7) / (x * y)

            # extra processing reward:
            reward += num_proc_service * 0.3 / (x * y)

        case (
            "pen_too_far_right_04"
        ):  # penalize all wps in the system that are too far right, so they cannot be finished anymore
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward += 0.8
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.2

            # rejection reward
            if rejected > 0:
                reward += -0.1

            # penalize too far moved pieces
            x, y = ENV_CONFIGS[env_name]["grid_size"]
            reward += count_wp_too_far_right * (-0.8) / (x * y)

            # extra processing reward:
            reward += num_proc_service * 0.2 / (x * y)

        case (
            "pen_too_far_right_05"
        ):  # penalize all wps in the system that are too far right, so they cannot be finished anymore
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward += 0.8
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.2

            # rejection reward
            if rejected > 0:
                reward += -0.1

            # penalize too far moved pieces
            x, y = ENV_CONFIGS[env_name]["grid_size"]
            reward += count_wp_too_far_right * (-0.8) / (x * y)

            # extra processing reward:
            reward += num_proc_service * 0.4 / (x * y)

        case (
            "andi_01"
        ):  # penalize all wps in the system that are too far right, so they cannot be finished anymore
            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished plan
                    reward += 0.8
                    if inner_order[0] == 0:  # bonus if order also correct
                        reward += 0.2
                else:
                    reward = -0.8  # workpiece finished, but not correct plan

            # rejection reward
            if rejected > 0:
                reward += -0.8

            # penalize too far moved pieces
            x, y = ENV_CONFIGS[env_name]["grid_size"]
            # reward += count_wp_too_far_right * (-0.8) / (x * y)

            # extra processing reward:
            reward += num_proc_service * 0.4 / (x * y)

        case "rising_punish_01":
            custom_rise_iterations = ENV_CONFIGS[env_name]["reward_params"][
                "custom_rise_iterations"
            ]
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]
            start_value = 0
            end_value = 4

            frac = min(max(current_train_iteration / custom_rise_iterations, 0.0), 1.0)
            weight = start_value + frac * (end_value - start_value)

            # print(f"current weight {weight}")

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1

        case "rising_punish_03":
            custom_rise_iterations = ENV_CONFIGS[env_name]["reward_params"][
                "custom_rise_iterations"
            ]
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]
            start_value = 0
            end_value = 4

            frac = min(max(current_train_iteration / custom_rise_iterations, 0.0), 1.0)
            weight = start_value + frac * (end_value - start_value)

            # print(f"current weight {weight}")

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1 - weight / 2

        case "rising_punish_04":
            custom_rise_iterations = ENV_CONFIGS[env_name]["reward_params"][
                "custom_rise_iterations"
            ]
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]
            start_value = 0
            end_value = 8

            frac = min(max(current_train_iteration / custom_rise_iterations, 0.0), 1.0)
            weight = start_value + frac * (end_value - start_value)

            # print(f"current weight {weight}")

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1 - weight / 2

        case "rising_punish_05":
            custom_rise_iterations = ENV_CONFIGS[env_name]["reward_params"][
                "custom_rise_iterations"
            ]
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]
            start_value = 0
            end_value = 16

            frac = min(max(current_train_iteration / custom_rise_iterations, 0.0), 1.0)
            weight = start_value + frac * (end_value - start_value)

            # print(f"current weight {weight}")

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1 - weight / 2

        case "rising_punish_02":
            custom_rise_iterations = ENV_CONFIGS[env_name]["reward_params"][
                "custom_rise_iterations"
            ]
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]
            start_value = 0
            end_value = 4

            rise_since = 100

            if current_train_iteration > rise_since:
                frac = min(
                    max(
                        (current_train_iteration - rise_since) / custom_rise_iterations,
                        0.0,
                    ),
                    1.0,
                )
                weight = start_value + frac * (end_value - start_value)
            else:
                weight = 0

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1

        case "rising_punish_06":
            custom_rise_iterations = ENV_CONFIGS[env_name]["reward_params"][
                "custom_rise_iterations"
            ]
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]
            start_value = 0
            end_value = 8

            frac = min(max(current_train_iteration / custom_rise_iterations, 0.0), 1.0)
            weight = start_value + frac * (end_value - start_value)

            # print(f"current weight {weight}")

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    reward = 1
                    if inner_order[0] == 0:
                        reward += 0.5
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1 - weight / 2

        case "rising_punish_06_02":
            custom_rise_iterations = ENV_CONFIGS[env_name]["reward_params"][
                "custom_rise_iterations"
            ]
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]
            start_value = 0
            end_value = 8

            frac = min(max(current_train_iteration / custom_rise_iterations, 0.0), 1.0)
            weight = start_value + frac * (end_value - start_value)

            # print(f"current weight {weight}")

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    reward = 1
                    if inner_order[0] == 0:
                        reward += 1
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1 - weight / 2

        case "rising_punish_07":
            custom_rise_iterations = ENV_CONFIGS[env_name]["reward_params"][
                "custom_rise_iterations"
            ]
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]
            start_value = 0
            end_value = 12

            frac = min(max(current_train_iteration / custom_rise_iterations, 0.0), 1.0)
            weight = start_value + frac * (end_value - start_value)

            # print(f"current weight {weight}")

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    reward = 1
                    if inner_order[0] == 0:
                        reward += 0.5
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1 - weight / 2

        case "depot_punish_01":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            depot_1 = 100
            depot_value_1 = 5
            depot_2 = 200
            depot_value_2 = 10

            if current_train_iteration > depot_1:
                weight = depot_value_1
                if current_train_iteration > depot_2:
                    weight = depot_value_2
            else:
                weight = 0

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1 - weight / 2

        case "depot_punish_02":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            depot_1 = 100
            depot_value_1 = 10
            depot_2 = 200
            depot_value_2 = 20
            depo_3 = 300
            depot_value_3 = 40

            if current_train_iteration > depot_1:
                weight = depot_value_1
                if current_train_iteration > depot_2:
                    weight = depot_value_2
                    if current_train_iteration > depo_3:
                        weight = depot_value_3
            else:
                weight = 0

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    if inner_order[0] == 0:
                        reward = 1
                else:
                    reward = -0.11 - weight  # workpiece finished, but not correctly

            # rejection reward
            if rejected > 0:
                reward += -0.1 - weight / 2

        case "depot_punish_03":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            depot_1 = 100
            depot_value_1 = 5

            if current_train_iteration > depot_1:
                weight = depot_value_1
            else:
                weight = 0

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    reward += 1

                if inner_order[0] == 0:
                    reward += 0.5

                if ratio_finished[0] != 1:
                    reward += -1 - weight

            # rejection reward
            if rejected > 0:
                reward += -0.9

        case "depot_punish_04":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            depot_1 = 200
            depot_value_1 = 8
            depot_value_2 = 3

            if current_train_iteration > depot_1:
                weight_plan = depot_value_1
                weight_rej = depot_value_2
            else:
                weight_plan = 0
                weight_rej = 0

            reward = 0

            if num_fin_wps > 0:
                if ratio_finished[0] == 1:  # wp finished correctly
                    reward += 1
                    if inner_order[0] == 0:
                        reward += 0.5

                if ratio_finished[0] != 1:
                    reward += -1 - weight_plan

            # rejection reward
            if rejected > 0:
                reward += -0.9 - weight_rej

        case "all_constraints_01":
            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2

                if ratio_finished[0] != 1:
                    reward += -0.8

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 120:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_simple_linear":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 90:
                weight = 3
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_simple_linear_01":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 90:
                weight = 4
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_simple_linear_PLATEAU":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 90:
                weight = 4
            else:
                weight = 4

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_simple_linear_ANTIPLATEAU":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 90:
                weight = 1
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_simple_linear_02":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 90:
                weight = 1
            else:
                weight = 0.5

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_simple_linear_03":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 90:
                weight = 1
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_plateau":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 120:
                weight = 2
            else:
                weight = 2

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_01":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 150:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_02":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 180:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_02_01":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 220:
                weight = 1
            else:
                weight = 0.5

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_02_02":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 220:
                weight = 2
            else:
                weight = 0.5

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_02_03":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 220:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_03":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 180:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.25

        case "all_constraints_01_02_04":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 180:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -1

        case "all_constraints_01_02_05":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 180:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.85

        case "all_constraints_01_02_06":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 250:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_07":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 400:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_02_08":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 250:
                weight = 2
            if current_train_iteration > 500:
                weight = 3
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -1

        case "all_constraints_01_02_09":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 250:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -1

        case "all_constraints_01_02_10":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 250:
                weight = 2
            if current_train_iteration > 500:
                weight = 2.5
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -1

        case "all_constraints_01_02_11":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 180:
                weight = 3
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -1

        case "all_constraints_01_03":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 0:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.2 * weight

                if ratio_finished[0] != 1:
                    reward += -0.8 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_01_04":
            current_train_iteration = ENV_CONFIGS[env_name]["reward_params"][
                "current_train_iteration"
            ]

            if current_train_iteration > 0:
                weight = 2
            else:
                weight = 1

            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.25 * weight

                if ratio_finished[0] != 1:
                    reward += -1 * weight

            # rejection reward
            if rejected > 0:
                reward += -0.6125

        case "all_constraints_02":
            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.4

                if ratio_finished[0] != 1:
                    reward += -1.6

            # rejection reward
            if rejected > 0:
                reward += -0.5

        case "all_constraints_03":
            reward = 0

            if num_fin_wps > 0:
                if (
                    ratio_finished[0] == 1 and inner_order[0] == 0
                ):  # wp finished correctly
                    reward = 0.9

                if inner_order[0] != 0:
                    reward = -0.3

                if ratio_finished[0] != 1:
                    reward += -1.2

            # rejection reward
            if rejected > 0:
                reward += -0.75

    return reward


def compare_actions(action1, action2):
    # Create a mask that is True where both arrays do NOT have the ignore_value.
    action1 = np.array(action1)
    ignore_value = action1.size * 2
    valid_mask = (action1 != ignore_value) & (action2 != ignore_value)

    # Count the number of valid positions
    valid_count = np.sum(valid_mask)

    # If there are no valid positions, return np.nan
    if valid_count == 0:  # all actions are no wp, no other choice for action!
        return 1

    # Create a mask for positions where the arrays are equal
    equal_mask = (action1 == action2) & valid_mask

    # Calculate the share as the ratio of equal positions to valid positions
    share = np.sum(equal_mask) / valid_count

    return share
