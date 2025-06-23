# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:43:56 2025

@author: mimib

Config files, each representing a matrix production system configuration. used in training, validation, and testing of RL models.

"""

import numpy as np

ENV_CONFIGS = dict(
    default=dict(
        env_name="default",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(1, 3),
        machine_or_buffer=np.array([1, 1, 0]),
        failure_probs=[0.1, 0.2, -1],
        machine_abilities=[1, 0, -1],
        product_groups=[0, 1],
        production_plans=[[0, 1], [0, 0]],
        padded_production_plans=np.array(
            [[0, 1, -2], [0, 0, -2]]
        ),  # always -2 at the end for the finished index
        arrival_interval=2,
        number_of_workpieces=10,
        arrival_pattern=[0, 1],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=0.9,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
    most_simple_1by3=dict(
        env_name="most_simple_1by3",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(1, 3),
        machine_or_buffer=np.array([1, 1, 1]),
        failure_probs=[0, 0, 0],
        machine_abilities=[2, 1, 0],
        product_groups=[0],
        production_plans=[[0, 1, 2]],
        padded_production_plans=np.array(
            [[0, 1, 2, -2]]
        ),  # always -2 at the end for the finished index
        arrival_interval=2,
        number_of_workpieces=10,
        arrival_pattern=[0],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=0.9,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
    most_simple_3by1=dict(
        env_name="most_simple_3by1",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 1),
        machine_or_buffer=np.array([1, 1, 1]),
        failure_probs=[0, 0, 0],
        machine_abilities=[2, 1, 0],
        product_groups=[0],
        production_plans=[[0, 1, 2]],
        padded_production_plans=np.array(
            [[0, 1, 2, -2]]
        ),  # always -2 at the end for the finished index
        arrival_interval=2,
        number_of_workpieces=10,
        arrival_pattern=[0],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=0.9,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
    most_simple_2by3=dict(
        env_name="most_simple_2by3",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(2, 3),
        machine_or_buffer=np.array([1, 1, 1, 1, 1, 1]),
        failure_probs=[0, 0, 0, 0, 0, 0],
        machine_abilities=[5, 2, 4, 1, 3, 0],
        product_groups=[0, 1],
        production_plans=[[0, 1, 2], [3, 4, 5]],
        padded_production_plans=np.array(
            [[0, 1, 2, -2], [3, 4, 5, -2]]
        ),  # always -2 at the end for the finished index
        arrival_interval=2,
        number_of_workpieces=10,
        arrival_pattern=[0, 1],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=0.9,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
    simple_2by3=dict(
        env_name="simple_2by3",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(2, 3),
        machine_or_buffer=np.array([1, 1, 1, 1, 0, 0]),
        failure_probs=[0, 0, 0, 0, -1, -1],
        machine_abilities=[0, 1, 0, 0, -1, -1],
        product_groups=[0, 1],
        production_plans=[[0, 1], [0, 0]],
        padded_production_plans=np.array(
            [[0, 1, -2], [0, 0, -2]]
        ),  # always -2 at the end for the finished index
        arrival_interval=2,
        number_of_workpieces=10,
        arrival_pattern=[0, 1],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=0.9,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
    X1_stochastic=dict(
        env_name="X1_stochastic",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=100,
        grid_size=(3, 5),
        machine_or_buffer=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
        failure_probs=[
            -1,
            -1,
            -1,
            0.3,
            0.3,
            0.2,
            0.3,
            0.3,
            0.2,
            0.3,
            0.3,
            0.2,
            -1,
            -1,
            -1,
        ],
        machine_abilities=[-1, -1, -1, 2, 1, 0, 2, 1, 0, 2, 1, 0, -1, -1, -1],
        product_groups=[0, 1, 2, 3],
        production_plans=[
            [0, 0, 1],  # for pg 0
            [0, 1, 0, 2],  # for pg 1
            [0, 1, 1, 0, 2],  # for pg 2
            [2, 2, 2, 1],
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 1, -2, -2, -2],  # for pg 0
                [0, 1, 0, 2, -2, -2],  # for pg 1
                [0, 1, 1, 0, 2, -2],  # for pg 2
                [2, 2, 2, 1, -2, -2],
            ]
        ),
        arrival_interval=1,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2, 3],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=0.9,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
    simple_3by3=dict(
        env_name="simple_3by3",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0, 0.4, 0.1, 0.1, 0.2, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=1,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0_1",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
    simple_3by3_oldarrival=dict(
        env_name="simple_3by3_oldarrival",
        old_arrival_process=True,
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0, 0.4, 0.1, 0.1, 0.2, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=1,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_simple_linear_ANTIPLATEAU",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01=dict(
        env_name="simple_3by3_01",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0, 0.4, 0.1, 0.1, 0.2, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_02",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_02=dict(
        env_name="simple_3by3_01_02",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.2, 0.4, 0.1, 0.4, 0.4, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=1,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_11",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_02_old_arrival=dict(
        env_name="simple_3by3_01_02_old_arrival",
        old_arrival_process=True,
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.2, 0.4, 0.1, 0.4, 0.4, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=1,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_03=dict(
        env_name="simple_3by3_01_03",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.2, 0.4, 0.1, 0.4, 0.4, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_02",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_04=dict(
        env_name="simple_3by3_01_04",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.4, 0.4, 0.3, 0.4, 0.4, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_02",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_05=dict(
        env_name="simple_3by3_01_05",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.4, 0.4, 0.3, 0.4, 0.4, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2, 2, 1],  # for pg 0
            [0, 3, 3],  # for pg 1
            [0, 1, 1, 1, 2, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, 2, 1, -2],  # for pg 0
                [0, 3, 3, -2, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, 2, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_06",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_06=dict(
        env_name="simple_3by3_01_06",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 0, 1, 0, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.4, -1, 0.3, -1, 0.4, 0.3],
        machine_abilities=[-1, -1, 3, 1, -1, 2, -1, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_06",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_07=dict(
        env_name="simple_3by3_01_07",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 0, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.4, -1, 0.3, 0.1, 0.6, 0.3],
        machine_abilities=[-1, -1, 3, 1, -1, 2, 2, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [3, 3],  # for pg 1
            [1, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [3, 3, -2, -2, -2, -2],  # for pg 1
                [1, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_simple_linear_02",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_07_01=dict(
        env_name="simple_3by3_01_07_01",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 0, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.4, -1, 0.3, 0.1, 0.3, 0.3],
        machine_abilities=[-1, -1, 3, 1, -1, 2, 2, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_simple_linear_03",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_07_02=dict(
        env_name="simple_3by3_01_07_02",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 0, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.4, 0.4, 0.3, 0.4, 0.4, 0.3],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_simple_linear_01",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_07_03=dict(
        env_name="simple_3by3_01_07_03",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 0, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_simple_linear_01",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_01_08=dict(
        env_name="simple_3by3_01_08",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 0, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.7, 0.4, 0.5, 0.3, 0.5, 0.5, 0.5],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [3, 3],  # for pg 1
            [1, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [3, 3, -2, -2, -2, -2],  # for pg 1
                [1, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=2,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="all_constraints_01_02_02_03",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
            current_train_iteration=0,
        ),
    ),
    simple_3by3_nofailure=dict(
        env_name="simple_3by3_nofailure",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0, 0, 0, 0, 0, 0, 0],
        machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [0, 3],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=1,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0_1",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=1,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
    simple_3by3_bad_latest_stage=dict(
        env_name="simple_3by3_bad_latest_stage",
        verbosity_env=0,
        verbosity_net=0,
        max_steps=200,
        grid_size=(3, 3),
        machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        failure_probs=[-1, -1, 0.6, 0, 0.4, 0.1, 0.1, 0.2, 0.3],
        machine_abilities=[-1, 1, 3, 1, 2, 0, 1, 0, 3],
        product_groups=[0, 1, 2],
        production_plans=[
            [0, 0, 2, 1, 2],  # for pg 0
            [3, 3, 1],  # for pg 1
            [0, 1, 1, 1],  # for pg 2
        ],
        padded_production_plans=np.array(
            [
                [0, 0, 2, 1, 2, -2],  # for pg 0
                [0, 3, -2, -2, -2, -2],  # for pg 1
                [0, 1, 1, 1, -2, -2],  # for pg 2
            ]
        ),
        arrival_interval=1,
        number_of_workpieces=12,
        arrival_pattern=[0, 1, 2],
        reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
        reward_type="simple0_maxcorplan_benchpolicy",
        reward_params=dict(
            finished_wps_reward_base=1.1,
            beta=0.9,
            bonus=2,
            no_finished_wps_penalty=0,
            invalid_action_pen_for_one=0,
        ),
    ),
)

# DEFAULT_ENV = "medium"  # Change this to switch environments
