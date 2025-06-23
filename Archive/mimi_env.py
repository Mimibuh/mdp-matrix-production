# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:31:15 2025

@author: mimib
"""

import gymnasium
from gymnasium import spaces
import numpy as np
import random
import copy
import torch
from itertools import groupby


class Desp(gymnasium.Env):
    def __init__(self, env_config=None):
        super().__init__()

        # INFO OUTPUT
        self.info_ = ["Initializing environment..."]
        self.info__ = ["Initializing environment..."]

        default_config = {
            "verbosity_env": 0,
            "verbosity_net": 0,
            "max_steps": 1000,
            "grid_size": (1, 3),
            "machine_or_buffer": [
                1,
                1,
                0,
            ],  # 0 is buffer, 1 is machine, indexing like action/state indexing
            "failure_probs": [
                0.1,
                0.2,
                -1,
            ],  # -1 is buffer --> no failure, indexing like action/state
            "machine_abilities": [
                1,
                0,
                -1,
            ],  # -1 is buffer --> no ability, indexing like action indexing
            "product_groups": [0, 1],
            "production_plans": [[0, 1], [0, 0]],  # for pg 0  # for pg 1
            "arrival_interval": 2,  # every second step a wp arrives, if queue not blocked
            "number_of_workpieces": 10,  # how many to be produced
            "arrival_pattern": [0, 1],
        }

        if env_config == None:
            env_config = default_config

        self.verbosity_env = env_config["verbosity_env"]  # 0 -No, 1 +env, 2 ++env
        self.verbosity_net = env_config["verbosity_net"]  # 0 -No, 1 +net, 2 ++net
        self.max_steps = env_config["max_steps"]
        self.grid_size = env_config["grid_size"]
        self.machine_or_buffer = env_config[
            "machine_or_buffer"
        ]  # 0 is buffer, 1 is machine, indexing like action/state indexing
        self.failure_probs = env_config[
            "failure_probs"
        ]  # -1 is buffer --> no failure, indexing like action/state
        self.machine_abilities = env_config[
            "machine_abilities"
        ]  # -1 is buffer --> no ability, indexing like action indexing
        self.product_groups = env_config["product_groups"]
        self.production_plans = env_config["production_plans"]
        self.arrival_interval = env_config["arrival_interval"]
        self.arrival_pattern = env_config["arrival_pattern"]
        self.number_of_workpieces = env_config["number_of_workpieces"]

        self.total_amount_action_components = (
            (self.grid_size[0] * self.grid_size[1]) * 2 + 1 + 1 + 1
        )
        self.number_of_cells = self.grid_size[0] * self.grid_size[1]
        self.id_counter = -1  # first arrival sets id to 0

        # extract stage abilities, CAUTION!! from first to last stage
        self.stage_abilities = []
        for y in range(self.grid_size[1], 0, -1):
            all_abilities = self.machine_abilities[
                (y - 1) * self.grid_size[0] : y * self.grid_size[0]
            ]
            # print("y: ", y, "   alll ab: ", all_abilities)
            unique_abilities = list(set(all_abilities))
            self.stage_abilities.append(unique_abilities)

        # extract final indices by pgs
        self.final_indices_by_pgs = []
        for element in self.production_plans:
            self.final_indices_by_pgs.append([len(element)])

        # extract number of product groups
        self.number_of_pgs = len(self.product_groups)

        # extract longest production plan
        self.longest_prod_plan = np.array(self.final_indices_by_pgs).max()

        # extract longest stay on one machine
        tmp = []
        for plan in self.production_plans:
            tmp.append(max(len(list(group)) for _, group in groupby(plan)))

        self.longest_time_on_one = max(tmp)

        self.action_decoding = {}
        # decode dict
        index = 0
        for wait_service in range(2):
            for y in range(self.grid_size[1] - 1, -1, -1):
                for x in range(self.grid_size[0] - 1, -1, -1):
                    info = "w" if wait_service == 0 else "s"
                    self.action_decoding[index] = (info, (x, y))
                    index += 1

        # add queue and finish location
        self.action_decoding[index] = ("w", (-1, -1))  # in queue
        index += 1
        self.action_decoding[index] = ("w", (-2, -2))  # finished
        index += 1
        self.action_decoding[index] = "noWP"  # no wp here

        self.action_redecoding = {
            value: key for key, value in self.action_decoding.items()
        }

        # decode list index to cell:
        self.action_index_decoding = {}
        index = 0
        for y in range(self.grid_size[1] - 1, -1, -1):
            for x in range(self.grid_size[0] - 1, -1, -1):
                self.action_index_decoding[index] = (x, y)
                index += 1

        self.action_index_decoding[self.number_of_cells] = (-1, -1)

        # STATES as grid + extra cell waiting queue + outgoing order cells
        # gridcells: [0 functioning/ 1 failure, 0 no wp / 1 wp here, id, pg, index_donelist, rem_service]
        # cell queue: [0 no wp / 1 wp here, pg, timestemp_intervall, index_next]
        # outgoing order cells: [id, id, id, ... ]

        grid_attributes = [
            2,
            2,
            self.number_of_workpieces,
            self.number_of_pgs,
            self.longest_prod_plan + 1,
            self.longest_time_on_one,
        ]
        # num_grid_cells = self.grid_size[0] * self.grid_size[1]
        flattened_grid = grid_attributes * self.number_of_cells

        # Outgoing Order: [id] per outgoing cell
        outgoing_attributes = [self.number_of_workpieces + 1]
        num_outgoing_cells = self.number_of_workpieces
        flattened_outgoing = outgoing_attributes * num_outgoing_cells

        # Queue: [wp_present, pg, timestamp_interval, index_next]
        queue_attributes = [
            2,
            self.number_of_pgs,
            self.arrival_interval,
            len(self.arrival_pattern),
        ]
        flattened_queue = queue_attributes  # Assuming a fixed-size queue

        # Combine all into a single MultiDiscrete space
        self.observation_space = spaces.MultiDiscrete(
            flattened_grid + flattened_queue + flattened_outgoing
        )

        self.default_state = (
            [0, 0, 0, 0, 0, 0] * self.number_of_cells
            + [0, 0, 0, 0]
            + [self.number_of_workpieces] * self.number_of_workpieces
        )

        self.action_space = spaces.MultiDiscrete(
            [self.total_amount_action_components] * (self.number_of_cells + 1)
        )

        # dictionary with keys: (pg, index_done) and elements (latest y ebene)
        self.dic_latest_possible_stages = self.latest_possible_stages()
        # another dic that converts this to (pg, index_done, current_stage) and elements (valid actions)
        self.dic_valid_actions_by_remaining_stages = (
            self.valid_actions_by_remaining_stages()
        )

        # Enumerate environment config
        self.info__.extend(
            f"{i}. {key}: {value}"
            for i, (key, value) in enumerate(env_config.items(), start=1)
        )

        # Add additional information
        self.info__.append(f"\nExtracting stage abilities... {self.stage_abilities}")
        self.info__.append(
            f"Extracting longest time on one machine... {self.longest_time_on_one}"
        )
        self.info__.append(
            "-----------------------------------------------------------------------------------------------------"
        )
        self.info__.append(
            f"Total number of subactions: {self.total_amount_action_components}"
        )
        self.info__.append(
            f"Total number of cells (grid + queue): {self.number_of_cells + 1}"
        )
        self.info__.append(
            f"\nDecoding dict for each sub_action: {self.action_decoding}"
        )
        self.info__.append(
            f"\nMapping from index to location coordinate (used for states and actions): {self.action_index_decoding}"
        )
        self.info__.append(
            "-----------------------------------------------------------------------------------------------------"
        )

        self.reset()
        self.reverse_index_map = self.generate_index_map(
            self.grid_size[0], self.grid_size[1]
        )

        # Join list into a single string
        self.info_ = "\n".join(self.info_)
        self.info__ = "\n".join(self.info__)

        if self.verbosity_env == 1:
            print(self.info_)
        if self.verbosity_env == 2:
            print(self.info__)

    def generate_index_map(self, rows, cols):
        """
        Generate an index map of shape (rows, cols) filling from bottom-right
        to top-left (i.e., each column bottom-to-top, then move left).
        """
        index_map = np.zeros((rows, cols), dtype=int)
        index = 0
        # Go from rightmost column down to 0
        for c in reversed(range(cols)):
            # Fill this column from bottom row up to row=0
            for r in reversed(range(rows)):
                index_map[r, c] = index
                index += 1
        return index_map

    def still_to_do(self, plan, done):
        return plan[done:]

    def check_plan_with_remaining_stages(
        self, product_group, done_index, current_stage
    ):
        # print(product_group, done_index, current_stage)
        stage = 0  # start with the first stage

        plan = self.production_plans[product_group]

        remaining_stages = self.stage_abilities[current_stage:]
        # print(remaining_stages)
        to_do = self.still_to_do(plan, done_index)
        # print("todo", to_do)

        while to_do:
            element = to_do[0]
            # print("\nelement:", element)
            # print("stage: ", stage)
            if (
                element not in remaining_stages[stage]
                and stage == len(remaining_stages) - 1
            ):
                return False
            elif element not in remaining_stages[stage]:
                stage += 1
            else:
                to_do.pop(0)

        return True

    def latest_possible_stages(self):

        # i need a dictionary with keys: (pg, index_done) and elements (latest y stage)

        dic = {}

        for pg in self.product_groups:
            for index_done in range(len(self.production_plans[pg])):
                y = 0
                while y < self.grid_size[1] and self.check_plan_with_remaining_stages(
                    pg, index_done, y
                ):
                    y += 1

                dic[(pg, index_done)] = y - 1
            dic[(pg, len(self.production_plans[pg]))] = (
                -2
            )  # if done, then going ou is possible --> latest stage = -2

        return dic

    def valid_actions_by_remaining_stages(self):
        # i need a dic that converts the latest_pos_stages dic to key:(pg, index_done) and elements:(valid actions)
        # caution: downstream constraint is ignored here because it is masked in the custom model anyways
        dic = {}

        for pg in self.product_groups:
            for index_done in range(len(self.production_plans[pg]) + 1):
                latest_stage = self.dic_latest_possible_stages[(pg, index_done)]
                if latest_stage == -2:  # all actions possible
                    action_indices = list(range(self.action_space.nvec[0]))
                else:  # all actions until latest stage are allowed
                    action_indices = []
                    for key in list(self.action_decoding.keys())[
                        :-2
                    ]:  # exclude finish action here!
                        # print(self.action_decoding[key])
                        if self.action_decoding[key][1][1] <= latest_stage:
                            action_indices.append(key)
                    action_indices.append(
                        self.action_space.nvec[0] - 1
                    )  # reinclude last action "noWP" (is always possible)
                    # action_indices.append(self.action_space.nvec[0]-2) # second last action "stay in queue" is always possible
                dic[(pg, index_done)] = action_indices

        return dic

    def reset(self, seed=None, options=None):
        print("Resetting the environment...")

        # Generate the initial observation
        self.state = copy.deepcopy(self.default_state)
        self.step_count = 0
        self.id_counter = -1

        return self.state, {}

    def get_grid_state(self, state=None):
        if not state:
            state = self.state

        return state[: self.number_of_cells * 6]

    def set_grid_state(self, grid_as_list, state=None):
        if not state:
            state = self.state

        state[: self.number_of_cells * 6] = grid_as_list

        return

    def get_queue_state(self, state=None):
        if not state:
            state = self.state

        return state[self.number_of_cells * 6 : self.number_of_cells * 6 + 4]

    def set_queue_state(self, queue_as_list, state=None):
        if not state:
            state = self.state

        self.state[self.number_of_cells * 6 : self.number_of_cells * 6 + 4] = (
            queue_as_list
        )

        return

    def get_outgoing_state(self, state=None):
        if not state:
            state = self.state

        return state[self.number_of_cells * 6 + 4 :]

    def set_outgoing_state(self, outgoing_as_list, state=None):
        if not state:
            state = self.state

        self.state[self.number_of_cells * 6 + 4 :] = outgoing_as_list

        return

    def get_grid_action(self, action):
        return action[:-1]

    def get_queue_action(self, action):
        return [action[-1]]

    def list_to_matrix(self, lst, values_per_cell=6):
        """
        Converts a flat list stored in bottom-right zigzag order into a 3D matrix.

        :param lst: The input flat list (each cell contains multiple values).
        :param rows: Number of rows in the target matrix.
        :param cols: Number of columns in the target matrix.
        :param values_per_cell: Number of values per matrix cell.
        :return: A 3D numpy array with shape (rows, cols, values_per_cell).
        """

        # Step 1: Reshape into (rows, cols, values_per_cell)
        data_2d = np.array(lst).reshape(-1, values_per_cell)

        # Use fancy indexing to reorder sub-lists into the desired 2×2 layout
        matrix = data_2d[self.reverse_index_map]

        return matrix

    def matrix_to_list(self, matrix):
        lst = []
        for y in range(matrix.shape[1] - 1, -1, -1):
            for x in range(matrix.shape[0] - 1, -1, -1):
                lst.append(matrix[x, y])

        return list(np.array(lst).flatten())

    def get_remaining_service(self, pg, index_current_todo):
        # print("calculate remaining service time... pg: ", pg, "  index_current_todo: ", index_current_todo)
        rt = 0
        current_ability = self.production_plans[pg][index_current_todo]
        for ability in self.production_plans[pg][index_current_todo + 1 :]:
            if ability == current_ability:
                rt += 1
            else:
                break

        return rt

    def machine_failure(self):
        self.info__.append("Check machine failure...")

        # Convert the grid from list to matrix form
        state_grid_as_list = self.get_grid_state()

        for i in range(len(self.machine_or_buffer)):
            if self.machine_or_buffer[i] == 1:  # is machine
                # Get the failure probability for this machine
                p = self.failure_probs[i]
                # Check for failure based on the probability
                if random.random() <= p:
                    self.info__.append(
                        f"Machine number {i} (indexed like action/state) fails..."
                    )
                    state_grid_as_list[i * 6] = 1  # Update the grid state
                else:
                    state_grid_as_list[i * 6] = (
                        0  # reset previously failed machines to waiting
                    )
                state_grid_as_list[i * 6]

        # update state
        self.set_grid_state(state_grid_as_list)

        return

    def arrival(self):
        self.info__.append("Check arrival...")

        if (
            self.id_counter == self.number_of_workpieces - 1
        ):  # all workpieces arrived already
            self.info__.append(
                "No, new arrival. All workpieces are already in the system."
            )
            return

        # cell queue: [0 no wp / 1 wp here, pg, timestemp_intervall, index_next]
        queue = self.get_queue_state()
        is_empty = queue[0]
        current_timestep = queue[2]
        index_next = queue[3]
        if current_timestep == self.arrival_interval - 1:  # new arrival
            if is_empty == 0:  # arrival only if queue is empty
                new_index = (
                    0 if index_next == len(self.arrival_pattern) - 1 else index_next + 1
                )
                new_pg = self.arrival_pattern[index_next]
                # update queue
                self.set_queue_state([1, new_pg, current_timestep, new_index])
                self.id_counter += 1
                self.info__.append(
                    f"New Workpiece arrives... pg:{new_pg}, id:{self.id_counter}"
                )

    def goal_reached(self):
        self.info__.append("Checking goal...")

        # check if all workpieces are finished
        should_ids = [i for i in range(self.number_of_workpieces)]

        outgoing = self.get_outgoing_state()

        for sid in should_ids:
            if sid not in outgoing:
                return False
        return True

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.tolist()  # Convert tensor to list
        elif isinstance(action, np.ndarray):
            action = action.tolist()  # Convert numpy array to list

        self.info_, self.info__ = ["\nStep..."], ["\nStep..."]
        self.info__.append(f"In state: {self.state}")
        self.info__.append(f"Do action: {action}")

        self.step_count += 1

        # Simulate state transition
        # state lists
        old_grid_list = self.get_grid_state()
        old_queue_list = self.get_queue_state()
        old_outgoing_list = self.get_outgoing_state()
        new_grid_list = self.get_grid_state(self.default_state)

        # state matrix
        old_state_grid_matrix = self.list_to_matrix(old_grid_list, 6)
        new_state_grid_matrix = self.list_to_matrix(new_grid_list, 6)

        # action lists
        action_grid_list = self.get_grid_action(action)
        action_queue = self.get_queue_action(action)

        # action matrix
        action_grid_matrix = self.list_to_matrix(action_grid_list, 1)
        reward = 0

        for x in range(action_grid_matrix.shape[0]):
            for y in range(action_grid_matrix.shape[1]):
                # status gridcells: [0 functioning/ 1 failure, 0 no wp / 1 wp here, id, pg, index_donelist, rem_service]
                # action gridcells: [0 no wp / 1 wp here, 0 waiting/ 1 service, x loc, y loc]
                a_number = action_grid_matrix[x, y][0]
                # print(a_number)
                # print("hier: ", self.action_decoding)

                if a_number == self.total_amount_action_components - 1:  # no wp action
                    continue
                elif a_number == self.total_amount_action_components - 2:  # finished wp
                    # print("\nWorkpiece with id:" + str(old_state_grid_matrix[x, y][2]) + " is finished and out!\n")
                    reward += 1
                    # print("out: ", old_outgoing_list)
                    index = old_outgoing_list.index(
                        self.number_of_workpieces
                    )  # Find the index of the first occurrence of 10
                    old_outgoing_list[index] = old_state_grid_matrix[x, y][
                        2
                    ]  # Replace the value at that index
                    self.set_outgoing_state(old_outgoing_list)
                    continue
                elif (
                    a_number != self.total_amount_action_components - 3
                ):  # position change
                    # print("hier: ", self.action_decoding[a_number])
                    what, location = self.action_decoding[a_number]
                    x_, y_ = location

                    if what == "w":  # wait at next location, no service
                        new_state_grid_matrix[x_, y_] = [
                            old_state_grid_matrix[x_, y_, 0],
                            1,
                            old_state_grid_matrix[x, y, 2],
                            old_state_grid_matrix[x, y, 3],
                            old_state_grid_matrix[x, y, 4],
                            old_state_grid_matrix[x, y, 5],
                        ]
                    else:  # service at next location
                        pg = old_state_grid_matrix[x, y, 3]
                        index_prod_plan = old_state_grid_matrix[x, y, 4]
                        # print("pg: ", pg, "   index_prod_plan: ", index_prod_plan)
                        rt = self.get_remaining_service(pg, index_prod_plan)
                        # rt = 2
                        new_state_grid_matrix[x_, y_] = [
                            0,
                            1,
                            old_state_grid_matrix[x, y, 2],
                            old_state_grid_matrix[x, y, 3],
                            old_state_grid_matrix[x, y, 4] + 1,
                            rt,
                        ]

        a_number = action_queue[0]

        if (
            a_number < self.total_amount_action_components - 3
        ):  # position change, other actions not possible or do nothing
            # print("a_number: ", a_number)
            # print("action dec dic: ", self.action_decoding)
            what, location = self.action_decoding[a_number]
            x_, y_ = location

            pg = old_queue_list[1]

            if what == "w":  # wait at next location
                new_state_grid_matrix[x_, y_] = [
                    old_state_grid_matrix[x_, y_, 0],  # machine status of new location
                    1,  # workpiece is there
                    self.id_counter,  # set new id
                    pg,  # pg of wp in queue
                    0,  # index_done
                    0,  # remaining_service
                ]
            else:  # service at next location
                index_prod_plan = 0
                rt = self.get_remaining_service(pg, index_prod_plan)
                new_state_grid_matrix[x_, y_] = [
                    old_state_grid_matrix[x_, y_, 0],
                    1,
                    self.id_counter,
                    pg,
                    1,
                    rt,
                ]
            # reset queue:
            old_queue_list[0] = 0  # no wp here
            old_queue_list[1] = 0  # reset pg

        # new state set grid
        self.set_grid_state(self.matrix_to_list(new_state_grid_matrix))
        self.set_queue_state(old_queue_list)

        # apply failure and arrival
        self.machine_failure()
        self.arrival()

        # update timestep interval
        old_queue_list = self.get_queue_state()
        if old_queue_list[2] == self.arrival_interval - 1:
            old_queue_list[2] = 0
        else:
            old_queue_list[2] += 1
        # set new queue list
        self.set_queue_state(old_queue_list)

        if self.goal_reached():
            self.info_.append("Goal reached. Terminate episode...")
            self.info__.append("Goal reached. Terminate episode...")
            terminated = True
            truncated = False
        elif self.step_count >= self.max_steps:
            self.info_.append("Max steps reached. Truncate episode...")
            self.info__.append("Max steps reached. Truncate episode...")
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        if reward == 0:
            reward = -1

        if reward > 1:
            self.info_.append(f"REWARD PROBLEM, r={reward}")
            self.info__.append(f"REWARD PROBLEM, r={reward}")

        # Join list into a single string
        self.info_ = "\n".join(self.info_)
        self.info__ = "\n".join(self.info__)
        if self.verbosity_env == 1:
            print(self.info_)
        if self.verbosity_env == 2:
            print(self.info__)

        self.info_, self.info__ = [], []

        return (
            self.state,
            reward,
            terminated,
            truncated,
            {"step_count": self.step_count},
        )


# Default Environment
default_env = Desp()


# Test System X1
X1_config = {
    "verbosity_env": 2,
    "verbosity_net": 0,
    "max_steps": 1000,
    "grid_size": (3, 5),
    "machine_or_buffer": [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
    ],  # 0 is buffer, 1 is machine, indexing like action/state indexing
    "failure_probs": [
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
    ],  # -1 is buffer --> no failure, indexing like action/state
    "machine_abilities": [
        -1,
        -1,
        -1,
        2,
        1,
        0,
        2,
        1,
        0,
        2,
        1,
        0,
        -1,
        -1,
        -1,
    ],  # -1 is buffer --> no ability, indexing like action indexing
    "product_groups": [0, 1, 2, 3],
    "production_plans": [
        [0, 0, 1],  # for pg 0
        [0, 1, 0, 2],  # for pg 1
        [0, 1, 1, 0, 2],  # for pg 2
        [2, 2, 2, 1],
    ],  # for pg 3
    "arrival_interval": 1,  # every second step a wp arrives, if queue not blocked
    "number_of_workpieces": 12,  # how many to be produced
    "arrival_pattern": [0, 1, 2, 3],
}

if __file__ == "__main__":
    print("\nTest System X1")
    X1_env = Desp(X1_config)

    # simulate random action
    new_state, reward, term, trun, info_dic = X1_env.step(X1_env.action_space.sample())

    print("\nnext state: ", new_state)
