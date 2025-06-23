# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:36:08 2025

@author: mimib

Environment for matrix production, a grid of machines and buffers with workpieces
"""


import gymnasium
from gymnasium import spaces
import numpy as np
import random
import copy
import torch
from itertools import groupby
from tools.tools_reward_shaping import get_reward


def matrix_to_list(matrix):
    transposed = matrix.transpose(1, 0, 2)  # shape: (Y, X, Z)
    reversed_ = transposed[::-1, ::-1, :]  # shape: (Y, X, Z), but reversed in Y and X.
    flattened = reversed_.flatten()

    return flattened.tolist()


def generate_index_map(rows, cols):
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


def still_to_do(plan, done):
    return plan[done:]


def get_remapping(current_order):
    order_sorted = np.sort(current_order)

    remapping = {}
    i = 0
    for element in order_sorted:
        remapping[element] = i
        i += 1

    return remapping


def get_queue_action(action):
    return [action[-1]]


def get_grid_action(action):
    return action[:-1]


def action_decoding_to_numpy(dic):
    # Define the mapping for the letters
    mapping = {"w": 0, "s": 1, "noWP": -1}

    # Build a list of rows in sorted order of the keys
    rows = []
    for i in sorted(dic.keys()):
        letter, (x, y) = dic[i]
        value = mapping.get(letter, -1)  # Default to -1 if letter not found
        rows.append([value, x, y])

    # Convert the list into a NumPy array with shape (33, 3)
    arr = np.array(rows)

    return arr


def environment_action_handling(grid_size):
    action_decoding = {}
    index = 0
    for wait_service in range(2):
        for y in range(grid_size[1] - 1, -1, -1):
            for x in range(grid_size[0] - 1, -1, -1):
                info = "w" if wait_service == 0 else "s"
                action_decoding[index] = (info, (x, y))
                index += 1

    # add queue and finish location to dict
    action_decoding[index] = ("w", (-1, -1))  # in queue
    action_decoding[index + 1] = ("w", (-2, -2))  # finished
    action_decoding[index + 2] = ("noWP", (-3, -3))  # no wp here

    numpy_action_decoding = action_decoding_to_numpy(action_decoding)

    action_redecoding = {value: key for key, value in action_decoding.items()}

    # decode list index to cell:
    action_index_decoding = {}
    index = 0
    for y in range(grid_size[1] - 1, -1, -1):
        for x in range(grid_size[0] - 1, -1, -1):
            action_index_decoding[index] = (x, y)
            index += 1

    action_index_decoding[grid_size[0] * grid_size[1]] = (-1, -1)
    action_index_redecoding = {
        value: key for key, value in action_index_decoding.items()
    }

    return (
        action_decoding,
        numpy_action_decoding,
        action_redecoding,
        action_index_decoding,
        action_index_redecoding,
    )


class Matrix_production(gymnasium.Env):
    def __init__(self, env_config):
        super().__init__()

        # INFO OUTPUT
        self.info_ = ["Initializing environment..."]
        self.info__ = ["Initializing environment..."]

        self.env_name = env_config["env_name"]
        self.verbosity_env = env_config["verbosity_env"]
        self.max_steps = env_config["max_steps"]
        self.grid_size = env_config["grid_size"]
        self.product_groups = env_config["product_groups"]
        self.production_plans = env_config["production_plans"]
        self.padded_production_plans = env_config["padded_production_plans"]
        self.arrival_interval = env_config["arrival_interval"]
        self.arrival_pattern = env_config["arrival_pattern"]
        self.number_of_workpieces = env_config["number_of_workpieces"]
        self.machine_or_buffer = env_config["machine_or_buffer"]
        self.failure_probs = env_config["failure_probs"]
        self.machine_abilities = env_config["machine_abilities"]
        self.reset_strategy = env_config["reset_strategy"]

        self.reward_type = env_config["reward_type"]
        self.reward_params = env_config["reward_params"]

        self.old_arrival_process = env_config.get("old_arrival_process", False)

        # machine_or_buffer -->  # 0 is buffer, 1 is machine, indexing like action/state indexing
        # machine_abilities <-- -1 is buffer --> no ability, indexing like action indexing
        # failure probs <-- -1 is buffer --> no failure, indexing like action/state
        self.currently_working_machines = copy.deepcopy(self.machine_or_buffer)
        self.production_plan_lengths = np.array(
            [len(plan) for plan in self.production_plans]
        )

        self.reset_utilization = self.reset_strategy["utilization"]
        self.reset_ctrl_done = self.reset_strategy["control_done_index"]

        self.num_different_subactions_environment = (
            (self.grid_size[0] * self.grid_size[1]) * 2 + 1 + 1 + 1
        )
        self.num_different_subactions_network = (
            (self.grid_size[0] * self.grid_size[1]) + 1 + 1 + 1
        )

        self.num_cells = self.grid_size[0] * self.grid_size[1]
        self.id_counter = -1  # first arrival sets id to 0
        self.number_of_finished_wps = 0  # summed for one episode
        self.episode_reward = (
            0  # total reward summed over all steps of an episode, not discounted
        )
        self.episode_production_reward = 0  # reward that comes from steps with finished workpieces, other steps neglected, not discounted
        self.episode_step_count = 0  # current step of the episode
        self.correct_order_finishes = 0
        self.correct_plan_finishes = 0
        self.correct_both_finishes = 0

        if self.old_arrival_process:
            self.in_buffer_waiting = "not implemented yet"
        else:
            self.in_buffer_waiting = []

        # extract stage abilities, CAUTION!! from first to last stage
        self.stage_abilities = []
        for y in range(self.grid_size[1], 0, -1):
            all_abilities = self.machine_abilities[
                (y - 1) * self.grid_size[0] : y * self.grid_size[0]
            ]
            unique_abilities = list(set(all_abilities))
            self.stage_abilities.append(unique_abilities)

        self.final_indices_by_pgs = [[len(plan)] for plan in self.production_plans]
        self.number_of_pgs = len(self.product_groups)
        self.longest_prod_plan = np.array(self.final_indices_by_pgs).max()

        # extract longest stay on one machine
        self.longest_time_on_one = max(
            [
                max(len(list(group)) for _, group in groupby(plan))
                for plan in self.production_plans
            ]
        )

        # environment action handling, also includes wait actions!
        (
            self.action_decoding,
            self.numpy_action_decoding,
            self.action_redecoding,
            self.action_index_decoding,
            self.action_index_redecoding,
        ) = environment_action_handling(self.grid_size)

        # STATES as grid + extra cell waiting queue + outgoing order cells
        # gridcells: [0 functioning/ 1 failure, 0 no wp / 1 wp here,  id/inner order, pg, index_donelist, rem_service]
        # cell queue: [0 no wp / 1 wp here, pg, timestemp_intervall, index_next]
        # test: cell queue: [0 no wp / 1 + how many wp here, pg, timestemp_intervall, index_next]
        # outgoing order cells: [id, id, id, ... ]

        grid_attributes = [
            2,
            2,
            self.num_cells,
            self.number_of_pgs,
            self.longest_prod_plan + 1,
            self.longest_time_on_one,  # remaining service is a unused function! currently always 0
        ]
        # num_grid_cells = self.grid_size[0] * self.grid_size[1]
        flattened_grid = grid_attributes * self.num_cells

        # Outgoing Order: [id] per outgoing cell
        # outgoing_attributes = [self.number_of_workpieces + 1]
        # num_outgoing_cells = self.number_of_workpieces
        # flattened_outgoing = outgoing_attributes * num_outgoing_cells

        if self.old_arrival_process:
            look_forward = 2
        else:
            look_forward = 201

        # Queue: [wp_present, pg, timestamp_interval, index_next]
        queue_attributes = [
            look_forward,  # presee how many workpieces are waiting, this has different meanings depending of the action of old_arrival
            self.number_of_pgs,
            self.arrival_interval,
            len(self.arrival_pattern),
        ]
        flattened_queue = queue_attributes  # Assuming a fixed-size queue

        # Combine all into a single MultiDiscrete space
        self.observation_space = spaces.MultiDiscrete(flattened_grid + flattened_queue)

        # Default machine/buffer state
        default_grid_state = np.array([[0, 0, 0, 0, 0, 0]] * self.num_cells)
        # Set buffer cells to `1` at the first position (which represents failure state)
        default_grid_state[:, 0] = np.where(
            self.machine_or_buffer == 0, 1, 0
        )  # Buffers: 1, machines: 0
        default_grid_state = default_grid_state.flatten().tolist()
        self.default_state = default_grid_state + [0, 0, 0, 0]

        # neural network action handling, less actions than in env, leave out wait actions!
        self.action_space = spaces.MultiDiscrete(
            [self.num_different_subactions_network] * (self.num_cells + 1)
        )

        self.downstream_relations = self.create_downstream_relations()

        # dictionary with keys: (pg, index_done) and elements (latest y ebene)
        self.dic_latest_possible_stages = self.latest_possible_stages()

        # Enumerate environment config
        self.info__.extend(
            f"{i}. {key}: {value}"
            for i, (key, value) in enumerate(env_config.items(), start=1)
        )

        # Add additional information
        info_lines = [
            f"\nExtracting stage abilities... {self.stage_abilities}",
            f"Extracting longest time on one machine... {self.longest_time_on_one}",
            "-----------------------------------------------------------------------------------------------------",
            f"Total number of subactions: {self.num_different_subactions_environment}",
            f"Total number of cells (grid + queue): {self.num_cells + 1}",
            f"\nDecoding dict for each sub_action: {self.action_decoding}",
            f"\nDecode action list index to cell coordinate: {self.action_index_decoding}",
            f"Redecode: cell coordinate to action list index: {self.action_index_redecoding}",
            f"\nMapping from index to location coordinate (used for states and actions): {self.action_index_decoding}",
            f"\nDownstream relations, map from cell to possible actions: {self.downstream_relations}",
            "-----------------------------------------------------------------------------------------------------",
        ]

        self.info__.extend(info_lines)

        self.reverse_index_map = generate_index_map(
            self.grid_size[0], self.grid_size[1]
        )
        self.reset()

        # Join list into a single string
        self.info_ = "\n".join(self.info_)
        self.info__ = "\n".join(self.info__)

        if self.verbosity_env == 1:
            print(self.info_)
        if self.verbosity_env == 2:
            print(self.info__)

        self.info_, self.info__ = [], []

    def create_downstream_relations(self):
        # dictionary that maps current cell (indexed like action/state) to allowed network actions by downstream constraint
        downstream_relations = {}
        for i in range(self.num_cells):  # all except queue
            x_i, y_i = self.action_index_decoding[i]
            allowed_coord = []
            for x_new in range(self.grid_size[0]):
                for y_new in range(y_i, self.grid_size[1]):
                    allowed_coord.append((x_new, y_new))

            # map coord to allowed actions
            allowed_actions = []
            for coord in allowed_coord:
                allowed_actions.append(self.action_index_redecoding[coord])
                # allowed_actions.append(
                #    self.action_index_redecoding[coord] + self.num_cells
                # )

            allowed_actions.append(
                self.num_different_subactions_network - 2
            )  # (-2, -2) action, finished workpiece, always allowed
            allowed_actions.append(
                self.num_different_subactions_network - 1
            )  # nowp action, always allowed

            # order tmp and set
            downstream_relations[i] = sorted(allowed_actions)

        # queue cell, all actions are downstream valid
        downstream_relations[self.num_cells] = list(
            range(self.num_different_subactions_network)
        )

        return downstream_relations

    def check_plan_with_remaining_stages(
        self, product_group, done_index, current_stage
    ):
        stage = 0  # start with the first stage
        plan = self.production_plans[product_group]
        remaining_stages = self.stage_abilities[current_stage:]
        to_do = still_to_do(plan, done_index)

        while to_do:
            element = to_do[0]
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
        # dictionary with keys: (pg, index_done) and elements (latest y stage)
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

    def reset(self, seed=None, options=None):
        if self.reset_strategy["start_zero"]:
            self.reset_zero()
        else:
            self.reset_with_utilization(seed)

        # conduct machine failures
        self.machine_failure_vectorized()

        # Generate the initial observation
        obs = np.array(self.state, dtype=np.int64)

        # reset trackers
        self.episode_step_count = 0
        self.number_of_finished_wps = 0
        self.episode_reward = 0
        self.episode_production_reward = 0
        self.correct_order_finishes = 0
        self.correct_plan_finishes = 0
        self.correct_both_finishes = 0

        if not self.old_arrival_process:
            self.in_buffer_waiting = [obs[-4]]

        return obs, {}

    def reset_zero(self):
        self.state = copy.deepcopy(self.default_state)

    def reset_with_utilization(self, seed=None):
        self.state = copy.deepcopy(self.default_state)

        # set random seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        new_state_grid = np.array(
            self.list_to_matrix(self.get_grid_state(self.default_state), 6)
        )
        decide_wp_presence = (
            np.random.rand(*new_state_grid.shape[:-1], 1) < self.reset_utilization
        ).astype(int)

        num_ones = np.sum(decide_wp_presence)
        num_zeros = new_state_grid.shape[0] * new_state_grid.shape[1] - num_ones
        inner_order_list = np.array((range(num_ones)))

        # shuffle order slighlty, if used: no valid state ensurance for order controlled latest stage policy
        # order_splits = np.array_split(inner_order_list, decide_wp_presence.shape[1])
        # sublists = [
        #    sublist[np.random.permutation(len(sublist))] for sublist in order_splits
        # ]
        # inner_order_list = np.concatenate(sublists)

        # Count occurrences of each product group
        pg_count = np.bincount(self.arrival_pattern, minlength=len(self.product_groups))
        pg_shares = pg_count / len(self.arrival_pattern)
        sampled_pgs = np.random.choice(
            self.product_groups, size=len(inner_order_list), p=pg_shares
        )

        max_ind_pg = np.fromiter(map(len, self.production_plans), dtype=int) - 1
        if self.reset_ctrl_done:  # choose random done index with respect to innerorder
            max_ind_sampled_pgs = max_ind_pg[sampled_pgs] + 1
            if max_ind_sampled_pgs.size > 0:
                # chosen_index_done = np.floor(
                #    max_ind_sampled_pgs * (1 - inner_order_list) + 1
                # )

                # set all to finished
                chosen_index_done = max_ind_sampled_pgs
            else:
                chosen_index_done = np.array([])
        else:  # choose totally random
            print("not implemented yet")

        rem_services = np.zeros(num_ones, dtype=int)
        workings = np.zeros(num_ones, dtype=int)
        wps = np.ones(num_ones, dtype=int)

        new_wp_data = np.column_stack(
            [
                workings,
                wps,
                inner_order_list,
                sampled_pgs,
                chosen_index_done,
                rem_services,
            ]
        )
        no_wp = np.tile(np.array([[0, 0, 0, 0, 0, 0]]), (num_zeros, 1))

        decide_wp_presence = np.array(matrix_to_list(decide_wp_presence))

        # Create output array with the correct shape
        final_state_list = np.zeros(
            (len(decide_wp_presence), no_wp.shape[1]), dtype=no_wp.dtype
        )

        # Apply selection using boolean masks
        mask_first = decide_wp_presence == 0
        mask_second = decide_wp_presence == 1

        final_state_list[mask_first] = no_wp
        final_state_list[mask_second] = new_wp_data

        # queue:
        wp_waiting = np.random.choice([0, 1])
        pg_waiting = (
            np.random.choice(self.product_groups, p=pg_shares) if wp_waiting == 1 else 0
        )
        queue = [
            wp_waiting,
            pg_waiting,
            np.random.choice(list(range(self.arrival_interval))),
            np.random.choice(list(range(len(self.arrival_pattern)))),
        ]

        self.set_grid_state(final_state_list.ravel())
        self.set_queue_state(queue)

        self.id_counter = num_ones - 1 + queue[0]

    def get_grid_state(self, state=None):
        if state is None:
            state = self.state

        return state[:-4]

    def set_grid_state(self, grid_as_list):
        if isinstance(grid_as_list, np.ndarray):
            new_state = grid_as_list.tolist()
            self.state[:-4] = new_state

            return (
                grid_as_list.tolist()
            )  # Converts all nested arrays to lists and NumPy scalars to Python scalars

        self.state[:-4] = grid_as_list

    def get_queue_state(self):
        return self.state[-4:]

    def set_queue_state(self, queue_as_list):
        if isinstance(queue_as_list, np.ndarray):
            return (
                queue_as_list.tolist()
            )  # Converts all nested arrays to lists and NumPy scalars to Python scalars
        self.state[-4:] = queue_as_list

    def list_to_matrix(self, lst, values_per_cell=6):

        # Step 1: Reshape into (rows, cols, values_per_cell)
        data_2d = np.array(lst).reshape(-1, values_per_cell)

        # Use indexing to reorder sub-lists into the desired layout
        matrix = data_2d[self.reverse_index_map]

        return matrix

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

    def get_remaining_service_vectorized(self, pg_list, index_current_todo_list):
        num_plans, max_length = self.padded_production_plans.shape
        current_values = self.padded_production_plans[
            pg_list, index_current_todo_list
        ]  # Shape: (num_queries,)
        all_indices = np.arange(max_length)[None, :]  # shape: (1, max_length)
        valid_mask = (
            all_indices >= index_current_todo_list[:, None]
        )  # mask for valid indices: only consider indices >= start_idx.
        eq_mask = (
            self.padded_production_plans[pg_list, :] == current_values[:, None]
        )  # shape: (n, max_length) # create a boolean mask: True if the value equals current_vals.
        mask = np.where(
            valid_mask, eq_mask, True
        )  # indices before the start index, force the mask to True (so they don’t affect argmax).

        # For each row, find the first index (column) where the condition is False.
        # Note: np.argmax returns 0 if no False is found, handle later
        first_false = np.argmax(~mask, axis=1)

        rts = (
            first_false - index_current_todo_list
        )  # remaining service time as the difference between the first False index and the start index

        # Correction: if for a row all valid positions are True (i.e. no False encountered),
        # then np.argmax returns 0, but the correct run-length is (max_length - start_idx).
        all_true = np.all(mask, axis=1)
        rts[all_true] = max_length - index_current_todo_list[all_true]

        return rts

    def machine_failure_vectorized(self):
        state_grid_as_list = self.get_grid_state()
        # Convert machine_or_buffer and failure_probs to NumPy arrays
        machine_mask = self.machine_or_buffer == 1  # Boolean mask for machines
        failure_probs = np.array(self.failure_probs)
        random_values = np.random.rand(
            len(failure_probs)
        )  # random numbers for failure check
        failures = (
            random_values <= failure_probs
        ) & machine_mask  # Machines fail where random values are ≤ failure probability

        # Update the grid state:
        # - Set **failed machines** to `1`
        # - Set **working machines** to `0`
        # - Set **buffers** to `1` (since they are always non-working)
        state_grid_as_list[::6] = np.where(
            machine_mask, failures.astype(int), 1
        ).tolist()

        # Update currently working machines list (1 = working, 0 = failed or buffer)
        self.currently_working_machines = np.where(
            machine_mask, 1 - failures.astype(int), 0
        ).tolist()

        # Update state
        self.set_grid_state(state_grid_as_list)

    def arrival(self):
        # cell queue: [0 no wp / 1 wp here, pg, timestemp_intervall, index_next]
        queue = self.get_queue_state()
        is_empty, current_timestep, index_next = queue[0], queue[2], queue[3]

        new_arrival = (current_timestep == self.arrival_interval - 1) & (is_empty == 0)
        if new_arrival:
            new_index = (index_next + 1) % len(self.arrival_pattern)
            new_pg = self.arrival_pattern[index_next]
            self.set_queue_state([1, new_pg, current_timestep, new_index])

            # Increment ID counter
            self.id_counter += 1

        if (current_timestep == self.arrival_interval - 1) & (is_empty == 1):
            rejected = 1
            # rejection_reward = -2
        else:
            rejected = 0

        return rejected

    def arrival_test(self):
        # cell queue: [0 no wp / 1 wp here, pg, timestemp_intervall, index_next]
        queue = self.get_queue_state()
        count_queue, current_timestep, index_next = queue[0], queue[2], queue[3]

        new_arrival = current_timestep == self.arrival_interval - 1
        if new_arrival:
            new_index = (index_next + 1) % len(self.arrival_pattern)
            new_count = count_queue + 1
            new_pg = self.arrival_pattern[index_next]
            self.set_queue_state([new_count, new_pg, current_timestep, new_index])

            # Increment ID counter
            self.id_counter += 1

        # if (current_timestep == self.arrival_interval - 1) & (is_empty == 1):
        #    rejected = 1
        #    # rejection_reward = -2
        # else:
        #    rejected = 0

        # return rejected

    def calculate_reward(
        self,
        *,
        num_fin_wps,
        inner_order,
        ratio_finished,
        rejected,
        share_wp_too_far_right,
        count_wp_too_far_right,
        num_proc_service,
        current_action,
        obs,
    ):
        # decide which reward, make cases
        assert (
            num_fin_wps <= 1
        ), f"num_fin_wps is {num_fin_wps}, expected <= 1\nstate: {obs}\ncurrent_action: {current_action}"

        reward = None

        if num_fin_wps > 0:
            if inner_order[0] == 0:
                self.correct_order_finishes += 1
                if ratio_finished[0] == 1:
                    self.correct_both_finishes += 1
            if ratio_finished[0] == 1:
                self.correct_plan_finishes += 1

        reward = get_reward(
            num_fin_wps=num_fin_wps,
            inner_order=inner_order,
            ratio_finished=ratio_finished,
            num_proc_service=num_proc_service,
            rejected=rejected,
            share_wp_too_far_right=share_wp_too_far_right,
            count_wp_too_far_right=count_wp_too_far_right,
            reward_type=self.reward_type,
            reward_params=self.reward_params,
            observation=obs,
            actual_action=current_action,
            env_name=self.env_name,
        )

        return reward

    def decode_action_network_to_env(self, action):
        return action + self.num_cells

    def validate_and_adjust_action(self, action, penalty_for_one):
        # does not consider if the action is suitable for the current production step, check in step function!

        # sort out service chosen on buffer or failed machines, change to wait
        c_w_array = np.array(
            self.currently_working_machines
        )  # self.currently_working_machines -->  0 means either buffer or not working, 1 is working machine
        # print("currently working machines: ", c_w_array)
        available_machines = np.where(
            (self.machine_or_buffer == 1) & (c_w_array == 1), 1, 0
        )

        old_action = np.array(action)
        next_loc_if_service = np.where(
            (self.num_cells <= old_action) & (old_action < self.num_cells * 2),
            old_action - self.num_cells,
            -1,
        )
        next_loc_is_buffer = np.where(
            (next_loc_if_service > -1)
            & (np.take(available_machines, next_loc_if_service) == 0),
            1,
            -1,
        )  # set -1 also to 1, because they dont need to be changed
        updated_action = np.where(
            next_loc_is_buffer == -1, old_action, next_loc_if_service
        )

        num_changes = np.sum(old_action != updated_action)  # how many changes made?
        invalid_actions_penalty = penalty_for_one * num_changes

        return invalid_actions_penalty, list(updated_action)

    def count_too_far_right_wps(self, queue_action_number):
        # Count the number of workpieces that are too far right in the grid
        grid_state = np.array(self.list_to_matrix(self.get_grid_state(), 6))
        wp_mask = grid_state[:, :, 1] == 1

        num_wps = np.count_nonzero(wp_mask)
        count_wp_too_far_right = 0
        if num_wps > 0:
            done_index = grid_state[wp_mask, 4]
            pg = grid_state[wp_mask, 3]

            rows, cols = np.where(wp_mask)

            current_stage = cols

            # extract latest stage: dictionary with keys: (pg, index_done) and elements (latest y ebene)
            latest_stage = np.array(
                [
                    self.dic_latest_possible_stages[(pg[i], done_index[i])]
                    for i in range(len(done_index))
                ]
            )

            # Mask for entries where latest_stage != -2, these are already finished wps
            valid_mask = latest_stage != -2

            # Check if the workpiece is too far right, only for valid entries
            count_wp_too_far_right = np.sum(
                current_stage[valid_mask] > latest_stage[valid_mask]
            )

        # queue action, count too far if directly finished
        if queue_action_number == self.num_different_subactions_environment - 1:
            count_wp_too_far_right += 1
            num_wps += 1

        if count_wp_too_far_right > 0:
            share_wp_too_far_right = count_wp_too_far_right / num_wps
        else:
            share_wp_too_far_right = 0

        return share_wp_too_far_right, count_wp_too_far_right

    def step(self, action):
        if torch.is_tensor(action):
            action = action.cpu().numpy()  # Convert tensor action to NumPy array

        # action is assumed to be in network style, convert to env style
        action = self.decode_action_network_to_env(action)

        self.episode_step_count += 1
        outgoing_ids = []
        outgoing_finratios = []
        num_fin_wps = 0
        num_proc_service = 0

        old_obs = copy.deepcopy(self.state)

        # does not consider if the action is suitable for the current production step, check later in step function!
        # Validate and adjust action in bulk
        invalid_actions_penalty, action = self.validate_and_adjust_action(
            action, self.reward_params["invalid_action_pen_for_one"]
        )
        reward = invalid_actions_penalty

        # Get current and default states as matrices
        old_state_grid = np.array(self.list_to_matrix(self.get_grid_state(), 6))
        new_state_grid = np.array(
            self.list_to_matrix(self.get_grid_state(self.default_state), 6)
        )
        action_grid = np.array(self.list_to_matrix(get_grid_action(action), 1))

        # Get action effects
        action_types, positions = zip(
            *self.action_decoding.values()
        )  # Unpack action types and positions
        action_types = np.array(
            action_types, dtype="<U5"
        )  # Fixed-length string array (5 chars max)
        positions = np.array(
            positions, dtype=int
        )  # (N,2) array storing (x, y) as integers
        action_effects_types = action_types[action_grid]
        action_effects_positions = positions[action_grid]
        finished_mask = (
            action_grid == (self.num_different_subactions_environment - 2)
        ).squeeze(axis=2)
        outgoings = old_state_grid[finished_mask, 2]  # Extract finished workpieces IDs

        # Compute rewards for finished workpieces (vectorized)
        if outgoings.size > 0:
            pg = old_state_grid[finished_mask, 3]
            index_done = old_state_grid[finished_mask, 4]
            plan_lengths = np.vectorize(lambda p: len(self.production_plans[p]))(pg)
            ratio_finished = index_done / plan_lengths

            outgoing_ids.extend(outgoings)
            outgoing_finratios.extend(ratio_finished)
            num_fin_wps += outgoings.size
            self.number_of_finished_wps += outgoings.size

        # **Vectorized Position Updates**
        move_mask = (
            action_grid != self.num_different_subactions_environment - 1
        ).squeeze(axis=2) & ~finished_mask
        x_new, y_new = np.array(
            action_effects_positions[move_mask]
        ).T  # Extract movement locations directly from action_effects, Already decoded, just extract
        new_state_grid[x_new, y_new, :] = old_state_grid[
            move_mask, :
        ]  # Update state for moved workpieces (vectorized)

        # Vectorized Service Processing
        service_mask = move_mask & (action_effects_types == "s").squeeze(axis=2)
        x_new_service, y_new_service = np.array(
            action_effects_positions[service_mask]
        ).T
        pg_service = old_state_grid[service_mask, 3]
        index_plan = old_state_grid[service_mask, 4]
        needed_ability = self.padded_production_plans[pg_service, index_plan]
        keys = np.array(list(self.action_index_redecoding.keys()))  # Shape: (3,2)
        values = np.array(list(self.action_index_redecoding.values()))  # Shape: (3,)

        if x_new_service.size > 0 and y_new_service.size > 0:
            matching_indices = (
                (
                    keys[:, None]
                    == np.column_stack(
                        (x_new_service.squeeze(), y_new_service.squeeze())
                    )
                )
                .all(-1)
                .argmax(0)
            )
        else:
            matching_indices = np.array([], dtype=int)  # Directly return empty array

        machine_indices = values[matching_indices]
        machine_abilities_array = np.array(self.machine_abilities)
        machine_abilities = machine_abilities_array[
            machine_indices
        ]  # check if ability requirement match machine ability, what is left out in validate action
        matching_mask = needed_ability == machine_abilities

        # iterate over matching mask
        for i in range(len(matching_mask)):
            if needed_ability[i] == -2:
                assert matching_mask[i] != True, print(
                    f"machine abilities: {machine_abilities[i]}, "
                    f"needed_ability: {needed_ability[i]}, "
                    f"index_plan: {index_plan[i]}, "
                    f"pg_service: {pg_service[i]}"
                )

        if np.sum(matching_mask) > 0:  # for serviced workpieces
            num_proc_service += np.sum(matching_mask)  # for reward calculation

            # update remaining service time
            # new_state_grid[
            #    x_new_service[matching_mask.reshape(1, -1)],
            #    y_new_service[matching_mask.reshape(1, -1)],
            #    5,
            # ] = self.get_remaining_service_vectorized(
            #    pg_service[matching_mask], index_plan[matching_mask]
            # )

            new_state_grid[
                x_new_service[matching_mask.reshape(1, -1)],
                y_new_service[matching_mask.reshape(1, -1)],
                5,
            ] = 0

            # update done production index
            new_state_grid[
                x_new_service[matching_mask.reshape(1, -1)],
                y_new_service[matching_mask.reshape(1, -1)],
                4,
            ] = (index_plan[matching_mask] + 1)

        # Process queue action
        action_queue = get_queue_action(action)
        a_number = action_queue[0]
        old_queue_list = self.get_queue_state()

        if a_number < self.num_different_subactions_environment - 3:  # move in grid
            what, (x_, y_) = self.action_decoding[a_number]  # Unpack action decoding
            pg = old_queue_list[1]  # Production group of the workpiece
            needed_ability = self.padded_production_plans[
                pg, 0
            ]  # Get required ability, Always step 0 for new workpieces
            machine_index = self.action_index_redecoding[(x_, y_)]
            machine_ability = self.machine_abilities[
                machine_index
            ]  # Get machine ability at target location
            can_service = (what == "s") & (
                needed_ability == machine_ability
            )  # Check if service is possible (action is service and machine matches requirement)
            # **Vectorized Grid Update**, depending on can_service
            new_state_grid[x_, y_, :] = [
                old_state_grid[x_, y_, 0],
                1,
                self.id_counter,
                pg,
                int(can_service),
                0,
                # self.get_remaining_service(pg, 0) if can_service else 0,
            ]

            if can_service:
                num_proc_service += 1  # for reward calculation

            if self.old_arrival_process:
                old_queue_list[:2] = [
                    0,
                    0,
                ]  # Reset workpiece presence and production group
            else:
                old_queue_list[0] -= 1  # Decrease workpiece presence in queue
                if old_queue_list[0] == 0:  # If no workpiece left in queue
                    old_queue_list[1] = 0
                else:
                    index_next = old_queue_list[3]
                    old_queue_list[1] = self.arrival_pattern[
                        index_next
                    ]  # Update production group
                    new_index = (index_next + 1) % len(self.arrival_pattern)
                    old_queue_list[3] = new_index

        if (
            a_number == self.num_different_subactions_environment - 2
        ):  # move to finished workpieces
            # finish workpiece from queue

            # calculate reward
            # order is always the last from all elements in the grid
            wp_mask = (
                new_state_grid[..., 1] == 1
            )  # Select elements where workpiece exists
            wp_ids = new_state_grid[wp_mask, 2]  # Workpiece ID field
            latest_id = np.max(wp_ids) if wp_ids.size > 0 else -1
            outgoing_queue_id = latest_id + 1
            ratio_finished = 0

            self.number_of_finished_wps += 1

            outgoing_ids.append(outgoing_queue_id)
            outgoing_finratios.append(ratio_finished)
            num_fin_wps += 1

            if self.old_arrival_process:
                old_queue_list[:2] = [
                    0,
                    0,
                ]  # Reset workpiece presence and production group
            else:
                # update queue to next wp
                old_queue_list[0] -= 1  # Decrease workpiece presence in queue
                if old_queue_list[0] == 0:  # If no workpiece left in queue
                    old_queue_list[1] = 0
                else:
                    index_next = old_queue_list[3]
                    old_queue_list[1] = self.arrival_pattern[
                        index_next
                    ]  # Update production group
                    new_index = (index_next + 1) % len(self.arrival_pattern)
                    old_queue_list[3] = new_index

        if (
            a_number == self.num_different_subactions_environment - 3
        ):  # stay in queue action
            rejected = 1
        else:
            rejected = 0

        # Remap the inner orders in the grid
        wp_mask = new_state_grid[..., 1] == 1  # Select elements where workpiece exists
        wp_ids = new_state_grid[wp_mask, 2]  # Workpiece ID field
        current_inner_order = (
            wp_ids.tolist()
        )  # Update internal tracking inner order list # recreate inneroder from grid because i dont trust the constant  tracking of the inner order list

        remapping = get_remapping(current_inner_order)
        remap_keys = np.array(list(remapping.keys()))  # Old IDs
        remap_values = np.array(list(remapping.values()))  # New IDs

        if remapping:
            order = np.argsort(remap_keys)
            sorted_keys = remap_keys[order]
            sorted_values = remap_values[order]
            positions = np.searchsorted(sorted_keys, wp_ids)
            new_ids = sorted_values[positions]

            # set new ids
            new_state_grid[wp_mask, 2] = new_ids

        # update state grid and queue
        self.set_grid_state(matrix_to_list(new_state_grid))
        self.set_queue_state(old_queue_list)

        # report interstate grid and queue before arrival failure
        interstate_grid_before_arrival_failure = copy.deepcopy(new_state_grid)
        # copy machine failures from original state
        interstate_grid_before_arrival_failure[:, :, 0] = old_state_grid[:, :, 0]
        interstate_queue_before_arrival_failure = copy.deepcopy(old_queue_list)

        # Apply failures and arrivals, depends on the setting in config file
        self.machine_failure_vectorized()
        if self.old_arrival_process:
            rejected = self.arrival()
        else:
            self.arrival_test()

        # update time step interval
        new_queue_list = self.get_queue_state()
        new_queue_list[2] = (new_queue_list[2] + 1) % self.arrival_interval
        self.set_queue_state(new_queue_list)

        # self.episode_reward += reward

        # Check termination conditions
        terminated, truncated = False, self.episode_step_count >= self.max_steps

        share_wp_too_far_right, count_wp_too_far_right = self.count_too_far_right_wps(
            a_number
        )

        reward = self.calculate_reward(
            num_fin_wps=num_fin_wps,
            inner_order=outgoing_ids,
            ratio_finished=outgoing_finratios,
            rejected=rejected,
            share_wp_too_far_right=share_wp_too_far_right,
            count_wp_too_far_right=count_wp_too_far_right,
            num_proc_service=num_proc_service,
            current_action=action,
            obs=old_obs,
        )

        new_obs = np.array(self.state, dtype=np.int64)

        if not self.old_arrival_process:
            self.in_buffer_waiting.append(new_obs[-4])

        return (
            new_obs,
            float(reward),
            terminated,
            truncated,
            {
                "episode_step_count": self.episode_step_count,
                "interstate_grid_before_arrival_failure": interstate_grid_before_arrival_failure,
                "interstate_queue_before_arrival_failure": interstate_queue_before_arrival_failure,
            },
        )


if __name__ == "__main__":
    from train.config import ENV_CONFIGS

    def validate_action_test():
        selected_config = ENV_CONFIGS["default"]
        test_env = Matrix_production(selected_config)
        action = [0, 3, 1, 8]
        pen, new_action = test_env.validate_and_adjust_action(action, 0.1)
        print(f"old action: {action} \nnew action: {new_action} \npenalty: {pen}\n")

        pen, new_action = test_env.validate_and_adjust_action(action, 0.1)
        print(f"old action: {action} \nnew action: {new_action} \npenalty: {pen}\n")

        test_env.step(new_action)

    def machine_failure_test():
        selected_config = ENV_CONFIGS["X1_stochastic"]
        test_env = Matrix_production(selected_config)
        test_env.machine_failure_vectorized()
        print("new state after failure: ", test_env.state)
        print("reset state: ", test_env.reset())

    def matrix_to_list_test():
        selected_config = ENV_CONFIGS["default"]
        test_env = Matrix_production(selected_config)

        matrix_test = np.array([[[1, 3], [2, 9], [3, 6]], [[4, 2], [5, 6], [6, 4]]])
        print("test matrix: ", matrix_test)
        print("to list function: ", matrix_to_list(matrix_test))

    def count_too_far_right_test():
        selected_config = ENV_CONFIGS["simple_3by3"]
        test_env = Matrix_production(selected_config)

        print(test_env.check_plan_with_remaining_stages(2, 2, 1))
        print(test_env.dic_latest_possible_stages)

    def reset_random_test():
        selected_config = ENV_CONFIGS["most_simple_2by3"]
        test_env = Matrix_production(selected_config)

        obs, info = test_env.reset(seed=1)
        print(obs)

        print(test_env.downstream_relations)

    count_too_far_right_test()
