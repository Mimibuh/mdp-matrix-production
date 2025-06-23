import numpy as np

from train.config import ENV_CONFIGS
from train.environment import Matrix_production

"""
Different benchmark (rule-based) policies for the matrix production environment.
"""


class DefaultPolicyMaker:
    def __init__(self, policy_type, environment_name):
        self.policy_type = policy_type
        self.environment_name = environment_name

        selected_config = ENV_CONFIGS[environment_name]
        self.env = Matrix_production(selected_config)

        self.no_wp_action = self.env.num_different_subactions_network - 1
        self.finished_action = self.env.num_different_subactions_network - 2
        self.stay_in_queue_action = self.env.num_different_subactions_network - 3

        self.policy = self.create_policy()

    def create_policy(self):
        match self.policy_type:
            case "silly_random":
                return SillyRandomPolicy(
                    dummy_env=self.env,
                    no_wp_action=self.no_wp_action,
                    finished_action=self.finished_action,
                    stay_in_queue_action=self.stay_in_queue_action,
                )
            case "advanced_random":
                return AdvancedRandomPolicy(
                    dummy_env=self.env,
                    no_wp_action=self.no_wp_action,
                    finished_action=self.finished_action,
                    stay_in_queue_action=self.stay_in_queue_action,
                )
            case "latest_stage":
                return LatestStagePolicy(
                    dummy_env=self.env,
                    no_wp_action=self.no_wp_action,
                    finished_action=self.finished_action,
                    stay_in_queue_action=self.stay_in_queue_action,
                )
            case "order_latest_stage":
                return OrderControlledLatestStage(
                    dummy_env=self.env,
                    no_wp_action=self.no_wp_action,
                    finished_action=self.finished_action,
                    stay_in_queue_action=self.stay_in_queue_action,
                )
            case _:
                raise ValueError(f"Policy type {self.policy_type} not recognized")


class LatestStagePolicy:
    #
    def __init__(
        self, *, dummy_env, no_wp_action, finished_action, stay_in_queue_action
    ):
        self.env = dummy_env
        self.no_wp_action = no_wp_action
        self.finished_action = finished_action
        self.stay_in_queue_action = stay_in_queue_action

    def compute_action(self, obs):
        obs_grid = self.env.list_to_matrix(self.env.get_grid_state(obs), 6)

        action = []
        cell_index = 0
        for y in range(self.env.grid_size[1] - 1, -1, -1):
            for x in range(self.env.grid_size[0] - 1, -1, -1):
                if obs_grid[x, y, 1] == 1:
                    stay_here_action = self.env.action_index_redecoding[(x, y)]

                    choose_from = self.cal_choose_from(cell_index, action)

                    pg = obs_grid[x, y, 3]
                    done_index = obs_grid[x, y, 4]
                    needed_ab = self.env.padded_production_plans[pg, done_index]

                    if needed_ab < 0:  # workpiece is finished
                        if self.finished_action in choose_from:
                            action.append(self.finished_action)
                        else:
                            action.append(
                                self.choose_latest_matching_machine(
                                    choose_from,
                                    needed_ab,
                                    pg,
                                    done_index,
                                    obs_grid,
                                    cell_index,
                                )
                            )

                    else:
                        action.append(
                            self.choose_latest_matching_machine(
                                choose_from,
                                needed_ab,
                                pg,
                                done_index,
                                obs_grid,
                                cell_index,
                            )
                        )
                else:
                    action.append(self.no_wp_action)

                cell_index += 1

        # queue action:
        if obs[-4] >= 1:  # workpiece in queue
            stay_here_action = self.stay_in_queue_action

            choose_from = self.cal_choose_from(self.env.num_cells, action)

            pg = obs[-3]
            done_index = 0
            needed_ab = self.env.padded_production_plans[pg, done_index]

            if needed_ab < 0:  # workpiece is finished
                if self.finished_action in choose_from:
                    action.append(self.finished_action)
                else:
                    action.append(stay_here_action)
            else:
                action.append(
                    self.choose_latest_matching_machine(
                        choose_from, needed_ab, pg, done_index, obs_grid, cell_index
                    )
                )
        else:
            action.append(self.no_wp_action)

        # sanity check for duplicates in action list
        seen = set()
        has_duplicates = False

        for item in action:
            if item == self.no_wp_action:
                continue
            if item in seen:
                has_duplicates = True
                break
            seen.add(item)

        if has_duplicates:
            print("\nDuplicates found in action list:")
            print(action)

        return np.array(action, dtype=np.int64)

    def cal_choose_from(self, cell_index, previous_actions):
        choose_from = self.env.downstream_relations[cell_index]

        # remove actions according to previously chosen actions
        to_remove = set()
        for pa in previous_actions:
            to_remove.add(pa)

        choose_from = [x for x in choose_from if x not in to_remove]

        return choose_from

    def choose_latest_matching_machine(
        self, choose_from, needed_ab, pg, done_index, obs_grid, cell_index
    ):
        latest_pos_stage = self.env.dic_latest_possible_stages[pg, done_index]

        choose_from_stages = [x for x in choose_from if x < self.env.num_cells]

        pos_actions = []
        for a in choose_from_stages:
            if a < self.env.num_different_subactions_network - 3:
                pos_actions.append(
                    (
                        self.env.action_index_decoding[a][0],
                        self.env.action_index_decoding[a][1],
                    )
                )  # y locations for possible choose actions

        distances = [latest_pos_stage - a_y[1] for a_y in pos_actions]

        # Zip the lists together
        assert len(distances) == len(choose_from_stages)
        combined = list(zip(distances, choose_from_stages, pos_actions))
        # Sort by distances
        combined.sort()  # By default, sort() will sort by the first element of each tuple

        if (
            combined == [] and cell_index == self.env.num_cells
        ):  # workpiece in queue, all locations occupied
            return self.stay_in_queue_action

        # Unzip back into two lists
        distances_sorted, choose_from_sorted_by_distance, pos_actions_sorted = zip(
            *combined
        )

        # try to find matching machine
        for i, a in enumerate(choose_from_sorted_by_distance):
            if distances_sorted[i] >= 0:
                machine_ab = self.env.machine_abilities[
                    self.env.action_index_redecoding[pos_actions_sorted[i]]
                ]
                if (
                    machine_ab == needed_ab
                    and obs_grid[pos_actions_sorted[i][0], pos_actions_sorted[i][1], 0]
                    == 0
                ):
                    return a

        # if no matching machine was found, just move as farthest right as possible without completion violation
        for i, a in enumerate(choose_from_sorted_by_distance):
            if distances_sorted[i] >= 0:
                return a

        if needed_ab < 0:  # workpiece is finished
            return np.random.choice(choose_from)

        if cell_index == self.env.num_cells:  # workpiece in queue
            # check if there is a machine available
            return self.stay_in_queue_action

        if (
            max(distances) < 0
        ):  # also activates for workpieces that are finished but exit action is not available anymore
            # workpiece is beyond latest pos stage, cant be finished anymore
            print("workpiece is beyond latest pos stage, cant be finished anymore")
            print("obs_grid: ", obs_grid)
            if self.finished_action in choose_from:
                return self.finished_action
            else:
                return np.random.choice(choose_from)


class OrderControlledLatestStage:
    def __init__(
        self, *, dummy_env, no_wp_action, finished_action, stay_in_queue_action
    ):
        self.env = dummy_env
        self.no_wp_action = no_wp_action
        self.finished_action = finished_action
        self.stay_in_queue_action = stay_in_queue_action

    def compute_action(self, obs):
        obs_grid = self.env.list_to_matrix(self.env.get_grid_state(obs), 6)

        # extract workpiece locations  and their ids
        workpiece_locations = np.argwhere(obs_grid[:, :, 1] == 1)
        workpiece_ids = obs_grid[
            workpiece_locations[:, 0], workpiece_locations[:, 1], 2
        ]

        # sort workpieces by their ids
        sorted_indices = np.argsort(workpiece_ids)
        sorted_workpiece_locations = workpiece_locations[sorted_indices]
        sorted_workpiece_ids = workpiece_ids[sorted_indices]

        # sample subactions starting with the lowest workpiece id
        action = []
        for i in range(len(sorted_workpiece_ids)):
            x, y = sorted_workpiece_locations[i]
            pg = obs_grid[x, y, 3]
            done_index = obs_grid[x, y, 4]
            needed_ab = self.env.padded_production_plans[pg, done_index]

            # get cell_index from location x, y
            cell_index = self.env.action_index_redecoding[(x, y)]

            if i == 0:  # first workpiece, only this is allowed to exit the matrix
                if needed_ab < 0:  # workpiece is finished
                    action.append(self.finished_action)
                    continue

            choose_from = self.cal_choose_from(cell_index, action)

            action.append(
                self.choose_latest_matching_machine(
                    choose_from, needed_ab, pg, done_index, obs_grid, False
                )
            )

        # select queue action
        if obs[-4] >= 1:  # workpiece in queue
            cell_index = self.env.action_index_redecoding[(-1, -1)]
            choose_from = self.cal_choose_from(cell_index, action)

            pg = obs[-3]
            done_index = 0
            needed_ab = self.env.padded_production_plans[pg, done_index]

            action.append(
                self.choose_latest_matching_machine(
                    choose_from, needed_ab, pg, done_index, obs_grid, True
                )
            )

        # reorder actions according to the environment encoding, from bottom right cell starting
        action_final = []
        for y in range(self.env.grid_size[1] - 1, -1, -1):
            for x in range(self.env.grid_size[0] - 1, -1, -1):
                if obs_grid[x, y, 1] == 0:
                    action_final.append(self.no_wp_action)
                else:
                    id_here = obs_grid[x, y, 2]
                    action_final.append(action[id_here])

        # append queue action
        if obs[-4] >= 1:  # workpiece in queue
            action_final.append(action[-1])
        else:
            action_final.append(self.no_wp_action)

        # sanity check for duplicates in action list
        seen = set()
        has_duplicates = False

        for item in action_final:
            if item == self.no_wp_action:
                continue
            if item in seen:
                has_duplicates = True
                break
            seen.add(item)

        if has_duplicates:
            print("\nDuplicates found in action list:")
            print(action_final)

        return np.array(action_final, dtype=np.int64)

    def cal_choose_from(self, cell_index, previous_actions):
        # only consider in grid actions
        # remove finished action from choose_from, this stage is not relevant anymore
        previous_actions_ = [x for x in previous_actions if x != self.finished_action]

        choose_from_downstream = self.env.downstream_relations[cell_index]

        # remove actions according to previously chosen actions
        to_remove = set()
        for pa in previous_actions_:
            to_remove.add(pa)

        choose_from = [x for x in choose_from_downstream if x not in to_remove]
        choose_from = [x for x in choose_from if x < self.env.num_cells]

        # workpiece is not allowed to pass any of the previously chosen actions
        # extract stages from previous actions
        if len(previous_actions_) > 0:
            previous_actions_stages = [
                self.env.action_index_decoding[pa][1] for pa in previous_actions_
            ]

            # remove actions that go further than the min stage
            min_stage = min(previous_actions_stages)
            choose_from = [
                x
                for x in choose_from
                if self.env.action_index_decoding[x][1] <= min_stage
            ]

        return choose_from

    def choose_latest_matching_machine(
        self, choose_from, needed_ab, pg, done_index, obs_grid, is_queue
    ):
        latest_pos_stage = self.env.dic_latest_possible_stages[pg, done_index]

        choose_from_stages = [x for x in choose_from if x < self.env.num_cells]

        pos_actions = []
        for a in choose_from_stages:
            if a < self.env.num_different_subactions_network - 3:
                pos_actions.append(
                    (
                        self.env.action_index_decoding[a][0],
                        self.env.action_index_decoding[a][1],
                    )
                )  # y locations for possible choose actions

        distances = [latest_pos_stage - a_y[1] for a_y in pos_actions]

        # Zip the lists together
        assert len(distances) == len(choose_from_stages)
        combined = list(zip(distances, choose_from_stages, pos_actions))
        # Sort by distances
        combined.sort()  # By default, sort() will sort by the first element of each tuple

        if combined == [] and is_queue:  # workpiece in queue, all locations occupied
            return self.stay_in_queue_action

        if combined == []:
            print("Fehler")

        # Unzip back into two lists
        distances_sorted, choose_from_sorted_by_distance, pos_actions_sorted = zip(
            *combined
        )

        # try to find matching machine
        for i, a in enumerate(choose_from_sorted_by_distance):
            if distances_sorted[i] >= 0:
                machine_ab = self.env.machine_abilities[
                    self.env.action_index_redecoding[pos_actions_sorted[i]]
                ]
                if (
                    machine_ab == needed_ab
                    and obs_grid[pos_actions_sorted[i][0], pos_actions_sorted[i][1], 0]
                    == 0
                ):
                    return a

        # if no matching machine was found, just move as farthest right as possible without completion violation
        for i, a in enumerate(choose_from_sorted_by_distance):
            if distances_sorted[i] >= 0:
                return a

        if needed_ab < 0:  # workpiece is finished
            return np.random.choice(choose_from)

        if is_queue:  # workpiece in queue
            # check if there is a machine available
            return self.stay_in_queue_action

        if (
            max(distances) < 0
        ):  # also activates for workpieces that are finished but exit action is not available anymore
            # workpiece is beyond latest pos stage, cant be finished anymore
            print("workpiece is beyond latest pos stage, cant be finished anymore")
            print("obs_grid: ", obs_grid)
            if self.finished_action in choose_from:
                return self.finished_action
            else:
                return np.random.choice(choose_from)


class AdvancedRandomPolicy:
    def __init__(
        self, *, dummy_env, no_wp_action, finished_action, stay_in_queue_action
    ):
        self.env = dummy_env
        self.no_wp_action = no_wp_action
        self.finished_action = finished_action
        self.stay_in_queue_action = stay_in_queue_action

    def compute_action(self, obs):
        obs_grid = self.env.list_to_matrix(self.env.get_grid_state(obs), 6)

        action = []
        cell_index = 0
        for y in range(self.env.grid_size[1] - 1, -1, -1):
            for x in range(self.env.grid_size[0] - 1, -1, -1):
                if obs_grid[x, y, 1] == 1:  # workpiece present

                    pg = obs_grid[x, y, 3]
                    done_index = obs_grid[x, y, 4]
                    needed_ab = self.env.padded_production_plans[pg, done_index]

                    choose_from = self.cal_choose_from(cell_index, action)

                    if needed_ab < 0:  # workpiece is finished
                        if self.finished_action in choose_from:
                            action.append(self.finished_action)
                        else:
                            action.append(
                                self.random_choice_matching_machine(
                                    choose_from=choose_from,
                                    pg=pg,
                                    done_index=done_index,
                                    column=y,
                                    needed_ab=needed_ab,
                                    obs_grid=obs_grid,
                                    is_queue=False,
                                )
                            )

                    else:
                        action.append(
                            self.random_choice_matching_machine(
                                choose_from=choose_from,
                                pg=pg,
                                done_index=done_index,
                                column=y,
                                needed_ab=needed_ab,
                                obs_grid=obs_grid,
                                is_queue=False,
                            )
                        )
                else:
                    action.append(self.no_wp_action)

                cell_index += 1

        # queue action:
        if obs[-4] >= 1:  # workpiece in queue
            stay_here_action = self.stay_in_queue_action

            choose_from = self.cal_choose_from(self.env.num_cells, action)

            pg = obs[-3]
            done_index = 0
            needed_ab = self.env.padded_production_plans[pg, done_index]

            if needed_ab < 0:  # workpiece is finished
                if self.finished_action in choose_from:
                    action.append(self.finished_action)
                else:
                    action.append(stay_here_action)
            else:
                action.append(
                    self.random_choice_matching_machine(
                        choose_from=choose_from,
                        pg=pg,
                        done_index=done_index,
                        column=-1,
                        needed_ab=needed_ab,
                        obs_grid=obs_grid,
                        is_queue=True,
                    )
                )
        else:
            action.append(self.no_wp_action)

        # sanity check for duplicates in action list
        seen = set()
        has_duplicates = False

        for item in action:
            if item == self.no_wp_action:
                continue
            if item in seen:
                has_duplicates = True
                break
            seen.add(item)

        if has_duplicates:
            print("\nDuplicates found in action list:")
            print(action)

        return np.array(action, dtype=np.int64)

    def cal_choose_from(self, cell_index, previous_actions):
        choose_from = self.env.downstream_relations[cell_index]

        # remove actions according to previously chosen actions
        to_remove = set()
        for pa in previous_actions:
            to_remove.add(pa)

        choose_from = [x for x in choose_from if x not in to_remove]

        return choose_from

    def random_choice_matching_machine(
        self, *, choose_from, pg, done_index, column, needed_ab, obs_grid, is_queue
    ):
        latest_pos_stage = self.env.dic_latest_possible_stages[pg, done_index]
        if (
            column > latest_pos_stage
        ):  # workpiece is beyond latest pos stage, cant be finished anymore
            if self.finished_action in choose_from:
                return self.finished_action
            else:
                return np.random.choice(choose_from)

        choose_from = [x for x in choose_from if x < self.env.num_cells]

        if choose_from == [] and is_queue:
            return self.stay_in_queue_action

        # shuffle choose from for random choice
        np.random.shuffle(np.array(choose_from))

        # choose the first action that is available and matches the needed ability
        for a in choose_from:
            machine_ab = self.env.machine_abilities[a]
            if (
                machine_ab == needed_ab
                and obs_grid[
                    self.env.action_index_decoding[a][0],
                    self.env.action_index_decoding[a][1],
                    0,
                ]
                == 0
            ):
                return a

        # no machine is matching and available, choose a matching failed one
        for a in choose_from:
            machine_ab = self.env.machine_abilities[a]
            if (
                machine_ab == needed_ab
                and obs_grid[
                    self.env.action_index_decoding[a][0],
                    self.env.action_index_decoding[a][1],
                    0,
                ]
                == 1
            ):
                return a

        # no matching machine was found, randomly choose one
        return np.random.choice(choose_from)


class SillyRandomPolicy:
    def __init__(
        self, *, dummy_env, no_wp_action, finished_action, stay_in_queue_action
    ):
        self.env = dummy_env
        self.no_wp_action = no_wp_action
        self.finished_action = finished_action
        self.stay_in_queue_action = stay_in_queue_action

    def compute_action(self, obs):
        obs_grid = self.env.get_grid_state(obs)

        action = []
        for cell_index in range(self.env.num_cells):
            wp_present = obs[cell_index * 6 + 1] == 1

            if wp_present:
                choose_from = self.env.downstream_relations[cell_index]

                choose_from = [a for a in choose_from if a < self.env.num_cells]
                choose_from.append(self.finished_action)

                # Collect values to remove in a set
                to_remove = set()
                for pa in action:
                    to_remove.add(pa)

                # Filter the choose_from list
                choose_from = [x for x in choose_from if x not in to_remove]
                sub_action = np.random.choice(choose_from)
            else:
                sub_action = self.no_wp_action

            action.append(sub_action)

        if obs[-4] == 1:
            choose_from = self.env.downstream_relations[cell_index]
            choose_from = [a for a in choose_from if a <= self.env.num_cells]

            # Collect values to remove in a set
            to_remove = set()
            for pa in action:
                to_remove.add(pa)

            # Filter the choose_from list
            choose_from = [x for x in choose_from if x not in to_remove]

            if choose_from == []:
                action.append(self.stay_in_queue_action)
            else:
                sub_action = np.random.choice(choose_from)
                action.append(sub_action)

        else:
            action.append(self.no_wp_action)

        # sanity check for duplicates in action list
        seen = set()
        has_duplicates = False

        for item in action:
            if item == self.no_wp_action:
                continue
            if item in seen:
                has_duplicates = True
                break
            seen.add(item)

        if has_duplicates:
            print("\nDuplicates found in action list:")
            print(action)

        return np.array(action, dtype=np.int64)
