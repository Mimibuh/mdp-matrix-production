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

    def get_remaining_service_old(self, pg, index_current_todo):
        # print("calculate remaining service time... pg: ", pg, "  index_current_todo: ", index_current_todo)
        rt = 0
        current_ability = self.production_plans[pg][index_current_todo]
        for ability in self.production_plans[pg][index_current_todo + 1 :]:
            if ability == current_ability:
                rt += 1
            else:
                break

        return rt
