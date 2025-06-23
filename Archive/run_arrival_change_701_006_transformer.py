from pathlib import Path
from train.config import ENV_CONFIGS
from train.DefaultTrainer import DefaultTrainer
from tools.tools_rl_module import create_module_spec, create_model_config_transformer

# test rising penalty in reward


def main():
    env_name = "simple_3by3_01_06"

    load_pretraining = False
    policy_name = "latest_stage"
    model_name = "transformer"
    pretrain_path = f"{env_name}_{policy_name}_{model_name}"

    module_spec = create_module_spec(
        model_config=create_model_config_transformer(
            environment_name=env_name, with_prescaler=True
        )
    )

    DefaultTrainer(
        run_name=Path(__file__).stem,
        environment=env_name,
        module_spec=module_spec,
        run_comments=f"""reward_type: {ENV_CONFIGS[env_name]["reward_type"]}, with entropy test but no schedule
    """,
        learning_rate=[[0, 0.001], [100000000, 0.0001]],
        entropy_coeff=[[0, 0.001], [100000000, 0.0001]],
        lambda_=0.2,
        vl_clip_param=5,
        gamma=0.99,
        custom_rise_iterations=300,
    ).train()


if __name__ == "__main__":
    main()


# notes

# ENV CONFIG
# simple_3by3=dict(
#         env_name="simple_3by3",
#         verbosity_env=0,
#         verbosity_net=0,
#         max_steps=200,
#         grid_size=(3, 3),
#         machine_or_buffer=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1]),
#         failure_probs=[-1, -1, 0.6, 0, 0.4, 0.1, 0.1, 0.2, 0.3],
#         machine_abilities=[-1, -1, 3, 1, 1, 2, 0, 0, 0],
#         product_groups=[0, 1, 2],
#         production_plans=[
#             [0, 0, 2, 1, 2],  # for pg 0
#             [0, 3],  # for pg 1
#             [0, 1, 1, 1],  # for pg 2
#         ],
#         padded_production_plans=np.array(
#             [
#                 [0, 0, 2, 1, 2, -2],  # for pg 0
#                 [0, 3, -2, -2, -2, -2],  # for pg 1
#                 [0, 1, 1, 1, -2, -2],  # for pg 2
#             ]
#         ),
#         arrival_interval=1,
#         number_of_workpieces=12,
#         arrival_pattern=[0, 1, 2],
#         reset_strategy=dict(start_zero=False, utilization=0.6, control_done_index=True),
#         reward_type="simple_01",
#         reward_params=dict(
#             finished_wps_reward_base=1.1,
#             beta=1,
#             bonus=2,
#             no_finished_wps_penalty=0,
#             invalid_action_pen_for_one=0,
#         ),
#     ),


# VALIDATION DONFIG
# selected_config["reset_strategy"] = {
#             "start_zero": True,
#             "utilization": 0.6,
#             "control_done_index": True,
#         }
#         reward_params = dict(
#             finished_wps_reward_base=1.5,
#             beta=0.9,
#             bonus=2,
#             no_finished_wps_penalty=0,
#             invalid_action_pen_for_one=0,
#         )
