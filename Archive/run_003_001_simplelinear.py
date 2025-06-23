from pathlib import Path
from train.config import ENV_CONFIGS
from train.DefaultTrainer import DefaultTrainer
from tools.tools_rl_module import create_module_spec
from tools.tools_rl_module import create_model_config_linear


def main():
    env_name = "simple_3by3"

    load_pretraining = False
    policy_name = "latest_stage"
    model_name = "simplelinear"
    pretrain_path = f"{env_name}_{policy_name}_{model_name}"

    module_spec = create_module_spec(
        model_config=create_model_config_linear(
            environment_name=env_name,
            with_prescaler=True,
            load_pretraining=load_pretraining,
            pretrain_path=pretrain_path,
        )
    )
    DefaultTrainer(
        run_name=Path(__file__).stem,
        environment=env_name,
        module_spec=module_spec,
        run_comments=f"""reward_type: {ENV_CONFIGS[env_name]["reward_type"]}, with learning rate schedule
    """,
        learning_rate=0.001,
        lr_schedule=[[0, 0.001], [100000, 0.0001]],
    ).train()


if __name__ == "__main__":
    main()

# +additional_info_filename = "mimi_laptop"
# +additional_notes_wandb = "testing new reward shaping and convolutional network"
# -net_architecture = "simple_linear"
# +net_architecture = "covolutional"
# +# net_architecture = "linear"
#  # net_architecture = "transformer"
