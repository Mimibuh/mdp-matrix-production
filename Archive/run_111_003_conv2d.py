from pathlib import Path
from train.config import ENV_CONFIGS
from train.DefaultTrainer import DefaultTrainer
from tools.tools_rl_module import (
    create_model_config_conv_2d_deeper_redecide,
    create_module_spec_Redecision,
)


def main():
    env_name = "simple_3by3_nofailure"

    load_pretraining = False
    policy_name = "latest_stage"
    model_name = "conv2ddeeper"
    pretrain_path = f"{env_name}_{policy_name}_{model_name}"

    module_spec = create_module_spec_Redecision(
        model_config=create_model_config_conv_2d_deeper_redecide(
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
        run_comments=f"""reward_type: {ENV_CONFIGS[env_name]["reward_type"]}, with entropy test but no schedule
    """,
        learning_rate=[[0, 0.001], [10000000, 0.0001]],
        entropy_coeff=0.01,
        lambda_=0.5,
    ).train()


if __name__ == "__main__":
    main()

# +additional_info_filename = "mimi_laptop"
# +additional_notes_wandb = "testing new reward shaping and convolutional network"
# -net_architecture = "simple_linear"
# +net_architecture = "covolutional"
# +# net_architecture = "linear"
#  # net_architecture = "transformer"
