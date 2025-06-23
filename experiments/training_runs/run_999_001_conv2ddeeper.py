# with benchmark pretrained on ppo module by reward shaping
# reload in Default Trainer


from pathlib import Path
from train.config import ENV_CONFIGS
from train.DefaultTrainer import DefaultTrainer

from tools.tools_rl_module import create_module_spec
from tools.tools_rl_module import create_model_config_conv_2d_deeper


def main():
    env_name = "simple_3by3"

    module_spec = create_module_spec(
        model_config=create_model_config_conv_2d_deeper(
            environment_name=env_name,
            with_prescaler=True,
            load_pretraining=False,
            pretrain_path=None,
        )
    )

    DefaultTrainer(
        run_name=Path(__file__).stem,
        environment=env_name,
        module_spec=module_spec,
        run_comments=f"""reward_type: {ENV_CONFIGS[env_name]["reward_type"]}, with entropy test but no schedule
    """,
        learning_rate=0.0001,
        entropy_coeff=0.01,
        load_previous_model=True,
        checkpoint_filename="simple_3by3_run_002_001_conv2ddeeper_t_2025-04-07_23-16-41_benchpolicy_repl_ssh2",
        trainstep=1460,
    ).train()


if __name__ == "__main__":
    main()

# +additional_info_filename = "mimi_laptop"
# +additional_notes_wandb = "testing new reward shaping and convolutional network"
# -net_architecture = "simple_linear"
# +net_architecture = "covolutional"
# +# net_architecture = "linear"
#  # net_architecture = "transformer"
