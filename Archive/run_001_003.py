from pathlib import Path

from tools.tools_rl_module import create_module_spec
from train.DefaultTrainer import DefaultTrainer
from tools.tools_rl_module import create_model_config_linear


def main():
    env_name = "X1_stochastic"
    module_spec = create_module_spec(
        model_config=create_model_config_linear(
            environment_name=env_name, with_prescaler=True
        )
    )
    DefaultTrainer(
        run_name=Path(__file__).stem,
        environment=env_name,
        module_spec=module_spec,
        run_comments="",
        learning_rate=0.0006,
    ).train()


if __name__ == "__main__":
    main()

# +additional_info_filename = "mimi_laptop"
# +additional_notes_wandb = "testing new reward shaping and convolutional network"
# -net_architecture = "simple_linear"
# +net_architecture = "covolutional"
# +# net_architecture = "linear"
#  # net_architecture = "transformer"
