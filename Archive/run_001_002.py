from pathlib import Path
from tools.tools_rl_module import create_module_spec
from tools.tools_rl_module import create_model_config_conv_2d_deeper
from train.DefaultTrainer import DefaultTrainer


def main():
    env_name = "X1_stochastic"
    module_spec = create_module_spec(
        model_config=create_model_config_conv_2d_deeper(
            environment_name=env_name, with_prescaler=False
        )
    )
    DefaultTrainer(
        run_name=Path(__file__).stem,
        environment=env_name,
        module_spec=module_spec,
        run_comments="""
    test without prescaler
    """,
    ).train()


if __name__ == "__main__":
    main()
