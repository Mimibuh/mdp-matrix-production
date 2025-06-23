from pathlib import Path
from train.config import ENV_CONFIGS
from train.DefaultTrainer import DefaultTrainer
from tools.tools_rl_module import create_module_spec, create_model_config_conv_2d_deeper

# test rising penalty in reward


def main():
    env_name = "simple_3by3_oldarrival"

    load_pretraining = False
    policy_name = "latest_stage"
    model_name = "conv2d"
    pretrain_path = f"{env_name}_{policy_name}_{model_name}"

    module_spec = create_module_spec(
        model_config=create_model_config_conv_2d_deeper(
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
        learning_rate=[[0, 0.001], [100000000, 0.0001]],
        entropy_coeff=[[0, 0.001], [100000000, 0.0001]],
        lambda_=0.2,
        vl_clip_param=5,
        gamma=0.99,
        custom_rise_iterations=300,
    ).train()


if __name__ == "__main__":
    main()
