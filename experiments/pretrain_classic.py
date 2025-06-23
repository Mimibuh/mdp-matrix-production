from tools.tools_environment import create_dummy_env
from tools.tools_rl_module import create_model_config_conv_2d_deeper
from train.DefaultPretrainer import DefaultPretrainer


def main():

    policy_name = "latest_stage"
    env_name = "simple_3by3"
    model_architecture = "conv2ddeeper"  # change in config function if needed!
    with_prescaler = True

    # Create dummy environment
    temp_env = create_dummy_env(env_name=env_name)
    action_dims = list(temp_env.action_space.nvec)

    model_config = create_model_config_conv_2d_deeper(
        environment_name=env_name, with_prescaler=with_prescaler
    )

    pretrain_config = dict(
        environment_name=env_name,
        policy_name=policy_name,
        model_architecture=model_architecture,
        action_dims=action_dims,
    )

    # Create pretraining model
    pre_net = DefaultPretrainer(model_config, pretrain_config)
    pre_net.setup(
        observation_space=temp_env.observation_space, action_space=temp_env.action_space
    )

    pre_net.collect_data(100000)
    pre_net.pretrain(epochs=100)


if __name__ == "__main__":
    main()
