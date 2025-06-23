from tools.tools_rl_module import create_model_config_linear
from tools.tools_rl_module import create_module_spec_test
from tools.tools_environment import create_dummy_env
import torch
import numpy as np
from test.DefaultPolicyMaker import DefaultPolicyMaker


def main():
    env_name = "simple_3by3"
    load_pretraining = False
    policy_name = "latest_stage"
    model_name = "simplelinear"
    pretrain_path = f"{env_name}_{policy_name}_{model_name}"

    # Create the model configuration
    model_config = create_model_config_linear(
        environment_name=env_name,
        with_prescaler=True,
        load_pretraining=load_pretraining,
        pretrain_path=pretrain_path,
    )

    # Create the module spec from the model configuration.
    module_spec = create_module_spec_test(
        model_config=model_config, environment_name=env_name
    )

    # Instantiate your RL module using the module spec.
    # The exact method depends on your implementation of create_module_spec.
    # For example, if your module spec has a build() method:
    rl_module = module_spec.build()  # This should return an instance of DefaultRLModule

    # Run setup to initialize all the sub-modules (shared encoder, policy head, etc.)
    rl_module.setup()

    test_env = create_dummy_env(env_name)
    test_env.reset()

    obs_batch = [test_env.state]

    print("observation: ", obs_batch[0])

    pol_mk = DefaultPolicyMaker("latest_stage", env_name)
    own_policy = pol_mk.policy
    policy_suggested_action = own_policy.compute_action(obs_batch[0])
    print("policy suggested action: ", policy_suggested_action)

    obs_batch = np.array(obs_batch, dtype=np.float32)
    obs_batch = torch.tensor(obs_batch, dtype=torch.float32)

    for _ in range(1):
        # Run an inference forward pass.
        inference_output = rl_module._pi(obs_batch, inference=False)
        print("Inference output:", inference_output)

        if np.array_equal(inference_output, policy_suggested_action):
            print("same sampled! ")
            print("rl action: ", inference_output)

    # Optionally, test value estimation.
    # value_output = rl_module.compute_values(batch)
    # print("Value estimates:", value_output)


if __name__ == "__main__":
    main()
