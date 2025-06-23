from pathlib import Path

from ray.rllib.core.rl_module import RLModuleSpec

from tools.tools_hydra import hydra_proxy
from train.ModelSimpleLinear import ModelSimpleLinear
from train.DefaultPrescaler import DefaultPrescaler

# from train.ModelSimpleConv2d import ModelSimpleConv2d
from train.config import ENV_CONFIGS
from train.environment import Matrix_production
from train.DefaultRLModule import DefaultRLModule
from train.DefaultValueHead import DefaultValueHead
from train.DefaultPolicyHead import DefaultPolicyHead
from train.DefaultActionSampler import DefaultActionSampler
from train.DefaultTrainer import DefaultTrainer


def create_model_config(environment_name: str, *, with_prescaler: bool):
    # Define the RLModuleSpec with correct parameters
    selected_config = ENV_CONFIGS[environment_name]
    temp_env = Matrix_production(selected_config)  # Make sure this works properly
    gs = temp_env.grid_size
    downstream_relations = temp_env.downstream_relations

    shared_encoder = hydra_proxy(ModelSimpleLinear)(grid_size=gs, out_features=256)
    if with_prescaler:
        shared_encoder = hydra_proxy(DefaultPrescaler)(model=shared_encoder)

    return dict(
        action_sampler=hydra_proxy(DefaultActionSampler)(
            downstream_relations=downstream_relations
        ),
        shared_encoder=shared_encoder,
        policy_head=hydra_proxy(DefaultPolicyHead)(),
        value_head=hydra_proxy(DefaultValueHead)(),
    )


def create_module_spec(*, model_config):
    module_spec = RLModuleSpec(
        module_class=DefaultRLModule, model_config=model_config, catalog_class=None
    )
    return module_spec


def main():
    env_name = "most_simple_1by3"
    module_spec = create_module_spec(
        model_config=create_model_config(environment_name=env_name, with_prescaler=True)
    )
    DefaultTrainer(
        run_name=Path(__file__).stem,
        environment=env_name,
        module_spec=module_spec,
        run_comments="""
    current default test
    """,
    ).train()


if __name__ == "__main__":
    main()

# +additional_info_filename = "mimi_laptop"
# +additional_notes_wandb = "testing new reward shaping and convolutional network"
# -net_architecture = "simple_linear"
# +net_architecture = "covolutional"
# +# net_architecture = "linear"
#  # net_architecture = "transformer"
