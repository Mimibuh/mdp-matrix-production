from tools.global_config import custom_checkpoint_path
from tools.tools_environment import create_dummy_env
from ray.rllib.core.rl_module import RLModule
import pathlib
from tools.tools_hydra import hydra_proxy
from train.ModelSimpleConv2d import (
    ModelSimpleConv2dDeeper,
    ModelSimpleConv2dDeeperRedecide,
)
from train.ModelSimpleConv2dResNet import ModelSimpleConv2dResNet
from train.ModelSimpleLinear import ModelSimpleLinear
from train.DefaultPrescaler import DefaultPrescaler, DefaultPrescaler_Redecide
from train.DefaultActionSampler import DefaultActionSampler
from train.DefaultPolicyHead import DefaultPolicyHead
from train.DefaultValueHead import DefaultValueHead
from ray.rllib.core.rl_module import RLModuleSpec
from train.DefaultRLModule import DefaultRLModule, DefaultRLModule_Redecision
from train.ModelSimpleTransformer import ModelSimpleTransformer

"""
Tools for creating shared encoder configs, and everything else compromised to rl modules /spec
"""


def load_rl_module(checkpoint_filename, trainstep):

    checkpoint_path = (
        custom_checkpoint_path / checkpoint_filename / f"trainstep_{trainstep}"
    )

    rl_module_path = (
        pathlib.Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    )
    rl_module = RLModule.from_checkpoint(str(rl_module_path))["default_policy"]

    return rl_module


def create_module_spec(*, model_config):
    module_spec = RLModuleSpec(
        module_class=DefaultRLModule, model_config=model_config, catalog_class=None
    )
    return module_spec


def create_module_spec_Redecision(*, model_config):
    module_spec = RLModuleSpec(
        module_class=DefaultRLModule_Redecision,
        model_config=model_config,
        catalog_class=None,
    )
    return module_spec


def create_model_config_linear(
    environment_name: str,
    *,
    with_prescaler: bool,
    load_pretraining: bool = False,
    pretrain_path=None,
):
    # Define the RLModuleSpec with correct parameters
    temp_env = create_dummy_env(environment_name)
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
        load_pretraining=load_pretraining,
        pretrain_path=pretrain_path,
    )


def create_model_config_conv_2d_deeper(
    environment_name: str,
    *,
    with_prescaler: bool,
    load_pretraining: bool = False,
    pretrain_path=None,
):
    # Define the RLModuleSpec with correct parameters
    temp_env = create_dummy_env(environment_name)
    gs = temp_env.grid_size
    downstream_relations = temp_env.downstream_relations

    shared_encoder = hydra_proxy(ModelSimpleConv2dDeeper)(
        grid_size=gs, out_features=256
    )
    if with_prescaler:
        shared_encoder = hydra_proxy(DefaultPrescaler)(model=shared_encoder)

    return dict(
        action_sampler=hydra_proxy(DefaultActionSampler)(
            downstream_relations=downstream_relations
        ),
        shared_encoder=shared_encoder,
        policy_head=hydra_proxy(DefaultPolicyHead)(),
        value_head=hydra_proxy(DefaultValueHead)(),
        load_pretraining=load_pretraining,
        pretrain_path=pretrain_path,
    )


def create_model_config_conv2d_resnet(environment_name: str, *, with_prescaler: bool):
    # Define the RLModuleSpec with correct parameters
    temp_env = create_dummy_env(environment_name)
    gs = temp_env.grid_size
    downstream_relations = temp_env.downstream_relations

    shared_encoder = hydra_proxy(ModelSimpleConv2dResNet)(
        grid_size=gs, out_features=256
    )
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


def create_model_config_conv_2d_deeper_redecide(
    environment_name: str,
    *,
    with_prescaler: bool,
    load_pretraining: bool = False,
    pretrain_path=None,
):
    # Define the RLModuleSpec with correct parameters
    temp_env = create_dummy_env(environment_name)
    gs = temp_env.grid_size
    downstream_relations = temp_env.downstream_relations

    shared_encoder = hydra_proxy(ModelSimpleConv2dDeeperRedecide)(
        grid_size=gs, out_features=256
    )
    if with_prescaler:
        shared_encoder = hydra_proxy(DefaultPrescaler_Redecide)(model=shared_encoder)

    return dict(
        action_sampler=hydra_proxy(DefaultActionSampler)(
            downstream_relations=downstream_relations
        ),
        shared_encoder=shared_encoder,
        policy_head=hydra_proxy(DefaultPolicyHead)(),
        value_head=hydra_proxy(DefaultValueHead)(),
        load_pretraining=load_pretraining,
        pretrain_path=pretrain_path,
    )


def create_model_config_transformer(environment_name: str, *, with_prescaler: bool):
    # Define the RLModuleSpec with correct parameters
    temp_env = create_dummy_env(environment_name)
    gs = temp_env.grid_size
    downstream_relations = temp_env.downstream_relations

    shared_encoder = hydra_proxy(ModelSimpleTransformer)(grid_size=gs, out_features=256)
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


def create_module_spec_test(*, model_config, environment_name):
    dummy_env = create_dummy_env(environment_name)

    module_spec = RLModuleSpec(
        module_class=DefaultRLModule,
        model_config=model_config,
        observation_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        catalog_class=None,
    )
    return module_spec
