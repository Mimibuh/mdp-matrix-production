from train.environment import Matrix_production
from train.config import ENV_CONFIGS


def create_dummy_env(env_name):
    selected_config = ENV_CONFIGS[env_name]
    return Matrix_production(selected_config)
