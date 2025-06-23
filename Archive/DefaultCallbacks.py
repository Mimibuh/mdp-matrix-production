from ray.rllib.algorithms.callbacks import DefaultCallbacks


def update_env_reward_params(env, beta):
    # Check if the environment is vectorized (i.e., has an "envs" attribute).
    if hasattr(env, "envs"):
        for sub_env in env.envs:
            update_env_reward_params(sub_env, beta)
        return
    else:
        print("doesnt have envs")

    # Try to get the underlying environment.
    if hasattr(env, "unwrapped"):
        base_env = env.unwrapped
    elif hasattr(env, "env"):
        base_env = env.env
    else:
        base_env = env

    if hasattr(base_env, "reward_params"):
        base_env.reward_params.update({"beta": beta})
    else:
        print(
            f"Environment {base_env} (type: {type(base_env)}) has no attribute 'reward_params'."
        )


class BetaAnnealingCallback(DefaultCallbacks):
    def __init__(self, initial_beta=2.0, final_beta=0.1, max_iterations=1000):
        super().__init__()
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.max_iterations = max_iterations

    def on_train_result(self, *args, **kwargs):
        trainer = kwargs.get("trainer") or kwargs.get("algorithm")
        if trainer is None:
            print("Trainer not provided in on_train_result callback.")
            return

        iteration = trainer.iteration
        progress = min(iteration / self.max_iterations, 1.0)
        new_beta = self.initial_beta - (self.initial_beta - self.final_beta) * progress

        def update_runner(runner):
            if hasattr(runner, "foreach_env"):
                runner.foreach_env(lambda env: update_env_reward_params(env, new_beta))
            elif hasattr(runner, "env"):
                update_env_reward_params(runner.env, new_beta)
            else:
                print("Runner does not have 'foreach_env' or 'env' attribute.")

        try:
            trainer.env_runner_group.foreach_env_runner(update_runner)
        except Exception as e:
            print("Error updating env runner group:", e)

        print(f"Iteration {iteration}: Updated beta to {new_beta}")
