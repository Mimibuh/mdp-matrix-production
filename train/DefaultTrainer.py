import time
from datetime import datetime

import numpy as np
import ray
import wandb
from matplotlib import pyplot as plt
from ray.rllib.algorithms import PPOConfig, PPO
from ray.rllib.utils.typing import RLModuleSpecType
from ray.tune import register_env

from test.DefaultValidator import DefaultValidator
from train.config import ENV_CONFIGS
from tools.global_config import custom_checkpoint_path
from train.environment import Matrix_production
from tools.tools_matplot import plot_random_steps_rl_module

import io
from PIL import Image

"""
DefaultTrainer for training RL modules using PPO algorithm in Ray RLlib.

Accpets a number of hyperparams, tracks all information in wandb.

Trains the rl module with ppo from rllib. Validation metrics are comuetd every 10 iterations and tracked in wandb.
"""


class DefaultTrainer:
    def __init__(
        self,
        *,
        environment,
        run_name,
        run_comments,
        module_spec: RLModuleSpecType,
        plot_state_action: bool = True,
        learning_rate=0.0003,
        gamma: float = 0.99,
        sgd_minibatch_size: int = 128,
        big_batch_size: int = 4096,
        train_iterations=10000,
        entropy_coeff=0.0,
        load_previous_model: bool = False,
        checkpoint_filename=None,
        trainstep=None,
        use_gae: bool = True,
        lambda_: float = 0.9,
        vl_clip_param: float = 5,
        clip_param: float = 0.2,
        grad_clip: float = 1,
        custom_rise_iterations=None,
    ):
        self.run_name = run_name
        self.run_comments = run_comments
        self.plot_state_action = plot_state_action
        self.load_previous_model = load_previous_model
        self.checkpoint_filename = checkpoint_filename
        self.trainstep = trainstep

        self.train_iterations = train_iterations
        self.sgd_minibatch_size = sgd_minibatch_size
        self.big_batch_size = big_batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.entropy_coeff = entropy_coeff
        self.clip_param = clip_param
        # self.env_runners = 0
        self.environment_name = environment
        self.module_spec = module_spec
        self.num_sgd_iter = 10
        self.vf_loss_coeff = 1
        self.vf_clip_param = vl_clip_param
        self.use_kl_loss = False
        self.kl_coeff = 0.0
        self.kl_target = 0.0
        self.use_gae = use_gae
        self.lambda_ = lambda_

        self.custom_rise_iterations = custom_rise_iterations

    def train(self):
        ENV_CONFIGS[self.environment_name]["reward_params"][
            "custom_rise_iterations"
        ] = self.custom_rise_iterations

        environment_name = self.environment_name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        base_checkpoint_dir = (
            custom_checkpoint_path / f"{environment_name}_{self.run_name}_t_{timestamp}"
        )
        custom_checkpoint_path.mkdir(parents=True, exist_ok=True)
        base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def env_creator(env_config):
            print("Creating environment with env_config:", env_config)
            return Matrix_production(env_config)

        register_env("Matrix_production", env_creator)

        # Use the default or specified config
        selected_config = ENV_CONFIGS[environment_name]

        # configs:
        env_max_steps = selected_config["max_steps"]
        reset_strat = selected_config["reset_strategy"]
        reward_type = selected_config["reward_type"]
        reward_params = selected_config["reward_params"]

        wandb.init(
            project="matrix_prod",
            name=f"{environment_name}_{self.run_name}_{timestamp}",  # Optional: Set a descriptive run name
            config=dict(
                train_iterations=self.train_iterations,
                sgd_minibatch_size=self.sgd_minibatch_size,
                big_batch_size=self.big_batch_size,
                env_max_steps=env_max_steps,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                grad_clip=self.grad_clip,
                entropy_coeff=self.entropy_coeff,
                use_gae=self.use_gae,
                lambda_=self.lambda_,
                clip_param=self.clip_param,
                num_sgd_iter=self.num_sgd_iter,
                vf_loss_coeff=self.vf_loss_coeff,
                vf_clip_param=self.vf_clip_param,
                use_kl_loss=self.use_kl_loss,
                kl_coeff=self.kl_coeff,
                kl_target=self.kl_target,
                reward_type=reward_type,
                reward_params=reward_params,
                reset_strategy=reset_strat,
                custom_rise_iterations=self.custom_rise_iterations,
                additional_notes_wandb=self.run_comments,
                date=datetime.now().strftime("%Y-%m-%d"),
                time=datetime.now().strftime("%H:%M:%S"),
            ),
        )

        ray.shutdown()

        config = (
            PPOConfig()
            .environment(env="Matrix_production", env_config=selected_config)
            .framework("torch")
            .rl_module(rl_module_spec=self.module_spec)
            .env_runners(num_env_runners=0)
            # .learners(num_learners=4)
            .training(
                lr=self.learning_rate,
                minibatch_size=self.sgd_minibatch_size,
                train_batch_size=self.big_batch_size,
                grad_clip=self.grad_clip,
                entropy_coeff=self.entropy_coeff,
                clip_param=self.clip_param,
                gamma=self.gamma,
                num_sgd_iter=self.num_sgd_iter,
                vf_loss_coeff=self.vf_loss_coeff,
                vf_clip_param=self.vf_clip_param,
                use_kl_loss=self.use_kl_loss,
                kl_coeff=self.kl_coeff,
                kl_target=self.kl_target,
                use_gae=self.use_gae,
                lambda_=self.lambda_,
            )
        )

        trainer: PPO = config.build()

        if self.load_previous_model:
            print("Loading previous model from checkpoint:", self.checkpoint_filename)
            path = (
                custom_checkpoint_path
                / self.checkpoint_filename
                / f"trainstep_{self.trainstep}"
            )
            trainer.restore(str(path))

        # Start training the PPO agent
        for iteration in range(self.train_iterations):
            # update current train iteration in config
            ENV_CONFIGS[environment_name]["reward_params"][
                "current_train_iteration"
            ] = iteration

            iter_start_time = time.time()  # Start timer
            print("train iteration: ", iteration)

            result = trainer.train()

            iter_end_time = time.time()  # End timer
            iteration_time = iter_end_time - iter_start_time  # Compute elapsed time

            # Extract learner stats safely (in case they're missing)
            env_stats = result.get("env_runners", {})
            learner_stats = result.get("learners", {}).get("default_policy", {})

            # Log to wandb
            wandb.log(
                dict(
                    reward=env_stats.get("episode_return_mean"),
                    iteration_time_sec=iteration_time,
                    iteration=iteration,
                    policy_entropy=learner_stats.get("entropy"),
                    policy_kl=learner_stats.get("mean_kl_loss"),
                    vf_loss=learner_stats.get("vf_loss"),
                    policy_loss=learner_stats.get("policy_loss"),
                    total_loss=learner_stats.get("total_loss"),
                    learning_rate=learner_stats.get("default_optimizer_learning_rate"),
                    entropy_coeff=learner_stats.get("curr_entropy_coeff"),
                ),
                step=iteration,
            )

            if self.plot_state_action:
                mod = trainer.get_module()
                num_plot_steps = 5

                fig = plot_random_steps_rl_module(
                    num_plot_steps=num_plot_steps,
                    environment_name=environment_name,
                    rl_module=mod,
                )

                # Save the figure to a BytesIO buffer with high DPI for sharpness
                with io.BytesIO() as buf:
                    fig.savefig(buf, format="png", dpi=300)
                    buf.seek(0)
                    img = Image.open(buf)
                    # Log figure to WandB
                    wandb.log(dict(Grid_Visualization=wandb.Image(img)), step=iteration)

                # Close the figure to free memory
                plt.close(fig)

            if iteration % 10 == 0:
                # save model, make sure the path exists
                checkpoint_filename = f"trainstep_{iteration}"

                checkpoint_path = base_checkpoint_dir / checkpoint_filename
                checkpoint_path = trainer.save_to_path(str(checkpoint_path))
                print("Checkpoint saved to:", checkpoint_path)

                # test validation here
                val = DefaultValidator(environment_name)
                mod = trainer.get_module()
                test_seeds, val_results = val.test_rl_model(mod)

                wandb.log(
                    dict(
                        val_mean_reward=np.mean(val_results["rewards"]),
                        val_mean_both_cor_finishes=np.mean(
                            val_results["no_cor_both_finishes"]
                        ),
                        val_mean_order_cor_finishes=np.mean(
                            val_results["no_cor_order_finishes"]
                        ),
                        val_mean_plan_cor_finishes=np.mean(
                            val_results["no_cor_plan_finishes"]
                        ),
                        val_mean_total_finishes=np.mean(
                            val_results["no_total_finishes"]
                        ),
                        val_std_reward=val_results["rewards"].std(),
                        val_max_reward=val_results["rewards"].max(),
                        val_min_reward=val_results["rewards"].min(),
                    ),
                    step=iteration,
                )
