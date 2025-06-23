from pathlib import Path


from test.DefaultValidator import DefaultValidator
from tools.tools_metrics import print_single_results
from tools.tools_rl_module import load_rl_module
from test.DefaultPolicyMaker import DefaultPolicyMaker

# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
model_folder_name = (
    r"simple_3by3_01_run_arrival_change_201_006_transformer_t_2025-04-26_17-01-32"
)
trainstep = 660
environment_name = r"simple_3by3_01"
policy_name = "latest_stage"

# csv_files = [r"val_mean_cor_finished.csv", r"val_mean_reward.csv", r"train_episode_mean_reward.csv"]
##csv_plot_legend_labels = ["val_num_correctly_finished", "val_episode_mean_reward", "train_episode_mean_reward"]
# csv_plot_titles = ["Validation: mean correctly finished workpieces", "Validation: mean episode reward", "Training: mean episode reward"]


# automated settings
custom_path_interesting_runs = (
    r"C:\Users\mimib\Desktop\Masterarbeit Produktionsmanagement\interesting_runs"
)

validator100 = DefaultValidator(environment_name, test_seeds=list(range(2)))
rl_mod = load_rl_module(
    checkpoint_filename=Path(custom_path_interesting_runs)
    / model_folder_name
    / "rl_mod",
    trainstep=trainstep,
)
pol_mk = DefaultPolicyMaker(policy_name, environment_name)
own_policy = pol_mk.policy

used_seeds_hybrid, results_hybrid = validator100.test_hybrid(
    rl_module=rl_mod, policy=own_policy
)
used_seeds, results_rl = validator100.test_rl_model(rl_module=rl_mod)
used_seeds_policy, results_policy = validator100.test_own_policy(policy=own_policy)

print_single_results(results_rl, name="RL Module")
print_single_results(results_policy, name="Own Policy")
print_single_results(results_hybrid, name="Hybrid")
