from pathlib import Path


from test.DefaultValidator import DefaultValidator
from tools.tools_rl_module import load_rl_module
from test.DefaultPolicyMaker import DefaultPolicyMaker
import numpy as np

# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
model_folder_name = r"3x3HighIn_simple_3by3_oldarrival_run_arrival_change_201_007_simple_linear_t_2025-06-15_18-52-58"
trainstep = 330
environment_name = r"simple_3by3_oldarrival"
compare_policy_name = "order_latest_stage"
hybrid_policy_name = "latest_stage"


# csv_files = [r"val_mean_cor_finished.csv", r"val_mean_reward.csv", r"train_episode_mean_reward.csv"]
##csv_plot_legend_labels = ["val_num_correctly_finished", "val_episode_mean_reward", "train_episode_mean_reward"]
# csv_plot_titles = ["Validation: mean correctly finished workpieces", "Validation: mean episode reward", "Training: mean episode reward"]


# automated settings
custom_path_interesting_runs = (
    r"C:\Users\mimib\Desktop\Masterarbeit Produktionsmanagement\interesting_runs"
)

validator100 = DefaultValidator(environment_name, test_seeds=list(range(100)))
rl_mod = load_rl_module(
    checkpoint_filename=Path(custom_path_interesting_runs)
    / model_folder_name
    / "rl_mod",
    trainstep=trainstep,
)
pol_mk = DefaultPolicyMaker(compare_policy_name, environment_name)
own_policy = pol_mk.policy

compare_policy = DefaultPolicyMaker(compare_policy_name, environment_name).policy
hybrid_policy = DefaultPolicyMaker(hybrid_policy_name, environment_name).policy

used_seeds_hybrid, results_hybrid = validator100.test_hybrid(
    rl_module=rl_mod, policy=hybrid_policy
)
used_seeds, results_rl = validator100.test_rl_model(rl_module=rl_mod)
used_seeds_policy, results_policy = validator100.test_own_policy(policy=compare_policy)

print("RL MODULE: mean validation reward: ", results_rl["rewards"].mean())
print("POLICY: mean validation reward: ", results_policy["rewards"].mean())
print("HYBRID: mean validation reward: ", results_hybrid["rewards"].mean())

print(
    "\nRL MODULE: order and plan correct: ", results_rl["no_cor_both_finishes"].mean()
)
print("POLICY: order and plan correct: ", results_policy["no_cor_both_finishes"].mean())
print("HYBRID: order and plan correct: ", results_hybrid["no_cor_both_finishes"].mean())

print("\nRL MODULE: order correct: ", results_rl["no_cor_order_finishes"].mean())
print("POLICY: order correct: ", results_policy["no_cor_order_finishes"].mean())
print("HYBRID: order correct: ", results_hybrid["no_cor_order_finishes"].mean())

print("\nRL MODULE: plan correct: ", results_rl["no_cor_plan_finishes"].mean())
print("POLICY: plan correct: ", results_policy["no_cor_plan_finishes"].mean())
print("HYRBID: plan correct: ", results_hybrid["no_cor_plan_finishes"].mean())

print("\nRL MODULE: generally finished: ", results_rl["no_total_finishes"].mean())
print("POLICY: generally finished: ", results_policy["no_total_finishes"].mean())
print("HYBRID: generally finished: ", results_hybrid["no_total_finishes"].mean())


# print(f"HYBRID: counts of invalid model actions: {results_hybrid['count_invalid_model_actions']}")

# Element-wise percentage change from A to B
percent_changes = (
    (results_rl["no_cor_both_finishes"] - results_policy["no_cor_both_finishes"])
    / results_policy["no_cor_both_finishes"]
    * 100
)
# Mean percentage change across seeds
mean_percent_change = np.mean(percent_changes)
print(
    f"\nMean percentage change in correctly finished workpieces from policy to RL module: {mean_percent_change:.4f}%"
)

# percentage of wrong pieces policy
print("\nPolicy:")
mean_correct_finished_policy = np.mean(results_policy["no_cor_both_finishes"])
print(
    f"Mean number of correctly finished workpieces: {mean_correct_finished_policy:.4f}"
)
# order wrong
order_wrong_policy = (
    results_policy["no_total_finishes"] - results_policy["no_cor_order_finishes"]
)
order_wrong_percentage_policy = (
    order_wrong_policy / results_policy["no_total_finishes"] * 100
)
mean_order_wrong_percentage_policy = np.mean(order_wrong_percentage_policy)
mean_order_wrong_policy = np.mean(order_wrong_policy)
print(f"Mean number of order wrong pieces: {mean_order_wrong_policy:.4f}")
print(
    f"Mean percentage of order wrong pieces: {mean_order_wrong_percentage_policy:.4f}%"
)
# plan wrong
plan_wrong_policy = (
    results_policy["no_total_finishes"] - results_policy["no_cor_plan_finishes"]
)
plan_wrong_percentage_policy = (
    plan_wrong_policy / results_policy["no_total_finishes"] * 100
)
mean_plan_wrong_percentage_policy = np.mean(plan_wrong_percentage_policy)
mean_plan_wrong_policy = np.mean(plan_wrong_policy)
print(f"Mean number of plan wrong pieces: {mean_plan_wrong_policy:.4f}")
print(f"Mean percentage of plan wrong pieces: {mean_plan_wrong_percentage_policy:.4f}%")

# percentage of wrong pieces rl module
print("\nRl module:")
mean_correct_finished_rl = np.mean(results_rl["no_cor_both_finishes"])
print(f"Mean number of correctly finished workpieces: {mean_correct_finished_rl:.4f}")
# order wrong
order_wrong_rl = results_rl["no_total_finishes"] - results_rl["no_cor_order_finishes"]
order_wrong_percentage_rl = order_wrong_rl / results_rl["no_total_finishes"] * 100
mean_order_wrong_percentage_rl = np.mean(order_wrong_percentage_rl)
mean_order_wrong_rl = np.mean(order_wrong_rl)
print(f"Mean number of order wrong pieces: {mean_order_wrong_rl:.4f}")
print(f"Mean percentage of order wrong pieces: {mean_order_wrong_percentage_rl:.4f}%")
# plan wrong
plan_wrong_rl = results_rl["no_total_finishes"] - results_rl["no_cor_plan_finishes"]
print(f"plan wrong: {plan_wrong_rl}")
print(f"no total finishes: {results_rl['no_total_finishes']}")
print(f"count total plan wrong: {np.sum(plan_wrong_rl)}")
plan_wrong_percentage_rl = plan_wrong_rl / results_rl["no_total_finishes"] * 100
print(f"plan wrong percentage: {plan_wrong_percentage_rl}")
mean_plan_wrong_percentage_rl = np.mean(plan_wrong_percentage_rl)
mean_plan_wrong_rl = np.mean(plan_wrong_rl)
print(f"Mean number of plan wrong pieces: {mean_plan_wrong_rl:.4f}")
print(f"Mean percentage of plan wrong pieces: {mean_plan_wrong_percentage_rl:.4f}%")
