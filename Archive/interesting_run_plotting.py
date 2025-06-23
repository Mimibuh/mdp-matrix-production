from pathlib import Path

from matplotlib import pyplot as plt

from test.DefaultValidator import DefaultValidator
from tools.tools_matplot import plot_selected_columns_from_csv, plot_data_per_seed, \
    plot_random_steps_rl_module, plot_random_steps_own_policy, compare_steps_rl_module_policy
from tools.tools_rl_module import load_rl_module
from test.DefaultPolicyMaker import DefaultPolicyMaker

# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
model_folder_name = r"3x3_HighIn_simple_3by3_oldarrival_run_arrival_change_201_007_transformer_t_2025-06-09_09-54-32"
trainstep = 330
environment_name = r"simple_3by3_oldarrival"
policy_name = "latest_stage"

csv_files = [r"val_mean_cor_finished.csv", r"val_mean_reward.csv", r"train_episode_mean_reward.csv"]
csv_plot_legend_labels = ["val_num_correctly_finished", "val_episode_mean_reward", "train_episode_mean_reward"]
csv_plot_titles = ["Validation: mean correctly finished workpieces", "Validation: mean episode reward", "Training: mean episode reward"]


# automated settings
custom_path_interesting_runs = r"C:\Users\mimib\Desktop\Masterarbeit Produktionsmanagement\interesting_runs"

validator100 = DefaultValidator(environment_name, test_seeds=list(range(10)))
rl_mod = load_rl_module(checkpoint_filename= Path(custom_path_interesting_runs) / model_folder_name / "rl_mod", trainstep=trainstep)
pol_mk = DefaultPolicyMaker(policy_name, environment_name)
own_policy = pol_mk.policy

used_seeds_hybrid, results_hybrid = validator100.test_hybrid(rl_module=rl_mod, policy=own_policy)
used_seeds, results_rl = validator100.test_rl_model(rl_module=rl_mod)
used_seeds_policy, results_policy = validator100.test_own_policy(policy=own_policy)

print("RL MODULE: mean validation reward: ", results_rl["rewards"].mean())
print("POLICY: mean validation reward: ", results_policy["rewards"].mean())
print("HYBRID: mean validation reward: ", results_hybrid["rewards"].mean())

print("\nRL MODULE: order and plan correct: ", results_rl["no_cor_both_finishes"].mean())
print("POLICY: order and plan correct: ", results_policy["no_cor_both_finishes"].mean())
print("\nHYBRID: order correct: ", results_hybrid["no_cor_order_finishes"].mean())

print("\nRL MODULE: order correct: ", results_rl["no_cor_order_finishes"].mean())
print("POLICY: order correct: ", results_policy["no_cor_order_finishes"].mean())
print("\nHYBRID: plan correct: ", results_hybrid["no_cor_plan_finishes"].mean())

print("\nRL MODULE: plan correct: ", results_rl["no_cor_plan_finishes"].mean())
print("POLICY: plan correct: ", results_policy["no_cor_plan_finishes"].mean())
print("\nHYBRID: generally finished: ", results_hybrid["no_total_finishes"].mean())

print("\nRL MODULE: generally finished: ", results_rl["no_total_finishes"].mean())
print("POLICY: generally finished: ", results_policy["no_total_finishes"].mean())
print("\nHYBRID: correctly finished: ", results_hybrid["no_cor_both_finishes"].mean())

#print(f"HYBRID: counts of invalid model actions: {results_hybrid['count_invalid_model_actions']}")

#validation plots
no_cor_both_combined = [results_rl["no_cor_both_finishes"], results_policy["no_cor_both_finishes"]]
fig = plot_data_per_seed(baseline=200, datas=no_cor_both_combined,
                         data_names=[f"rl_module", f"benchpolicy"],
                         used_seeds=used_seeds,
                         plot_title="Validation: number of correctly finished workpieces per seed")

no_cor_both_combined = [results_rl["no_cor_both_finishes"], results_policy["no_cor_both_finishes"]]
fig = plot_data_per_seed(baseline=200, datas=no_cor_both_combined,
                         data_names=[f"rl_module", f"benchpolicy"],
                         used_seeds=used_seeds,
                         plot_title="Validation: number of correctly finished workpieces per seed")

no_total_combined = [results_rl["no_total_finishes"], results_policy["no_total_finishes"]]
fig = plot_data_per_seed(baseline=200, datas=no_total_combined,
                            data_names=[f"rl_module", f"benchpolicy"],
                            used_seeds=used_seeds,
                            plot_title="Validation: total number of exited workpieces per seed (not necessarily correctly finished)")

no_incorrect_order_combined = [results_rl["no_total_finishes"] - results_rl["no_cor_order_finishes"], results_policy["no_total_finishes"]-results_policy["no_cor_order_finishes"]]
fig = plot_data_per_seed(baseline=200, datas=no_incorrect_order_combined,
                            data_names=[f"rl_module", f"benchpolicy"],
                            used_seeds=used_seeds,
                            plot_title="Validation: number of exited workpieces with incorrect order per seed")

no_incorrect_plan_combined = [results_rl["no_total_finishes"] - results_rl["no_cor_plan_finishes"], results_policy["no_total_finishes"]-results_policy["no_cor_plan_finishes"]]
fig = plot_data_per_seed(baseline=200, datas=no_incorrect_plan_combined,
                            data_names=[f"rl_module", f"benchpolicy"],
                            used_seeds=used_seeds,
                            plot_title="Validation: number of exited workpieces with incorrect plan per seed")


#fig = plot_data_per_seed(baseline=100, datas=results_policy["no_cor_both_finishes"], data_name="val_num_correctly_finished_random_policy", used_seeds=used_seeds, plot_title="Validation with random policy: number of correctly finished workpieces per seed")


#fig.show()

fig = plot_random_steps_rl_module(num_plot_steps=3, environment_name=environment_name, rl_module=rl_mod)


fig = plot_random_steps_own_policy(num_plot_steps=2, environment_name=environment_name, policy=own_policy, seed=0)
figs = compare_steps_rl_module_policy(rl_module=rl_mod, environment_name=environment_name, num_plot_steps=2, own_policy=own_policy, seed=1, start=0)

#fig.show()

for i in range(len(csv_files)):
    csv_path = Path(custom_path_interesting_runs) / model_folder_name / r"exported_csvs" / csv_files[i]
    fig = plot_selected_columns_from_csv(
        csv_path=csv_path,
        column_indices=[0],  # relative to non-Step columns
        legend_labels=[csv_plot_legend_labels[i]],
        title=csv_plot_titles[i]
    )
    fig.show()


plt.show()


