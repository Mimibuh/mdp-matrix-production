from tools.global_config import custom_trained_models_path
from tools.tools_analyze_test import (
    plot_arrays,
    compare_policies_scatter,
    compare_scatter_and_line_per_policy,
)
from test.DefaultValidator import DefaultValidator
from tools.tools_rl_module import load_rl_module
import pickle
from pathlib import Path

import tol_colors as tc

# Set default color palette for matplotlib (use 'bright', 'muted', etc.)
tc.set_default_colors(cset="bright")
# tc.Muted


# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
# model_folder_name = r"3x3_2_LowIn_simple_3by3_01_07_03_run_arrival_change_201_011_simple_linear_t_2025-06-18_13-08-06"
model_folder_name = r"3x3_2_LowIn_simple_3by3_01_07_03_simple_linear"

trainstep = 1210
environment_name = r"simple_3by3_01_07_03"
policy_name = "order_latest_stage"


# automated settings
custom_path_interesting_runs = custom_trained_models_path

seed_start = 100
seed_end = 200
num_seeds = str(seed_end - seed_start)

tester100 = DefaultValidator(
    environment_name, test_seeds=list(range(seed_start, seed_end, 1))
)
rl_mod = load_rl_module(
    checkpoint_filename=Path(custom_path_interesting_runs)
    / model_folder_name
    / "rl_mod",
    trainstep=trainstep,
)

save_path = Path(custom_path_interesting_runs) / model_folder_name / "test_plots"


# uncomment this for a new training


# random_policy = DefaultPolicyMaker("silly_random", environment_name).policy
# advanced_random_policy = DefaultPolicyMaker("advanced_random", environment_name).policy
# latest_stage_policy = DefaultPolicyMaker("latest_stage", environment_name).policy
# order_latest_stage_policy = DefaultPolicyMaker("order_latest_stage", environment_name).policy
#
# # test all policies
# used_seeds, results_rl, env_actions = tester100.test_rl_model_nobatch(rl_module=rl_mod)
# used_seeds_hybrid, results_hybrid = tester100.test_hybrid_nobatch(rl_module = rl_mod, policy=latest_stage_policy)
#
# used_seeds_random, results_random = tester100.test_own_policy(policy=random_policy)
# used_seeds_advanced_random, results_advanced_random = tester100.test_own_policy(policy=advanced_random_policy)
# used_seeds_latest_stage, results_latest_stage = tester100.test_own_policy(policy=latest_stage_policy)
# used_seeds_order_latest_stage, results_order_latest_stage = tester100.test_own_policy(policy=order_latest_stage_policy)
#
# # Save all results in a dict
# results_to_save = {
#     "used_seeds": used_seeds,
#     "results_rl": results_rl,
#     "used_seeds_hybrid": used_seeds_hybrid,
#     "results_hybrid": results_hybrid,
#     "used_seeds_random": used_seeds_random,
#     "results_random": results_random,
#     "used_seeds_advanced_random": used_seeds_advanced_random,
#     "results_advanced_random": results_advanced_random,
#     "used_seeds_latest_stage": used_seeds_latest_stage,
#     "results_latest_stage": results_latest_stage,
#     "used_seeds_order_latest_stage": used_seeds_order_latest_stage,
#     "results_order_latest_stage": results_order_latest_stage
# }
#
# with open(save_path / f"test_results_{num_seeds}.pkl" , "wb") as f:
#    pickle.dump(results_to_save, f)


with open(save_path / f"test_results_{num_seeds}.pkl", "rb") as f:
    loaded_results = pickle.load(f)

# Unpack them for reuse
used_seeds = loaded_results["used_seeds"]
results_rl = loaded_results["results_rl"]
used_seeds_hybrid = loaded_results["used_seeds_hybrid"]
results_hybrid = loaded_results["results_hybrid"]
used_seeds_random = loaded_results["used_seeds_random"]
results_random = loaded_results["results_random"]
used_seeds_advanced_random = loaded_results["used_seeds_advanced_random"]
results_advanced_random = loaded_results["results_advanced_random"]
used_seeds_latest_stage = loaded_results["used_seeds_latest_stage"]
results_latest_stage = loaded_results["results_latest_stage"]
used_seeds_order_latest_stage = loaded_results["used_seeds_order_latest_stage"]
results_order_latest_stage = loaded_results["results_order_latest_stage"]


# rl results
rl_Nfsco = results_rl["no_cor_both_finishes"]
rl_Ntot = results_rl["no_total_finishes"]
rl_Nco = results_rl["no_cor_order_finishes"]
rl_Nfs = results_rl["no_cor_plan_finishes"]
rl_reward = results_rl["rewards"]
# calculate other values
rl_Nnfsco = rl_Nco - rl_Nfsco
rl_Nfswo = rl_Nfs - rl_Nfsco
rl_Nnfswo = rl_Ntot - rl_Nfswo - rl_Nnfsco - rl_Nfsco
rl_Nnfs = rl_Ntot - rl_Nfs
rl_Nwo = rl_Ntot - rl_Nco
arrays = [
    rl_Nfsco,
    rl_Nfswo,
    rl_Nnfsco,
    rl_Nnfswo,
    rl_Ntot,
    rl_Nco,
    rl_Nfs,
    rl_Nwo,
    rl_Nnfs,
]
labels = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
fig = plot_arrays(arrays, labels, save_path, title="rl_module_counts_" + num_seeds)
# plt.show


# hybrid results
hybrid_Nfsco = results_hybrid["no_cor_both_finishes"]
hybrid_Ntot = results_hybrid["no_total_finishes"]
hybrid_Nco = results_hybrid["no_cor_order_finishes"]
hybrid_Nfs = results_hybrid["no_cor_plan_finishes"]
# calculate other values
hybrid_Nnfsco = hybrid_Nco - hybrid_Nfsco
hybrid_Nfswo = hybrid_Nfs - hybrid_Nfsco
hybrid_Nnfswo = hybrid_Ntot - hybrid_Nfswo - hybrid_Nnfsco - hybrid_Nfsco
hybrid_Nnfs = hybrid_Ntot - hybrid_Nfs
hybrid_Nwo = hybrid_Ntot - hybrid_Nco
arrays_hybrid = [
    hybrid_Nfsco,
    hybrid_Nfswo,
    hybrid_Nnfsco,
    hybrid_Nnfswo,
    hybrid_Ntot,
    hybrid_Nco,
    hybrid_Nfs,
    hybrid_Nwo,
    hybrid_Nnfs,
]
labels_hybrid = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
fig_hybrid = plot_arrays(
    arrays_hybrid, labels_hybrid, save_path, title="hybrid_policy_counts_" + num_seeds
)
# plt.show()


# random results
random_Nfsco = results_random["no_cor_both_finishes"]
random_Ntot = results_random["no_total_finishes"]
random_Nco = results_random["no_cor_order_finishes"]
random_Nfs = results_random["no_cor_plan_finishes"]
# calculate other values
random_Nnfsco = random_Nco - random_Nfsco
random_Nfswo = random_Nfs - random_Nfsco
random_Nnfswo = random_Ntot - random_Nfswo - random_Nnfsco - random_Nfsco
random_Nnfs = random_Ntot - random_Nfs
random_Nwo = random_Ntot - random_Nco
arrays_random = [
    random_Nfsco,
    random_Nfswo,
    random_Nnfsco,
    random_Nnfswo,
    random_Ntot,
    random_Nco,
    random_Nfs,
    random_Nwo,
    random_Nnfs,
]
labels_random = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
fig_random = plot_arrays(
    arrays_random, labels_random, save_path, title="random_policy_counts_" + num_seeds
)
# plt.show()


# advanced random results
advanced_random_Nfsco = results_advanced_random["no_cor_both_finishes"]
advanced_random_Ntot = results_advanced_random["no_total_finishes"]
advanced_random_Nco = results_advanced_random["no_cor_order_finishes"]
advanced_random_Nfs = results_advanced_random["no_cor_plan_finishes"]
# calculate other values
advanced_random_Nnfsco = advanced_random_Nco - advanced_random_Nfsco
advanced_random_Nfswo = advanced_random_Nfs - advanced_random_Nfsco
advanced_random_Nnfswo = (
    advanced_random_Ntot
    - advanced_random_Nfswo
    - advanced_random_Nnfsco
    - advanced_random_Nfsco
)
advanced_random_Nnfs = advanced_random_Ntot - advanced_random_Nfs
advanced_random_Nwo = advanced_random_Ntot - advanced_random_Nco
arrays_advanced_random = [
    advanced_random_Nfsco,
    advanced_random_Nfswo,
    advanced_random_Nnfsco,
    advanced_random_Nnfswo,
    advanced_random_Ntot,
    advanced_random_Nco,
    advanced_random_Nfs,
    advanced_random_Nwo,
    advanced_random_Nnfs,
]
labels_advanced_random = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
fig_advanced_random = plot_arrays(
    arrays_advanced_random,
    labels_advanced_random,
    save_path,
    title="advanced_random_policy_counts_" + num_seeds,
)
# plt.show()


# latest stage results
latest_stage_Nfsco = results_latest_stage["no_cor_both_finishes"]
latest_stage_Ntot = results_latest_stage["no_total_finishes"]
latest_stage_Nco = results_latest_stage["no_cor_order_finishes"]
latest_stage_Nfs = results_latest_stage["no_cor_plan_finishes"]
# calculate other values
latest_stage_Nnfsco = latest_stage_Nco - latest_stage_Nfsco
latest_stage_Nfswo = latest_stage_Nfs - latest_stage_Nfsco
latest_stage_Nnfswo = (
    latest_stage_Ntot - latest_stage_Nfswo - latest_stage_Nnfsco - latest_stage_Nfsco
)
latest_stage_Nnfs = latest_stage_Ntot - latest_stage_Nfs
latest_stage_Nwo = latest_stage_Ntot - latest_stage_Nco
arrays_latest_stage = [
    latest_stage_Nfsco,
    latest_stage_Nfswo,
    latest_stage_Nnfsco,
    latest_stage_Nnfswo,
    latest_stage_Ntot,
    latest_stage_Nco,
    latest_stage_Nfs,
    latest_stage_Nwo,
    latest_stage_Nnfs,
]
labels_latest_stage = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
fig_latest_stage = plot_arrays(
    arrays_latest_stage,
    labels_latest_stage,
    save_path,
    title="latest_stage_policy_counts_" + num_seeds,
)
# plt.show()


# order latest stage results
order_latest_stage_Nfsco = results_order_latest_stage["no_cor_both_finishes"]
order_latest_stage_Ntot = results_order_latest_stage["no_total_finishes"]
order_latest_stage_Nco = results_order_latest_stage["no_cor_order_finishes"]
order_latest_stage_Nfs = results_order_latest_stage["no_cor_plan_finishes"]
# calculate other values
order_latest_stage_Nnfsco = order_latest_stage_Nco - order_latest_stage_Nfsco
order_latest_stage_Nfswo = order_latest_stage_Nfs - order_latest_stage_Nfsco
order_latest_stage_Nnfswo = (
    order_latest_stage_Ntot
    - order_latest_stage_Nfswo
    - order_latest_stage_Nnfsco
    - order_latest_stage_Nfsco
)
order_latest_stage_Nnfs = order_latest_stage_Ntot - order_latest_stage_Nfs
order_latest_stage_Nwo = order_latest_stage_Ntot - order_latest_stage_Nco
arrays_order_latest_stage = [
    order_latest_stage_Nfsco,
    order_latest_stage_Nfswo,
    order_latest_stage_Nnfsco,
    order_latest_stage_Nnfswo,
    order_latest_stage_Ntot,
    order_latest_stage_Nco,
    order_latest_stage_Nfs,
    order_latest_stage_Nwo,
    order_latest_stage_Nnfs,
]
labels_order_latest_stage = [
    r"N_{FS, CO}",
    r"N_{FS, WO}",
    r"N_{NFS, CO}",
    r"N_{NFS, WO}",
    r"N_{out}",
    r"N_{CO}",
    r"N_{FS}",
    r"N_{WO}",
    r"N_{NFS}",
]
fig_order_latest_stage = plot_arrays(
    arrays_order_latest_stage,
    labels_order_latest_stage,
    save_path,
    title="order_latest_stage_policy_counts_" + num_seeds,
)
# plt.show()


relevant_1_arrays_noRP = [
    rl_Nfsco,
    hybrid_Nfsco,
    order_latest_stage_Nfsco,
    latest_stage_Nfsco,
    advanced_random_Nfsco,
]
relevant_1_labels_noRP = [
    r"\textrm{RL-PPO}~N_{FS,CO}",
    r"\textrm{PPO-LSP}~N_{FS,CO}",
    r"\textrm{OC-LSP}~N_{FS,CO}",
    r"\textrm{LSP}~N_{FS,CO}",
    r"\textrm{ARP}~N_{FS,CO}",
]
compare_policies_scatter(
    relevant_1_arrays_noRP,
    relevant_1_labels_noRP,
    save_path,
    title="compare_relevant_policies_Nfsco_noRP_" + num_seeds,
)
relevant_1_arrays_all = [
    rl_Nfsco,
    hybrid_Nfsco,
    order_latest_stage_Nfsco,
    latest_stage_Nfsco,
    advanced_random_Nfsco,
    random_Nfsco,
]
relevant_1_labels_all = [
    r"\textrm{RL-PPO}~N_{FS,CO}",
    r"\textrm{PPO-LSP}~N_{FS,CO}",
    r"\textrm{OC-LSP}~N_{FS,CO}",
    r"\textrm{LSP}~N_{FS,CO}",
    r"\textrm{ARP}~N_{FS,CO}",
    r"\textrm{RP}~N_{FS,CO}",
]
compare_policies_scatter(
    relevant_1_arrays_all,
    relevant_1_labels_all,
    save_path,
    title="compare_relevant_policies_Nfsco_all_" + num_seeds,
)


relevant_2_arrays_noRP = [
    rl_Nnfs,
    hybrid_Nnfs,
    order_latest_stage_Nnfs,
    latest_stage_Nnfs,
    advanced_random_Nnfs,
]
relevant_2_labels_noRP = [
    r"\textrm{RL-PPO}~N_{NFS}",
    r"\textrm{PPO-LSP}~N_{NFS}",
    r"\textrm{OC-LSP}~N_{NFS}",
    r"\textrm{LSP}~N_{NFS}",
    r"\textrm{ARP}~N_{NFS}",
]
compare_policies_scatter(
    relevant_2_arrays_noRP,
    relevant_2_labels_noRP,
    save_path,
    title="compare_relevant_policies_Nnfs_noRP_" + num_seeds,
)
relevant_2_arrays_all = [
    rl_Nnfs,
    hybrid_Nnfs,
    order_latest_stage_Nnfs,
    latest_stage_Nnfs,
    advanced_random_Nnfs,
    random_Nnfs,
]
relevant_2_labels_all = [
    r"\textrm{RL-PPO}~N_{NFS}",
    r"\textrm{PPO-LSP}~N_{NFS}",
    r"\textrm{OC-LSP}~N_{NFS}",
    r"\textrm{LSP}~N_{NFS}",
    r"\textrm{ARP}~N_{NFS}",
    r"\textrm{RP}~N_{NFS}",
]
compare_policies_scatter(
    relevant_2_arrays_all,
    relevant_2_labels_all,
    save_path,
    title="compare_relevant_policies_Nnfs_all_" + num_seeds,
)


relevant_3_arrays_noRP = [
    rl_Nwo,
    hybrid_Nwo,
    order_latest_stage_Nwo,
    latest_stage_Nwo,
    advanced_random_Nwo,
]
relevant_3_labels_noRP = [
    r"\textrm{RL-PPO}~N_{WO}",
    r"\textrm{PPO-LSP}~N_{WO}",
    r"\textrm{OC-LSP}~N_{WO}",
    r"\textrm{LSP}~N_{WO}",
    r"\textrm{ARP}~N_{WO}",
]
compare_policies_scatter(
    relevant_3_arrays_noRP,
    relevant_3_labels_noRP,
    save_path,
    title="compare_relevant_policies_Nwo_noRP_" + num_seeds,
)

relevant_3_arrays_all = [
    rl_Nwo,
    hybrid_Nwo,
    order_latest_stage_Nwo,
    latest_stage_Nwo,
    advanced_random_Nwo,
    random_Nwo,
]
relevant_3_labels_all = [
    r"\textrm{RL-PPO}~N_{WO}",
    r"\textrm{PPO-LSP}~N_{WO}",
    r"\textrm{OC-LSP}~N_{WO}",
    r"\textrm{LSP}~N_{WO}",
    r"\textrm{ARP}~N_{WO}",
    r"\textrm{RP}~N_{WO}",
]
compare_policies_scatter(
    relevant_3_arrays_all,
    relevant_3_labels_all,
    save_path,
    title="compare_relevant_policies_Nwo_all_" + num_seeds,
)


# total and fs,co
primary_arrays_noRP = [
    rl_Nfsco,
    hybrid_Nfsco,
    order_latest_stage_Nfsco,
    latest_stage_Nfsco,
    advanced_random_Nfsco,
]
secondary_arrays_noRP = [
    rl_Ntot,
    hybrid_Ntot,
    order_latest_stage_Ntot,
    latest_stage_Ntot,
    advanced_random_Ntot,
]
labels_noRP = ["RL-PPO", "PPO-LSP", "OC-LSP", "LSP", "ARP"]
compare_scatter_and_line_per_policy(
    primary_arrays_noRP,
    secondary_arrays_noRP,
    labels_noRP,
    save_path,
    title="compare_Nnfs_and_total_throughput_noRP_" + num_seeds,
)

primary_arrays_all = [
    rl_Nfsco,
    hybrid_Nfsco,
    order_latest_stage_Nfsco,
    latest_stage_Nfsco,
    advanced_random_Nfsco,
    random_Nfsco,
]
secondary_arrays_all = [
    rl_Ntot,
    hybrid_Ntot,
    order_latest_stage_Ntot,
    latest_stage_Ntot,
    advanced_random_Ntot,
    random_Ntot,
]
labels_all = ["RL-PPO", "PPO-LSP", "OC-LSP", "LSP", "ARP", "RP"]

compare_scatter_and_line_per_policy(
    primary_arrays_all,
    secondary_arrays_all,
    labels_all,
    save_path,
    title=("compare_Nnfs_and_total_throughput_all_" + num_seeds),
)


# print metrics:
print("Metrics for RL-PPO:")
print(f"Ntot: {rl_Ntot.mean():.2f} ± {rl_Ntot.std():.2f}")
print(f"Nfsco: {rl_Nfsco.mean():.2f} ± {rl_Nfsco.std():.2f}")
print(f"percentage Nfsco: {rl_Nfsco.mean() / rl_Ntot.mean() * 100:.2f}")
print(f"Nnfsco: {rl_Nnfsco.mean():.2f} ± {rl_Nnfsco.std():.2f}")
print(f"Nfswo: {rl_Nfswo.mean():.2f} ± {rl_Nfswo.std():.2f}")
print(f"Nnfswo: {rl_Nnfswo.mean():.2f} ± {rl_Nnfswo.std():.2f}")
print(f"Nnfs: {rl_Nnfs.mean():.2f} ± {rl_Nnfs.std():.2f}")
print(f"percentage Nnfs : {rl_Nnfs.mean() / rl_Ntot.mean() * 100:.2f}")
print(f"Nwo: {rl_Nwo.mean():.2f} ± {rl_Nwo.std():.2f}")
print(f"percentage Nwo: {rl_Nwo.mean() / rl_Ntot.mean() * 100:.2f}")
print(
    f"Ntot-Nfsco: {(rl_Ntot - rl_Nfsco).mean():.2f} ± {(rl_Ntot - rl_Nfsco).std():.2f}"
)
print(f"percent violations: {(rl_Ntot - rl_Nfsco).mean() / rl_Ntot.mean() * 100:.2f}")

print("\nMetrics for PPO-LSP:")
print(f"Ntot: {hybrid_Ntot.mean():.2f} ± {hybrid_Ntot.std():.2f}")
print(f"Nfsco: {hybrid_Nfsco.mean():.2f} ± {hybrid_Nfsco.std():.2f}")
print(f"percentage Nfsco: {hybrid_Nfsco.mean() / hybrid_Ntot.mean() * 100:.2f}")
print(f"Nnfsco: {hybrid_Nnfsco.mean():.2f} ± {hybrid_Nnfsco.std():.2f}")
print(f"Nfswo: {hybrid_Nfswo.mean():.2f} ± {hybrid_Nfswo.std():.2f}")
print(f"Nnfswo: {hybrid_Nnfswo.mean():.2f} ± {hybrid_Nnfswo.std():.2f}")
print(f"Nnfs: {hybrid_Nnfs.mean():.2f} ± {hybrid_Nnfs.std():.2f}")
print(f"percentage Nnfs : {hybrid_Nnfs.mean() / hybrid_Ntot.mean() * 100:.2f}")
print(f"Nwo: {hybrid_Nwo.mean():.2f} ± {hybrid_Nwo.std():.2f}")
print(f"percentage Nwo: {hybrid_Nwo.mean() / hybrid_Ntot.mean() * 100:.2f}")
print(f"Ntot-Nfsco: {(hybrid_Ntot - hybrid_Nfsco).mean():.2f}")
print(
    f"percent violations: {(hybrid_Ntot - hybrid_Nfsco).mean() / hybrid_Ntot.mean() * 100:.2f}"
)


print("\nMetrics for OC-LSP:")
print(
    f"Ntot: {order_latest_stage_Ntot.mean():.2f} ± {order_latest_stage_Ntot.std():.2f}"
)
print(
    f"Nfsco: {order_latest_stage_Nfsco.mean():.2f} ± {order_latest_stage_Nfsco.std():.2f}"
)
print(
    f"percentage Nfsco: {order_latest_stage_Nfsco.mean() / order_latest_stage_Ntot.mean() * 100:.2f}"
)
print(
    f"Nnfsco: {order_latest_stage_Nnfsco.mean():.2f} ± {order_latest_stage_Nnfsco.std():.2f}"
)
print(
    f"Nfswo: {order_latest_stage_Nfswo.mean():.2f} ± {order_latest_stage_Nfswo.std():.2f}"
)
print(
    f"Nnfswo: {order_latest_stage_Nnfswo.mean():.2f} ± {order_latest_stage_Nnfswo.std():.2f}"
)
print(
    f"Nnfs: {order_latest_stage_Nnfs.mean():.2f} ± {order_latest_stage_Nnfs.std():.2f}"
)
print(
    f"percentage Nnfs : {order_latest_stage_Nnfs.mean() / order_latest_stage_Ntot.mean() * 100:.2f}"
)
print(f"Nwo: {order_latest_stage_Nwo.mean():.2f} ± {order_latest_stage_Nwo.std():.2f}")
print(
    f"percentage Nwo: {order_latest_stage_Nwo.mean() / order_latest_stage_Ntot.mean() * 100:.2f}"
)
print(
    f"Ntot-Nfsco: {(order_latest_stage_Ntot - order_latest_stage_Nfsco).mean():.2f} ± {(order_latest_stage_Ntot - order_latest_stage_Nfsco).std():.2f}"
)
print(
    f"percent violations: {(order_latest_stage_Ntot - order_latest_stage_Nfsco).mean() / order_latest_stage_Ntot.mean() * 100:.2f}"
)

print("\nMetrics for LSP:")
print(f"Ntot: {latest_stage_Ntot.mean():.2f} ± {latest_stage_Ntot.std():.2f}")
print(f"Nfsco: {latest_stage_Nfsco.mean():.2f} ± {latest_stage_Nfsco.std():.2f}")
print(
    f"percentage Nfsco: {latest_stage_Nfsco.mean() / latest_stage_Ntot.mean() * 100:.2f}"
)
print(f"Nnfsco: {latest_stage_Nnfsco.mean():.2f} ± {latest_stage_Nnfsco.std():.2f}")
print(f"Nfswo: {latest_stage_Nfswo.mean():.2f} ± {latest_stage_Nfswo.std():.2f}")
print(f"Nnfswo: {latest_stage_Nnfswo.mean():.2f} ± {latest_stage_Nnfswo.std():.2f}")
print(f"Nnfs: {latest_stage_Nnfs.mean():.2f} ± {latest_stage_Nnfs.std():.2f}")
print(
    f"percentage Nnfs : {latest_stage_Nnfs.mean() / latest_stage_Ntot.mean() * 100:.2f}"
)
print(f"Nwo: {latest_stage_Nwo.mean():.2f} ± {latest_stage_Nwo.std():.2f}")
print(f"percentage Nwo: {latest_stage_Nwo.mean() / latest_stage_Ntot.mean() * 100:.2f}")
print(
    f"Ntot-Nfsco: {(latest_stage_Ntot - latest_stage_Nfsco).mean():.2f} ± {(latest_stage_Ntot - latest_stage_Nfsco).std():.2f}"
)
print(
    f"percent violations: {(latest_stage_Ntot - latest_stage_Nfsco).mean() / latest_stage_Ntot.mean() * 100:.2f}"
)

print("\nMetrics for ARP:")
print(f"Ntot: {advanced_random_Ntot.mean():.2f} ± {advanced_random_Ntot.std():.2f}")
print(f"Nfsco: {advanced_random_Nfsco.mean():.2f} ± {advanced_random_Nfsco.std():.2f}")
print(
    f"percentage Nfsco: {advanced_random_Nfsco.mean() / advanced_random_Ntot.mean() * 100:.2f}"
)
print(
    f"Nnfsco: {advanced_random_Nnfsco.mean():.2f} ± {advanced_random_Nnfsco.std():.2f}"
)
print(f"Nfswo: {advanced_random_Nfswo.mean():.2f} ± {advanced_random_Nfswo.std():.2f}")
print(
    f"Nnfswo: {advanced_random_Nnfswo.mean():.2f} ± {advanced_random_Nnfswo.std():.2f}"
)
print(f"Nnfs: {advanced_random_Nnfs.mean():.2f} ± {advanced_random_Nnfs.std():.2f}")
print(
    f"percentage Nnfs : {advanced_random_Nnfs.mean() / advanced_random_Ntot.mean() * 100:.2f}"
)
print(f"Nwo: {advanced_random_Nwo.mean():.2f} ± {advanced_random_Nwo.std():.2f}")
print(
    f"percentage Nwo: {advanced_random_Nwo.mean() / advanced_random_Ntot.mean() * 100:.2f}"
)
print(
    f"Ntot-Nfsco: {(advanced_random_Ntot - advanced_random_Nfsco).mean():.2f} ± {(advanced_random_Ntot - advanced_random_Nfsco).std():.2f}"
)
print(
    f"percent violations: {(advanced_random_Ntot - advanced_random_Nfsco).mean() / advanced_random_Ntot.mean() * 100:.2f}"
)

print("\nMetrics for RP:")
print(f"Ntot: {random_Ntot.mean():.2f} ± {random_Ntot.std():.2f}")
print(f"Nfsco: {random_Nfsco.mean():.2f} ± {random_Nfsco.std():.2f}")
print(f"percentage Nfsco: {random_Nfsco.mean() / random_Ntot.mean() * 100:.2f}")
print(f"Nnfsco: {random_Nnfsco.mean():.2f} ± {random_Nnfsco.std():.2f}")
print(f"Nfswo: {random_Nfswo.mean():.2f} ± {random_Nfswo.std():.2f}")
print(f"Nnfswo: {random_Nnfswo.mean():.2f} ± {random_Nnfswo.std():.2f}")
print(f"Nnfs: {random_Nnfs.mean():.2f} ± {random_Nnfs.std():.2f}")
print(f"percentage Nnfs : {random_Nnfs.mean() / random_Ntot.mean() * 100:.2f}")
print(f"Nwo: {random_Nwo.mean():.2f} ± {random_Nwo.std():.2f}")
print(f"percentage Nwo: {random_Nwo.mean() / random_Ntot.mean() * 100:.2f}")
print(
    f"Ntot-Nfsco: {(random_Ntot - random_Nfsco).mean():.2f} ± {(random_Ntot - random_Nfsco).std():.2f}"
)
print(
    f"percent violations: {(random_Ntot - random_Nfsco).mean() / random_Ntot.mean() * 100:.2f}"
)


print("number of plan correct: ", latest_stage_Nfs)
print("number of plan correct: ", order_latest_stage_Nco)


# compare_policies_histogram(relevant_1_arrays, relevant_1_labels, save_path, title="compare_relevant_policies_histo_Nfsco")
# compare_policies_histogram(relevant_2_arrays, relevant_2_labels, save_path, title="compare_relevant_policies_histo_Nnfs")
# compare_policies_histogram(relevant_3_arrays, relevant_3_labels, save_path, title="compare_relevant_policies_histo_Nwo")
#
# plot_single_histograms(relevant_1_arrays, relevant_1_labels, save_path, title_prefix="Nfsco")
# plot_single_histograms(relevant_2_arrays, relevant_2_labels, save_path, title_prefix="Nnfs" )
# plot_single_histograms(relevant_3_arrays, relevant_3_labels, save_path, title_prefix="Nwo")
#
# compare_policies_histogram_grouped(relevant_1_arrays, relevant_1_labels, save_path, title="compare_relevant_policies_grouped_Nfsco")
# compare_policies_histogram_grouped(relevant_2_arrays, relevant_2_labels, save_path, title="compare_relevant_policies_grouped_Nnfs")
# compare_policies_histogram_grouped(relevant_3_arrays, relevant_3_labels, save_path, title="compare_relevant_policies_grouped_Nwo")


# Dummy setup to illustrate the plotting structure
# Replace these with actual arrays in real usage
# Each item in `all_arrays` is a list of arrays, one for each metric per policy
# Shape: num_metrics x num_seeds
# num_seeds = 100
# num_metrics = 3
# num_policies = 5
#
# # Simulated data
# np.random.seed(0)
# dummy_data = [
#     [np.random.normal(loc=100 + 10*i + 5*j, scale=3, size=num_seeds) for j in range(num_metrics)]
#     for i in range(num_policies)
# ]
# # Labels
# policy_names = ['RL', 'Hybrid', 'Random', 'Adv. Random', 'Latest Stage']
# metric_labels = [r"$N_{FS, CO}$", r"$N_{NFS, CO}$", r"$N_{out}$"]
#
# # Organize data for plotting
# data_by_metric = [[] for _ in range(num_metrics)]
# for policy_idx, policy_data in enumerate(dummy_data):
#     for metric_idx, values in enumerate(policy_data):
#         data_by_metric[metric_idx].append((policy_names[policy_idx], values))
#
#
#
#
#
#
# # Run the plotting (replace dummy data with your actual collected structure)
# compare_Nfsco = [rl_Nfsco, hybrid_Nfsco, random_Nfsco, advanced_random_Nfsco, latest_stage_Nfsco]
# compare_Nfsco_titles = [r"N_{FS, CO}~~\textrm{RL-PPO}",
#                         r"N_{FS, CO}~~\textrm{PPO-LSP}",
#                         r"N_{FS, CO}~~\textrm{RP}",
#                         r"N_{FS, CO}~~\textrm{ARP}",
#                         r"N_{FS, CO}~~\textrm{LSP}"]
#
#
# data_by_metric = [
#     [("RL", rl_Nfsco),
#      ("Hybrid", hybrid_Nfsco),
#      ("Random", random_Nfsco),
#      ("Adv. Random", advanced_random_Nfsco),
#      ("Latest Stage", latest_stage_Nfsco)],
#
#     [("RL", rl_Ntot),
#      ("Hybrid", hybrid_Ntot),
#      ("Random", random_Ntot),
#      ("Adv. Random", advanced_random_Ntot),
#      ("Latest Stage", latest_stage_Ntot)]
# ]
#
# metric_labels = [
#     r"N_{FS, CO}", r"N_{tot}"]
#
#
#
#
#
# plot_all_styles(data_by_metric, metric_labels, save_path)
