from matplotlib import pyplot as plt

from test.DefaultValidator import DefaultValidator
from tools.tools_matplot import plot_data_per_seed
from test.DefaultPolicyMaker import DefaultPolicyMaker

# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
environment_name = r"simple_3by3_01_07_03"
policy_name = "order_latest_stage"


validator100 = DefaultValidator(environment_name, test_seeds=list(range(100)))
pol_mk = DefaultPolicyMaker(policy_name, environment_name)
own_policy = pol_mk.policy

used_seeds_policy, results_policy = validator100.test_own_policy(policy=own_policy)

print("POLICY: mean validation reward: ", results_policy["rewards"].mean())
print("POLICY: order and plan correct: ", results_policy["no_cor_both_finishes"].mean())
print("POLICY: order correct: ", results_policy["no_cor_order_finishes"].mean())
print("POLICY: plan correct: ", results_policy["no_cor_plan_finishes"].mean())
print("POLICY: generally finished: ", results_policy["no_total_finishes"].mean())

# validation plots
fig = plot_data_per_seed(
    baseline=100,
    datas=[results_policy["no_cor_both_finishes"]],
    data_names=["val_num_correctly_finished_random_policy"],
    used_seeds=used_seeds_policy,
    plot_title=f"Validation with policy {policy_name}: number of correctly finished workpieces per seed",
)
# fig.show()

# fig = plot_random_steps_own_policy(
#   num_plot_steps=2, environment_name=environment_name, policy=own_policy
# )

# fig.show()

plt.show()
