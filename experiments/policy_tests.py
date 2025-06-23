from test.DefaultPolicyMaker import DefaultPolicyMaker
from test.DefaultValidator import DefaultValidator


# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
environment_name = r"simple_3by3"
policy_name = "latest_stage"

validator100 = DefaultValidator(environment_name, test_seeds=list(range(100)))

validator100.test_policy_determinstic_behavior(
    DefaultPolicyMaker("latest_stage", environment_name).policy
)
