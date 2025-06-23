import numpy as np


def safe_mean(key, results):
    if key in results:
        return np.mean(results[key])
    else:
        print(f"Warning: Key '{key}' not found in results.")
        return None


def print_single_results(results, name="Model"):
    """
    Prints evaluation results for a single model.
    Checks if each value exists before printing.

    Args:
        results (dict): dictionary with evaluation results
        name (str): name of the model (for nice printing)
    """

    print(f"\n\n{name}:")

    # Print mean reward
    mean_reward = safe_mean("rewards", results)
    if mean_reward is not None:
        print(f"Mean validation reward: {mean_reward:.4f}")

    # Correct finishes
    mean_both_correct = safe_mean("no_cor_both_finishes", results)
    if mean_both_correct is not None:
        print(f"Mean Order and plan correct: {mean_both_correct:.4f}")

    mean_order_correct = safe_mean("no_cor_order_finishes", results)
    if mean_order_correct is not None:
        print(f"Mean Order correct: {mean_order_correct:.4f}")

    mean_plan_correct = safe_mean("no_cor_plan_finishes", results)
    if mean_plan_correct is not None:
        print(f"Mean Plan correct: {mean_plan_correct:.4f}")

    mean_total_finishes = safe_mean("no_total_finishes", results)
    if mean_total_finishes is not None:
        print(f"Mean Generally finished: {mean_total_finishes:.4f}", results)

    # More detailed error analysis if possible
    if "no_total_finishes" in results and "no_cor_order_finishes" in results:
        order_wrong = results["no_total_finishes"] - results["no_cor_order_finishes"]
        order_wrong_percentage = order_wrong / results["no_total_finishes"] * 100
        mean_order_wrong = np.mean(order_wrong)
        mean_order_wrong_percentage = np.mean(order_wrong_percentage)
        print(f"\nMean number of order wrong pieces: {mean_order_wrong:.4f}")
        print(
            f"Mean percentage of order wrong pieces: {mean_order_wrong_percentage:.4f}%"
        )

    if "no_total_finishes" in results and "no_cor_plan_finishes" in results:
        plan_wrong = results["no_total_finishes"] - results["no_cor_plan_finishes"]
        plan_wrong_percentage = plan_wrong / results["no_total_finishes"] * 100
        mean_plan_wrong = np.mean(plan_wrong)
        mean_plan_wrong_percentage = np.mean(plan_wrong_percentage)
        print(f"\nMean number of plan wrong pieces: {mean_plan_wrong:.4f}")
        print(
            f"Mean percentage of plan wrong pieces: {mean_plan_wrong_percentage:.4f}%"
        )

    mean_counts = safe_mean("counts_invalid_model_actions", results)
    if mean_counts is not None:
        print(f"Mean invalid counts: {mean_counts:.4f}")
