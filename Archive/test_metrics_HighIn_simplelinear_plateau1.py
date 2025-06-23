from pathlib import Path
import seaborn as sns

from matplotlib import pyplot as plt
import tol_colors as tc

from test.DefaultValidator import DefaultValidator
from tools.tools_rl_module import load_rl_module
import numpy as np
import pickle


from matplotlib.gridspec import GridSpec

# Set default color palette for matplotlib (use 'bright', 'muted', etc.)
tc.set_default_colors(cset="bright")
tc.Muted


def plot_arrays(arrays, labels, save_path: Path, title: str = None):
    # Prepare filename and ensure save path exists
    save_path.mkdir(parents=True, exist_ok=True)
    filename = save_path / f"{title}.pdf"

    # Plot settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig, ax = plt.subplots(figsize=(8, 6))

    for array, label in zip(arrays, labels):
        num_seeds = len(array)
        seed_indices = np.arange(num_seeds)

        ax.plot(seed_indices, array, label=rf"${label}$", linewidth=2.0, marker="o")

    ax.set_xlabel("Seed Index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Workpiece Count", fontsize=14, fontweight="bold")

    if title:
        ax.set_title(title.replace("_", " ").title(), fontsize=16, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=12, frameon=False)
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout(pad=2.0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Save high-quality vector graphic
    fig.savefig(filename, bbox_inches="tight")
    return fig


def compare_policies_scatter(arrays, labels, save_path: Path, title: str = None):
    save_path.mkdir(parents=True, exist_ok=True)
    filename = save_path / f"{title}.pdf"

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis("off")

    marker_styles = ["o", "s", "X", "D", "^", "*", "v", "P", "<", ">"]
    max_seeds = max(len(arr) for arr in arrays)

    handles = []
    labels_for_legend = []

    for idx, (array, label) in enumerate(zip(arrays, labels)):
        num_seeds = len(array)
        x_jitter_strength = 0.15
        x_offset = (idx - len(arrays) / 2) * x_jitter_strength
        seed_indices = np.arange(num_seeds) + x_offset

        marker = marker_styles[idx % len(marker_styles)]
        z = 6 if marker == "X" else 1

        scatter = ax.scatter(
            seed_indices,
            array,
            label=rf"${label}$",
            s=60,
            alpha=0.9,
            marker=marker,
            zorder=z,
        )

        handles.append(scatter)
        labels_for_legend.append(rf"${label}$")

    ax.set_xlabel("Seed", fontsize=16, fontweight="bold")
    ax.set_ylabel("Workpiece Count", fontsize=18, fontweight="bold")

    if title:
        ax.set_title(title.replace("_", " ").title(), fontsize=16, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlim(-1, max_seeds)

    legend_ax.legend(
        handles,
        labels_for_legend,
        loc="center",
        fontsize=18,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(filename, bbox_inches="tight")
    return fig


def compare_scatter_and_line_per_policy(
    primary_arrays, secondary_arrays, labels, save_path: Path, title: str = None
):
    save_path.mkdir(parents=True, exist_ok=True)
    filename = save_path / f"{title}.pdf"

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis("off")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    marker_styles = ["o", "s", "X", "D", "^", "*", "v", "P", "<", ">"]

    max_seeds = max(len(arr) for arr in primary_arrays)

    handles = []
    labels_for_legend = []

    for idx, (prim, sec, label) in enumerate(
        zip(primary_arrays, secondary_arrays, labels)
    ):
        num_seeds = len(prim)
        x = np.arange(num_seeds)

        jitter_strength = 0.15
        x_prim = x + np.random.uniform(
            -jitter_strength, jitter_strength, size=num_seeds
        )

        color = colors[idx % len(colors)]
        marker = marker_styles[idx % len(marker_styles)]

        # Scatter for primary metric
        scatter = ax.scatter(
            x_prim,
            prim,
            label=rf"{label}~$N_{{FS,CO}}$",
            color=color,
            marker=marker,
            s=60,
            alpha=0.9,
            zorder=3,
        )

        # Line for secondary metric
        (line,) = ax.plot(
            x,
            sec,
            label=rf"{label}~$N_{{out}}$",
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.9,
            zorder=2,
        )

        handles.extend([scatter, line])
        labels_for_legend.extend([rf"{label}~$N_{{FS,CO}}$", rf"{label}~$N_{{out}}$"])

    ax.set_xlabel("Seed", fontsize=16, fontweight="bold")
    ax.set_ylabel("Value", fontsize=18, fontweight="bold")

    if title:
        ax.set_title(title.replace("_", " ").title(), fontsize=16, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlim(-1, max_seeds)

    legend_ax.legend(
        handles,
        labels_for_legend,
        loc="center",
        fontsize=14,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(filename, bbox_inches="tight")
    return fig


def compare_policies_histogram(arrays, labels, save_path: Path, title: str = None):
    import matplotlib.pyplot as plt
    import numpy as np

    save_path.mkdir(parents=True, exist_ok=True)
    filename = save_path / f"{title}.pdf"
    save_path.mkdir(parents=True, exist_ok=True)

    # LaTeX and font settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define number of bins based on data range (optional: adjust manually)
    all_data = np.concatenate(arrays)
    bins = np.histogram_bin_edges(all_data, bins="auto")

    # Choose colors from matplotlib default cycle or custom set
    colors = (
        plt.cm.tab10.colors
    )  # or use e.g. tc.tol_cqual (if you're using Tol palette)

    for idx, (array, label) in enumerate(zip(arrays, labels)):
        ax.hist(
            array,
            bins=bins,
            alpha=0.6,
            label=rf"${label}$",
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Workpiece Count", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")

    if title:
        ax.set_title(title.replace("_", " ").title(), fontsize=16, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=12,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout(pad=2.0)

    fig.savefig(filename, bbox_inches="tight")
    return fig


def plot_single_histograms(arrays, labels, save_path: Path, title_prefix: str = ""):
    import matplotlib.pyplot as plt
    import numpy as np

    save_path.mkdir(parents=True, exist_ok=True)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    for array, label in zip(arrays, labels):
        fig, ax = plt.subplots(figsize=(8, 5))

        bins = np.histogram_bin_edges(array, bins="auto")

        ax.hist(
            array, bins=bins, alpha=0.75, edgecolor="black", linewidth=0.5, color="C0"
        )

        ax.set_xlabel("Workpiece Count", fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
        ax.set_title(rf"{title_prefix} ${label}$", fontsize=16, fontweight="bold")

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig.tight_layout(pad=2.0)

        # Create filename based on label (cleaned up)
        clean_label = (
            label.replace(r"{", "")
            .replace(r"}", "")
            .replace(r"\textrm", "")
            .replace(r"\text", "")
            .replace(r"~", "_")
            .replace(",", "_")
            .replace(" ", "")
            .replace("-", "")
            .replace("\\", "")
        )
        filename = save_path / f"{title_prefix}_{clean_label}.pdf"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def compare_policies_histogram_grouped(
    arrays, labels, save_path: Path, title: str = None
):
    import matplotlib.pyplot as plt
    import numpy as np

    save_path.mkdir(parents=True, exist_ok=True)
    filename = save_path / f"{title}.pdf"

    # LaTeX and font settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    fig, ax = plt.subplots(figsize=(10, 6))
    num_policies = len(arrays)

    # Determine shared bins from all data
    all_data = np.concatenate(arrays)
    bins = np.histogram_bin_edges(all_data, bins=10)  # Set number of bins manually
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_width = bins[1] - bins[0]
    bar_width = bin_width / (num_policies + 0.5)  # Less aggressive shrinking

    colors = plt.cm.tab10.colors

    for i, (array, label) in enumerate(zip(arrays, labels)):
        hist, _ = np.histogram(array, bins=bins)
        offset = (i - num_policies / 2) * bar_width + bar_width / 2
        ax.bar(
            bin_centers + offset,
            hist,
            width=bar_width,
            label=rf"${label}$",
            color=colors[i % len(colors)],
            # edgecolor='black',
            alpha=0.8,
        )

    ax.set_xlabel("Workpiece Count", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
    if title:
        ax.set_title(title.replace("_", " ").title(), fontsize=16, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=12,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout(pad=2.0)

    fig.savefig(filename, bbox_inches="tight")
    return fig


# Plotting function
def plot_all_styles(data_by_metric, metric_labels, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)

    # Plot settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Latin Modern Roman")

    for metric_idx, metric_label in enumerate(metric_labels):
        metric_data = data_by_metric[metric_idx]

        # 1. Bar plot with mean and std
        means = [np.mean(values) for _, values in metric_data]
        stds = [np.std(values) for _, values in metric_data]
        policies = [name for name, _ in metric_data]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(policies, means, yerr=stds, capsize=5)
        ax.set_ylabel(rf"${metric_label}$")
        ax.set_title(f"{metric_label} - Mean ± Std over Seeds")
        fig.tight_layout()
        fig.savefig(
            save_path
            / f"{metric_label.strip('$').replace('{','').replace('}','')}_bar.pdf"
        )
        plt.close(fig)

        # 2. Boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=[values for _, values in metric_data])
        ax.set_xticklabels(policies)
        ax.set_ylabel(rf"${metric_label}$")
        ax.set_title(f"{metric_label} - Boxplot over Seeds")
        fig.tight_layout()
        fig.savefig(
            save_path
            / f"{metric_label.strip('$').replace('{','').replace('}','')}_box.pdf"
        )
        plt.close(fig)

        # 3. Violin plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=[values for _, values in metric_data])
        ax.set_xticklabels(policies)
        ax.set_ylabel(rf"${metric_label}$")
        ax.set_title(f"{metric_label} - Violin Plot over Seeds")
        fig.tight_layout()
        fig.savefig(
            save_path
            / f"{metric_label.strip('$').replace('{','').replace('}','')}_violin.pdf"
        )
        plt.close(fig)

        # 4. Scatter plot over seeds
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (policy, values) in enumerate(metric_data):
            ax.scatter(np.arange(len(values)), values, label=policy, s=10)
        ax.set_ylabel(rf"${metric_label}$")
        ax.set_xlabel("Seed Index")
        ax.set_title(f"{metric_label} - Scatter per Seed")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            save_path
            / f"{metric_label.strip('$').replace('{','').replace('}','')}_scatter.pdf"
        )
        plt.close(fig)


# CUSTOM SETTINGS --  CHECK THIS BEFORE RUNNING
# set model folder name in interesting_runs, set evironment name, set csv names
model_folder_name = r"PLATEAU1_simple_3by3_oldarrival_run_arrival_change_201_007_simple_linear_t_2025-06-17_12-29-08"
trainstep = 350
environment_name = r"simple_3by3_oldarrival"
policy_name = "order_latest_stage"


# automated settings
custom_path_interesting_runs = (
    r"C:\Users\mimib\Desktop\Masterarbeit Produktionsmanagement\interesting_runs"
)

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


# random_policy = DefaultPolicyMaker("silly_random", environment_name).policy
# advanced_random_policy = DefaultPolicyMaker("advanced_random", environment_name).policy
# latest_stage_policy = DefaultPolicyMaker("latest_stage", environment_name).policy
# order_latest_stage_policy = DefaultPolicyMaker("order_latest_stage", environment_name).policy
#
# # test all policies
# used_seeds, results_rl = tester100.test_rl_model(rl_module=rl_mod)
# used_seeds_hybrid, results_hybrid = tester100.test_hybrid(rl_module = rl_mod, policy=latest_stage_policy)
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
fig_latest_stage = plot_arrays(
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
