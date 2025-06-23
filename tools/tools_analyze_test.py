import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

"""
plotting tools for test metrics
"""

import tol_colors as tc

# Set default color palette for matplotlib (use 'bright', 'muted', etc.)
tc.set_default_colors(cset="bright")


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
        seed_indices = np.arange(num_seeds) + x_offset + 100

        marker = marker_styles[idx % len(marker_styles)]
        z = 6 if marker == "X" else 1

        scatter = ax.scatter(
            seed_indices,
            array,
            label=rf"${label}$",
            s=30,
            alpha=0.9,
            marker=marker,
            zorder=z,
        )

        handles.append(scatter)
        labels_for_legend.append(rf"${label}$")

    ax.set_xlabel("Seed", fontsize=16, fontweight="bold")
    ax.set_ylabel("Workpiece count", fontsize=18, fontweight="bold")

    if title:
        ax.set_title(title.replace("_", " ").title(), fontsize=16, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlim(99, 100 + max_seeds)

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
        x = np.arange(num_seeds) + 100

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
            s=30,
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
    ax.set_ylabel("Workpiece count", fontsize=18, fontweight="bold")

    if title:
        ax.set_title(title.replace("_", " ").title(), fontsize=16, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlim(99, 100 + max_seeds)

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
