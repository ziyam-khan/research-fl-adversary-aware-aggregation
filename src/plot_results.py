import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "run3-cifar10-30rounds")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

ATTACK_LABELS = {
    "no_attack": "No Attack",
    "random_30": "Random Dropout (30%)",
    "strategic_gradient": "Strategic-Gradient",
    "strategic_distribution": "Strategic-Distribution",
    "combined_strategic": "Combined Strategic",
}

AGG_LABELS = {
    "fedavg": "FedAvg",
    "median": "Coord. Median",
    "trimmed_mean": "Trimmed Mean",
    "aawa": "AAWA (Ours)",
}

AGG_STYLES = {
    "fedavg": {"color": "#d62728", "linestyle": "--", "linewidth": 1.8, "marker": "s", "markersize": 4},
    "median": {"color": "#1f77b4", "linestyle": "-.", "linewidth": 1.8, "marker": "^", "markersize": 4},
    "trimmed_mean": {"color": "#ff7f0e", "linestyle": ":", "linewidth": 1.8, "marker": "D", "markersize": 4},
    "aawa": {"color": "#2ca02c", "linestyle": "-", "linewidth": 2.5, "marker": "o", "markersize": 5},
}


def smooth(values, window=3):
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def get_available(results):
    all_attacks = ["no_attack", "random_30", "strategic_gradient",
                   "strategic_distribution", "combined_strategic"]
    attacks = [a for a in all_attacks if a in results]
    all_aggs = ["fedavg", "median", "trimmed_mean", "aawa"]
    aggs = [a for a in all_aggs if a in results.get(attacks[0], {})]
    return attacks, aggs


def plot_accuracy_curves(results):
    attacks_to_plot = [a for a in ["random_30", "combined_strategic"] if a in results]
    _, aggs = get_available(results)

    fig, axes = plt.subplots(1, len(attacks_to_plot), figsize=(5.5 * len(attacks_to_plot), 4.5))
    if len(attacks_to_plot) == 1:
        axes = [axes]

    for idx, atk in enumerate(attacks_to_plot):
        ax = axes[idx]
        for agg in aggs:
            accs = np.array(results[atk][agg]) * 100
            smoothed = smooth(accs)
            style = {k: v for k, v in AGG_STYLES[agg].items() if k != "marker" and k != "markersize"}
            ax.plot(range(len(smoothed)), smoothed, label=AGG_LABELS[agg], **style)
            ax.plot(range(0, len(smoothed), 5), smoothed[::5], marker=AGG_STYLES[agg]["marker"],
                    markersize=AGG_STYLES[agg]["markersize"], color=AGG_STYLES[agg]["color"], linestyle="none")
        ax.set_title(ATTACK_LABELS[atk])
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_ylim(10, 55)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "accuracy_curves")
    plt.savefig(path + ".pdf", bbox_inches="tight", dpi=300)
    plt.savefig(path + ".png", bbox_inches="tight", dpi=300)
    print(f"Saved: {path}.pdf")
    plt.close()


def plot_final_accuracy_bar(results):
    attacks, aggs = get_available(results)
    colors = {
        "fedavg": "#d62728", "median": "#1f77b4",
        "trimmed_mean": "#ff7f0e", "aawa": "#2ca02c"
    }
    markers = {
        "fedavg": "s", "median": "^",
        "trimmed_mean": "D", "aawa": "o"
    }

    final_accs = {}
    for agg in aggs:
        final_accs[agg] = [results[atk][agg][-1] * 100 for atk in attacks]

    y = np.arange(len(attacks))
    fig, ax = plt.subplots(figsize=(8, 3.5))

    for atk_idx in range(len(attacks)):
        ax.axhline(y=atk_idx, color="#e0e0e0", linewidth=0.8, zorder=0)

    label_offsets = {agg: [] for agg in aggs}
    for atk_idx in range(len(attacks)):
        vals = [(agg, final_accs[agg][atk_idx]) for agg in aggs]
        vals.sort(key=lambda x: x[1])
        for rank, (agg, val) in enumerate(vals):
            close = [v for a, v in vals if a != agg and abs(v - val) < 2.0]
            if close:
                above = sum(1 for a, v in vals[:rank] if abs(v - val) < 2.0)
                y_off = 8 + above * 12
            else:
                y_off = 8
            label_offsets[agg].append(y_off)

    for agg in aggs:
        ax.scatter(final_accs[agg], y, color=colors[agg], marker=markers[agg],
                   s=80, label=AGG_LABELS[agg], zorder=3, edgecolors="white", linewidths=0.5)
        for atk_idx, val in enumerate(final_accs[agg]):
            ax.annotate(f"{val:.1f}", (val, atk_idx), textcoords="offset points",
                        xytext=(0, label_offsets[agg][atk_idx]), ha="center", fontsize=7,
                        color=colors[agg])

    ax.set_xlabel("Final Test Accuracy (%)")
    ax.set_yticks(y)
    ax.set_yticklabels([ATTACK_LABELS[a] for a in attacks], fontsize=9)
    all_vals = [v for agg in aggs for v in final_accs[agg]]
    ax.set_xlim(min(all_vals) - 5, max(all_vals) + 5)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "final_accuracy_bar")
    plt.savefig(path + ".pdf", bbox_inches="tight", dpi=300)
    plt.savefig(path + ".png", bbox_inches="tight", dpi=300)
    print(f"Saved: {path}.pdf")
    plt.close()


def plot_suppression_sweep(results_sweep=None):
    sweep_path = os.path.join(RESULTS_DIR, "sweep_results.json")
    if not os.path.exists(sweep_path):
        print("No sweep results found, skipping.")
        return

    with open(sweep_path) as f:
        sweep = json.load(f)

    available_aggs = [a for a in ["fedavg", "median", "trimmed_mean", "aawa"] if a in sweep]

    fig, ax = plt.subplots(figsize=(7, 5))
    for agg in available_aggs:
        fracs = sorted(sweep[agg].keys(), key=float)
        accs = [sweep[agg][f] * 100 for f in fracs]
        frac_vals = [float(f) * 100 for f in fracs]
        ax.plot(frac_vals, accs, label=AGG_LABELS[agg], **AGG_STYLES[agg])

    ax.set_xlabel("Suppression Fraction (%)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Accuracy vs. Strategic Suppression Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(20, 50)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "suppression_sweep")
    plt.savefig(path + ".pdf", bbox_inches="tight", dpi=300)
    plt.savefig(path + ".png", bbox_inches="tight", dpi=300)
    print(f"Saved: {path}.pdf")
    plt.close()


def print_summary_table(results):
    attacks, aggs = get_available(results)

    print("\n=== Final Accuracy Table (LaTeX) ===")
    header = "Attack & " + " & ".join(AGG_LABELS[a] for a in aggs) + " \\\\"
    print(header)
    print("\\hline")
    for atk in attacks:
        row = ATTACK_LABELS[atk]
        vals = [results[atk][agg][-1] * 100 for agg in aggs]
        best = max(vals)
        for val in vals:
            if val == best:
                row += f" & \\textbf{{{val:.2f}}}"
            else:
                row += f" & {val:.2f}"
        row += " \\\\"
        print(row)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path) as f:
        results = json.load(f)

    plot_accuracy_curves(results)
    plot_final_accuracy_bar(results)
    plot_suppression_sweep()
    print_summary_table(results)


if __name__ == "__main__":
    main()
