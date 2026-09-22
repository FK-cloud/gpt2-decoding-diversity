# the bar charts, effect-size plot, and boxplot from the CSVs


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from config import RESULTS_DIR, ANALYSIS_LENGTHS

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'

BLUE = "#2E5FA1"
ORANGE = "#E07B2E"
LENGTHS = ANALYSIS_LENGTHS


def boot_ci(vals, rng, n_boot=2000):
    # quick bootstrap for a strategy's own mean, just for the error bars on the bar charts
    m = vals.mean()
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return m, m - lo, hi - m


def make_bar_chart(metrics, metric, fname, ylabel, rng):
    g_means, g_errlo, g_errhi = [], [], []
    t_means, t_errlo, t_errhi = [], [], []
    for L in LENGTHS:
        g = metrics[(metrics.strategy == "greedy") & (metrics.length == L)][metric].to_numpy()
        t = metrics[(metrics.strategy == "top_p_avg") & (metrics.length == L)][metric].to_numpy()
        m, lo, hi = boot_ci(g, rng)
        g_means.append(m); g_errlo.append(lo); g_errhi.append(hi)
        m, lo, hi = boot_ci(t, rng)
        t_means.append(m); t_errlo.append(lo); t_errhi.append(hi)

    x = np.arange(len(LENGTHS))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.bar(x - w / 2, g_means, width=w, yerr=[g_errlo, g_errhi], capsize=5,
           color=BLUE, label="Greedy", edgecolor="white")
    ax.bar(x + w / 2, t_means, width=w, yerr=[t_errlo, t_errhi], capsize=5,
           color=ORANGE, label="Top-p (p=0.9)", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L} tokens" for L in LENGTHS])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}\nby decoding strategy and length", fontsize=14)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=220)
    plt.close(fig)
    print(f"saved {fname}")


def make_effect_size_plot(bootstrap_main):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, ylabel in [
        (axes[0], "rep3", "\u0394 Rep-3 (Top-p \u2212 Greedy)"),
        (axes[1], "distinct2", "\u0394 Distinct-2 (Top-p \u2212 Greedy)"),
    ]:
        sub = bootstrap_main[bootstrap_main.metric == metric].sort_values("length")
        x = sub.length.to_numpy()
        y = sub.diff_topp_minus_greedy.to_numpy()
        lo = y - sub.ci_lower_95.to_numpy()
        hi = sub.ci_upper_95.to_numpy() - y
        color = BLUE if metric == "rep3" else ORANGE
        ax.errorbar(x, y, yerr=[lo, hi], marker='o', markersize=9, capsize=6, color=color, linewidth=2)
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        ax.set_xlabel("prefix length analyzed (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(LENGTHS)
        ax.set_title(ylabel.replace("\u0394 ", "effect size: "))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle("Decoding effect grows with generation length (95% paired bootstrap CI)", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "effect_size_growth.png"), dpi=220)
    plt.close(fig)
    print("saved effect_size_growth.png")


def make_boxplot(metrics, rng):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, ylabel in [
        (axes[0], "rep3", "Rep-3 (repetition rate)"),
        (axes[1], "distinct2", "Distinct-2 (lexical diversity)"),
    ]:
        data, labels = [], []
        for strategy, label in [("greedy", "Greedy"), ("top_p_avg", "Top-p")]:
            vals = metrics[(metrics.strategy == strategy) & (metrics.length == 200)][metric].to_numpy()
            data.append(vals)
            labels.append(label)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], [BLUE, ORANGE]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for i, vals in enumerate(data):
            jitter = rng.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals, color='black', alpha=0.5, s=18, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\nacross the 20 prompts, at 200 tokens")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "boxplot_distribution_200tok.png"), dpi=220)
    plt.close(fig)
    print("saved boxplot_distribution_200tok.png")


def main():
    bootstrap_main = pd.read_csv(os.path.join(RESULTS_DIR, "bootstrap_main.csv"))
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_per_prompt.csv"))
    rng = np.random.default_rng(2026)

    make_bar_chart(metrics, "rep3", "bar_rep3.png", "Rep-3 (repetition rate)", rng)
    make_bar_chart(metrics, "distinct2", "bar_distinct2.png", "Distinct-2 (lexical diversity)", rng)
    make_effect_size_plot(bootstrap_main)
    make_boxplot(metrics, rng)

    print("\ndone, results/ is updated")


if __name__ == "__main__":
    main()
