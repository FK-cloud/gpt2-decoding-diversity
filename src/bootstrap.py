"""
Paired bootstrap resampling utilities (resampling is always done over
the 20 PROMPTS, preserving the greedy/top-p pairing within each prompt).
"""

import numpy as np


def paired_bootstrap_diff(greedy_vals, topp_vals, n_boot, rng, ci=0.95):
    """greedy_vals, topp_vals: same length, aligned by prompt.
    Returns: point estimate, (lo, hi) CI, and raw per-prompt diffs."""
    n = len(greedy_vals)
    diffs = topp_vals - greedy_vals
    point = diffs.mean()

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)   # resample prompts with replacement
        boot_means[b] = diffs[idx].mean()

    alpha = 1 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, diffs


def bootstrap_effect_change(diffs_a, diffs_b, n_boot, rng, ci=0.95):
    """Tests whether the effect (top_p - greedy) changes between two
    conditions, e.g. length=50 vs length=200. diffs_a/diffs_b are the
    per-prompt paired differences at each condition, aligned by prompt.
    Returns: point estimate of (mean(diffs_b) - mean(diffs_a)), lo, hi."""
    n = len(diffs_a)
    assert len(diffs_a) == len(diffs_b)
    point = diffs_b.mean() - diffs_a.mean()

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diffs_b[idx].mean() - diffs_a[idx].mean()

    alpha = 1 - ci
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi
