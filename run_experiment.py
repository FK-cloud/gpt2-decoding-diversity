# generates 80 continuations (20 prompts x 1 greedy + 3 top-p runs),
# scores Rep-3 and Distinct-2 at 50/100/200 tokens, averages the top-p
# seeds per prompt, then does the paired bootstrap and the length-effect
# test. everything gets saved to results/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    ANALYSIS_LENGTHS, TOPP_SEEDS, N_BOOTSTRAP, CI,
    RNG_SEED_BOOTSTRAP, RESULTS_DIR,
)
from data.prompts import flat_prompts
from src.generation import generate
from src.metrics import rep_n, distinct_n
from src.bootstrap import paired_bootstrap_diff, bootstrap_effect_change


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED_BOOTSTRAP)
    alpha = 1 - CI

    prompts = flat_prompts()
    assert len(prompts) == 20

    # generate: 20 prompts x (1 greedy + 3 top-p seeds) = 80 runs
    gen_rows = []
    sample_outputs = []
    total = len(prompts) * (1 + len(TOPP_SEEDS))
    run_i = 0

    for item in prompts:
        idx, theme, prompt = item["global_idx"], item["theme"], item["prompt"]

        run_i += 1
        print(f"[{run_i}/{total}] greedy | prompt {idx} ({theme})")
        cont_ids, cont_text = generate(prompt, "greedy")
        gen_rows.append({
            "global_idx": idx, "theme": theme, "prompt": prompt,
            "strategy": "greedy", "seed": None,
            "token_ids": cont_ids, "text": cont_text,
        })
        if idx < 3:
            sample_outputs.append((prompt, "greedy", None, cont_text))

        for seed in TOPP_SEEDS:
            run_i += 1
            print(f"[{run_i}/{total}] top_p (seed={seed}) | prompt {idx} ({theme})")
            cont_ids, cont_text = generate(prompt, "top_p", seed=seed)
            gen_rows.append({
                "global_idx": idx, "theme": theme, "prompt": prompt,
                "strategy": "top_p", "seed": seed,
                "token_ids": cont_ids, "text": cont_text,
            })
            if idx < 3 and seed == TOPP_SEEDS[0]:
                sample_outputs.append((prompt, "top_p", seed, cont_text))

    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(os.path.join(RESULTS_DIR, "generations_raw.csv"), index=False)
    print(f"\nSaved {RESULTS_DIR}/generations_raw.csv")

    # now compute Rep-3 / Distinct-2 at each prefix length
    metric_rows = []
    for row in gen_rows:
        for L in ANALYSIS_LENGTHS:
            prefix = row["token_ids"][:L]
            metric_rows.append({
                "global_idx": row["global_idx"], "theme": row["theme"],
                "strategy": row["strategy"], "seed": row["seed"], "length": L,
                "rep3": rep_n(prefix, 3),
                "distinct2": distinct_n(prefix, 2),
            })
    metrics_df = pd.DataFrame(metric_rows)

    # --- Average top-p's 3 seeds at the PROMPT level ---
    topp_avg = (
        metrics_df[metrics_df.strategy == "top_p"]
        .groupby(["global_idx", "theme", "length"])[["rep3", "distinct2"]]
        .mean()
        .reset_index()
    )
    topp_avg["strategy"] = "top_p_avg"

    greedy_only = metrics_df[metrics_df.strategy == "greedy"][
        ["global_idx", "theme", "length", "strategy", "rep3", "distinct2"]
    ]

    metrics_per_prompt = pd.concat([greedy_only, topp_avg], ignore_index=True)
    metrics_per_prompt.to_csv(os.path.join(RESULTS_DIR, "metrics_per_prompt.csv"), index=False)
    print(f"Saved {RESULTS_DIR}/metrics_per_prompt.csv")

    # paired bootstrap: greedy vs top_p_avg, per length, per metric
    bootstrap_rows = []
    diffs_by_length = {}

    for L in ANALYSIS_LENGTHS:
        for metric in ["rep3", "distinct2"]:
            g = (
                metrics_per_prompt[(metrics_per_prompt.strategy == "greedy") & (metrics_per_prompt.length == L)]
                .sort_values("global_idx")[metric].to_numpy()
            )
            t = (
                metrics_per_prompt[(metrics_per_prompt.strategy == "top_p_avg") & (metrics_per_prompt.length == L)]
                .sort_values("global_idx")[metric].to_numpy()
            )
            point, lo, hi, diffs = paired_bootstrap_diff(g, t, N_BOOTSTRAP, rng, CI)
            bootstrap_rows.append({
                "length": L, "metric": metric,
                "greedy_mean": g.mean(), "top_p_mean": t.mean(),
                "diff_topp_minus_greedy": point,
                "ci_lower_95": lo, "ci_upper_95": hi,
                "significant": bool((lo > 0) or (hi < 0)),
            })
            diffs_by_length[(L, metric)] = diffs

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(os.path.join(RESULTS_DIR, "bootstrap_main.csv"), index=False)
    print(f"\nSaved {RESULTS_DIR}/bootstrap_main.csv")
    print(bootstrap_df)

    # does the decoding effect actually change from 50 -> 200 tokens?
    length_effect_rows = []
    for metric in ["rep3", "distinct2"]:
        diffs_50 = diffs_by_length[(50, metric)]
        diffs_200 = diffs_by_length[(200, metric)]
        point, lo, hi = bootstrap_effect_change(diffs_50, diffs_200, N_BOOTSTRAP, rng, CI)
        length_effect_rows.append({
            "metric": metric,
            "effect_at_50": diffs_50.mean(),
            "effect_at_200": diffs_200.mean(),
            "effect_change_200_minus_50": point,
            "ci_lower_95": lo, "ci_upper_95": hi,
            "effect_grows_with_length": bool((lo > 0) or (hi < 0)),
        })

    length_effect_df = pd.DataFrame(length_effect_rows)
    length_effect_df.to_csv(os.path.join(RESULTS_DIR, "bootstrap_length_effect.csv"), index=False)
    print(f"\nSaved {RESULTS_DIR}/bootstrap_length_effect.csv")
    print(length_effect_df)

    # line plots for the two metrics
    for metric, fname, ylabel in [
        ("rep3", "plot_rep3.png", "Rep-3 (repetition rate)"),
        ("distinct2", "plot_distinct2.png", "Distinct-2 (lexical diversity)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for strategy, color, marker in [("greedy", "tab:blue", "o"), ("top_p_avg", "tab:orange", "s")]:
            means, err_lo, err_hi = [], [], []
            for L in ANALYSIS_LENGTHS:
                vals = metrics_per_prompt[
                    (metrics_per_prompt.strategy == strategy) & (metrics_per_prompt.length == L)
                ][metric].to_numpy()
                m = vals.mean()
                boot = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(1000)])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                means.append(m); err_lo.append(m - lo); err_hi.append(hi - m)
            label = "Top-p (avg of 3 seeds)" if strategy == "top_p_avg" else "Greedy"
            ax.errorbar(ANALYSIS_LENGTHS, means, yerr=[err_lo, err_hi], marker=marker,
                         color=color, label=label, capsize=4)
        ax.set_xlabel("Prefix length analyzed (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Length (95% CI)")
        ax.set_xticks(ANALYSIS_LENGTHS)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(RESULTS_DIR, fname)
        fig.savefig(out_path, dpi=200)
        print(f"Saved {out_path}")

    # keep a few full examples around for the appendix
    with open(os.path.join(RESULTS_DIR, "sample_outputs.txt"), "w", encoding="utf-8") as f:
        for prompt, strategy, seed, text in sample_outputs:
            header = f"=== PROMPT: {prompt}\nSTRATEGY: {strategy}"
            if seed:
                header += f" (seed={seed})"
            f.write(header + "\n" + text + "\n\n")
    print(f"Saved {RESULTS_DIR}/sample_outputs.txt")

    print("\nDONE. All results are in the results/ folder.")


if __name__ == "__main__":
    main()
