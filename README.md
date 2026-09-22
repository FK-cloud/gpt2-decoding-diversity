# Choosing the Next Token
### How Decoding Strategy and Generation Length Shape Repetition and Lexical Diversity in GPT-2 Small

**Author:** Muhammad Faizan Kayani (1898693)
**Course:** NLP
**Instructor:** Prof. Kai Kugler
**Semester:** Summer 2026

---

## Overview

This project investigates whether the choice of decoding strategy in GPT-2 Small produces a measurable difference in how repetitive or lexically varied its generated text is, and whether that difference changes as generation continues.

We compare:
- **Greedy decoding** - deterministic, always selects the single most probable token
- **Top-p (nucleus) sampling** - samples from the smallest set of tokens whose cumulative probability reaches 0.9

Both strategies are evaluated using two automatic metrics - Rep-3 (repetition) and Distinct-2 (lexical diversity) - at three checkpoints of the same generation: 50, 100, and 200 tokens.

The central research question is:

> Do greedy decoding and top-p sampling produce measurably different levels of repetition and lexical diversity in GPT-2 Small, and do these differences become more pronounced as generated text becomes longer?

## Research Hypothesis

We hypothesize that top-p sampling will produce less repetitive, more lexically diverse output than greedy decoding at every tested length, and that this gap will widen the longer generation is allowed to continue because maximization-based decoding has no mechanism to escape a high-probability phrase once it locks onto one.

## Research Context

Autoregressive models generate text token-by-token, and the decoding rule used to pick each token shapes the output considerably. Greedy decoding always takes the highest-probability token, which sounds optimal but, as Holtzman et al. (2020) showed, tends to produce degenerate, looping text repeatedly selecting a token raises its own probability at the next step, so the model gets trapped. Nucleus sampling was proposed as a fix: rather than fixing on one token, it samples from a shrinking or expanding pool depending on the model's confidence.

Most comparisons of these two strategies evaluate a single, fixed generation length. This project instead asks a narrower, less commonly examined question: as the same generation keeps going, does the gap between the two strategies stay constant, or does it grow?

---

## Methodology

### Model
GPT-2 Small (gpt2, 124M parameters, HuggingFace transformers), used as-is with no fine-tuning.

### Prompt Set
20 fixed English prompts, 4 each across 5 themes — News, Narrative, Science, History, Opinion. This is a disclosed, fixed convenience sample chosen for reproducibility, not a validated benchmark.

### Decoding Conditions

| Strategy | Configuration | Runs per prompt |
|---|---|---|
| Greedy | do_sample=False | 1 |
| Top-p | do_sample=True, top_p=0.9, seeds 11/22/33 | 3 |

Each prompt is generated once to 200 tokens per condition. Rather than generating separately for each length, the 50- and 100-token observations are prefixes of that same 200-token continuation — this keeps the length comparison tied to one underlying generation instead of introducing fresh sampling noise at each length.

### Metrics
- Rep-3 = fraction of repeated 3-grams → lower is better
- Distinct-2 = fraction of unique bigrams → higher is better

Both are computed on GPT-2's BPE token IDs, not word-level tokens.

### Statistical Approach
Top-p's 3 seeds are averaged per prompt first, so the unit of analysis stays at 20 prompts rather than treating 60 outputs as independent. The main comparison uses a paired bootstrap (5,000 resamples, prompt as the resampling unit, 95% percentile CI) on (top-p − greedy) at each length. A second bootstrap directly tests whether that effect changes between 50 and 200 tokens, using the same resampled prompts at both lengths to keep the pairing intact.

---

## Results

Top-p sampling was less repetitive and more lexically diverse than greedy decoding at every length tested — and the size of that advantage grew the longer generation continued.

### Results Summary

| Length | Metric | Greedy | Top-p | Δ (Top-p − Greedy) | 95% CI |
|---|---|---|---|---|---|
| 50 | Rep-3 | 0.261 | 0.008 | −0.253 | [−0.348, −0.160] |
| 50 | Distinct-2 | 0.699 | 0.973 | +0.274 | [+0.182, +0.371] |
| 100 | Rep-3 | 0.454 | 0.024 | −0.430 | [−0.550, −0.307] |
| 100 | Distinct-2 | 0.511 | 0.943 | +0.433 | [+0.314, +0.550] |
| 200 | Rep-3 | 0.668 | 0.030 | −0.638 | [−0.732, −0.534] |
| 200 | Distinct-2 | 0.306 | 0.924 | +0.618 | [+0.520, +0.704] |

All six intervals exclude zero. Full precision: results/bootstrap_main.csv.

### 1. At 50 tokens
The strategies are already clearly separated: greedy's Rep-3 (0.261) is more than 30× top-p's (0.008), and top-p's Distinct-2 (0.973) sits near its ceiling while greedy's (0.699) is already noticeably lower.

### 2. At 100 tokens
The gap widens. Greedy's repetition rate rises to 0.454 while top-p stays at 0.024; greedy's diversity drops to 0.511 against top-p's 0.943.

### 3. At 200 tokens
The contrast is largest here: greedy Rep-3 (0.668) is over 22× top-p's (0.030), and greedy's Distinct-2 (0.306) has fallen to under a third of top-p's (0.924).

### Does the Gap Actually Grow?

| Metric | Effect @50 | Effect @200 | Change | 95% CI |
|---|---|---|---|---|
| Rep-3 | −0.253 | −0.638 | −0.384 | [−0.448, −0.316] |
| Distinct-2 | +0.274 | +0.618 | +0.344 | [+0.278, +0.406] |

Both change-intervals exclude zero — the widening is not noise. Full precision: results/bootstrap_length_effect.csv.

---

## Comparative Analysis

Unlike comparisons where the "better" option can flip depending on category, here the direction of the effect never changes — top-p wins on both metrics at every length. What changes is the size of that advantage, and it only grows. This matches the mechanism Holtzman et al. describe: greedy's degeneration isn't a one-time cost, it compounds the longer the model is allowed to run.

## Key Findings

1. Both metrics separate the strategies at every length tested — all 6 length × metric comparisons are significant (CI excludes 0).
2. The gap is not fixed — it grows substantially. Rep-3's disadvantage for greedy nearly triples (−0.253 → −0.638); Distinct-2's more than doubles (+0.274 → +0.618).
3. Greedy's own repetition compounds over time — its Rep-3 climbs from 0.261 to 0.668 purely as a function of length, independent of any comparison to top-p.
4. Top-p stays comparatively flat — Rep-3 never exceeds 0.03, Distinct-2 never drops below 0.92, across all three lengths.

## Hypothesis Evaluation

**Result: Supported.** Top-p was significantly less repetitive and more diverse than greedy at every length, and the length-effect test confirms that gap widens significantly from 50 to 200 tokens.

## Unexpected Outcome

Going in, the assumption was that top-p might also degrade somewhat with length, just less severely than greedy. That's not what happened — top-p's scores were remarkably flat across all three lengths. Greedy, meanwhile, didn't just stay bad — it got progressively worse. This suggests the "lock-in" effect isn't a fixed penalty greedy pays once, but a compounding one that worsens the longer a single generation runs.

## Visualization

- effect_size_growth.png — the headline figure: effect size (top-p − greedy) vs. length, both metrics
- bar_rep3.png, bar_distinct2.png — grouped bars by strategy and length, with CI
- plot_rep3.png, plot_distinct2.png — line plots of each metric vs. length
- boxplot_distribution_200tok.png — per-prompt spread at 200 tokens, not just the mean

## Limitations

- Only one top-p value (p=0.9) was tested — no top-k or temperature sampling
- 20 prompts is a controlled but small, non-benchmark sample
- Rep-3/Distinct-2 are surface-level lexical measures — they say nothing about coherence, factuality, or fluency
- Findings are specific to GPT-2 Small (124M); larger models likely degenerate less under greedy decoding
- Metrics are computed on BPE tokens, not word-level units, so they aren't directly numerically comparable to word-level implementations

## Interpretation

A significant Rep-3/Distinct-2 gap doesn't by itself mean top-p output is "better" — it means top-p text is measurably less repetitive and more lexically varied. Diversity is not the same as quality; a highly varied continuation can still be incoherent. What this experiment supports is a narrower claim: for this GPT-2 Small setup, the surface-level repetition/diversity gap between the two decoding strategies is real, and it grows with length.

---

## Reproducibility

```bash
pip install -r requirements.txt
python run_experiment.py     # full pipeline → results/*.csv + 2 line plots
python make_visuals.py       # bar charts, effect-size plot, boxplot
```

## Project Structure

```
gpt2-decoding-diversity/
├── data/prompts.py
├── src/ (generation.py, metrics.py, bootstrap.py)
├── results/ (CSVs + all figures)
├── config.py
├── run_experiment.py
├── make_visuals.py
├── run.sh
├── requirements.txt
└── README.md
```

## Conclusion

This project compared greedy decoding and top-p sampling in GPT-2 Small on two automatic metrics across three generation lengths, using paired bootstrap inference. Top-p was significantly less repetitive and more diverse than greedy at every length, and — more notably — that advantage roughly tripled (Rep-3) and doubled (Distinct-2) between 50 and 200 tokens. Greedy's own degeneration compounds with length; top-p's does not. These results are consistent with the likelihood–quality paradox described by Holtzman et al. (2020).

## Author

Muhammad Faizan Kayani (1898693)
NLP · Prof. Kai Kugler · Summer 2026
