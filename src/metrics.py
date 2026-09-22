"""Repetition and lexical diversity metrics, computed directly on GPT-2
BPE token-id sequences (not on whitespace-split words)."""

import numpy as np


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def rep_n(seq, n=3):
    # fraction of n-grams that are repeats of an earlier one in the sequence
    # (Welleck et al., 2020) -- higher means more repetitive/degenerate
    grams = ngrams(seq, n)
    if not grams:
        return np.nan
    seen, dup = set(), 0
    for g in grams:
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / len(grams)


def distinct_n(seq, n=2):
    # fraction of unique n-grams -- higher means more lexically diverse
    grams = ngrams(seq, n)
    if not grams:
        return np.nan
    return len(set(grams)) / len(grams)
