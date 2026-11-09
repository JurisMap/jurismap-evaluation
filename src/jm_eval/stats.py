"""Statistics helpers: intervals of proportions and of means over queries, paired tests, Holm.

Used by the label audit (Phase 4a) and the retrieval protocol (Phase 4c). Everything random takes
a seed, so a rerun gives the same numbers.
"""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval of a proportion (95% by default); (nan, nan) for n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_ci(values: np.ndarray, *, resamples: int = 10000, seed: int = 20260926,
                      level: float = 0.95, chunk: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval of the mean over rows (queries).

    `values` has shape (queries,) or (queries, columns). Every column is resampled with the same
    draws of queries, so intervals of several methods or metrics come from the same resamples.
    Returns (low, high), one value per column.
    """
    x = np.asarray(values, dtype=np.float64)
    squeeze = x.ndim == 1
    n = len(x)
    if n == 0:
        nan = np.full(1 if squeeze else x.shape[1], np.nan)
        return (float("nan"), float("nan")) if squeeze else (nan, nan)
    x = x.reshape(n, -1)
    rng = np.random.default_rng(seed)
    means = np.empty((resamples, x.shape[1]))
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        draws = rng.integers(0, n, size=(size, n))
        means[start:start + size] = x[draws].mean(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail], axis=0)
    return (float(low[0]), float(high[0])) if squeeze else (low, high)


def paired_randomisation_test(a: np.ndarray, b: np.ndarray, *, permutations: int = 10000,
                              seed: int = 20260926, chunk: int = 500) -> tuple[float, float]:
    """Two-sided paired randomisation (sign-flip) test of the mean difference a - b over queries.

    Returns (observed mean difference, p). p = (flips at least as extreme + 1) / (permutations + 1).
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = float(d.mean()) if len(d) else float("nan")
    if len(d) == 0:
        return observed, float("nan")
    rng = np.random.default_rng(seed)
    extreme, threshold = 0, abs(observed) - 1e-12
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        signs = rng.integers(0, 2, size=(size, len(d)), dtype=np.int8) * 2 - 1
        extreme += int((np.abs((signs * d).mean(axis=1)) >= threshold).sum())
    return observed, (extreme + 1) / (permutations + 1)


def holm(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in the order given."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def krippendorff_alpha_ordinal(units: list[list[float | None]]) -> float | None:
    """Krippendorff's alpha with the ordinal metric. `units`: one list per unit (e.g. a draft), one value
    per rater (None where a rater gave none). Units with fewer than two values are left out.

    Coincidence matrix o[c][k] (each pair of values in a unit weighted 1 / (m_u - 1)); ordinal distance
    delta^2(c, k) = (sum of n_g for g from c to k - (n_c + n_k) / 2)^2; alpha = 1 - (n - 1) * sum(o * d)
    / sum(n_c * n_k * d)."""
    pairable = [[v for v in u if v is not None] for u in units]
    pairable = [u for u in pairable if len(u) >= 2]
    values = sorted({v for u in pairable for v in u})
    if len(values) < 2:
        return None
    index = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for u in pairable:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[index[a], index[b]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    cum = np.concatenate([[0], np.cumsum(n_c)])
    d = np.zeros_like(o)
    for c in range(len(values)):
        for k in range(len(values)):
            lo, hi = min(c, k), max(c, k)
            d[c, k] = (cum[hi + 1] - cum[lo] - (n_c[c] + n_c[k]) / 2) ** 2
    expected = (np.outer(n_c, n_c) * d).sum()
    return float(1 - (n - 1) * (o * d).sum() / expected) if expected else None
