"""Metrics against an expert gold standard of scored pairs (Phase 4d, SUSEP; local only).

mAR and Recall@5 follow Mentzingen et al. (2024, Int. J. of Information Management Data Insights
4(2), 100247, Eqs. 1-3): a pair is relevant when the mean expert score is above zero; Recall@k of
a case with no relevant pair is 1; AR@k is Recall@k averaged over the cases; mAR is AR@k averaged
over k = 1..N with N the number of cases. nDCG@10 uses the mean score as the gain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def pair_matrix(ids: list, a: np.ndarray, b: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Symmetric matrix of pair values in the order of `ids` (NaN where a pair was not scored)."""
    index = {v: i for i, v in enumerate(ids)}
    m = np.full((len(ids), len(ids)), np.nan)
    for x, y, v in zip(a, b, values):
        m[index[x], index[y]] = m[index[y], index[x]] = v
    return m


def _ranked(similarity: np.ndarray, values: np.ndarray) -> np.ndarray:
    """values[i] reordered by decreasing similarity to case i, the case itself removed."""
    n = len(similarity)
    scores = similarity.astype(np.float64).copy()
    np.fill_diagonal(scores, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :n - 1]
    return np.take_along_axis(values, order, axis=1)


def recall_curves(similarity: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """recall[i, k-1]: Recall@k of case i for k = 1..N (N cases); 1 for a case with no relevant pair."""
    n = len(gold)
    hits = _ranked(similarity, np.nan_to_num(gold) > 0).astype(np.float64)
    hits = np.pad(hits, ((0, 0), (0, 1)))                       # k = N: every other case is already listed
    total = hits.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.cumsum(hits, axis=1) / total
    recall[total.ravel() == 0] = 1.0
    return recall[:, :n]


def per_case_metrics(similarity: np.ndarray, gold: np.ndarray, k_ndcg: int = 10) -> pd.DataFrame:
    """Per case: its mean Recall@k over k (the mean over cases is mAR), Recall@5 and graded nDCG@10.

    nDCG is NaN for a case with no relevant pair.
    """
    recall = recall_curves(similarity, gold)
    gains = _ranked(similarity, np.nan_to_num(gold))
    k = min(k_ndcg, gains.shape[1])
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains[:, :k] * discounts).sum(axis=1)
    ideal = (np.sort(gains, axis=1)[:, ::-1][:, :k] * discounts).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndcg = np.where(ideal > 0, dcg / ideal, np.nan)
    return pd.DataFrame({"mAR": recall.mean(axis=1), "Recall@5": recall[:, min(4, recall.shape[1] - 1)],
                         f"nDCG@{k_ndcg}": ndcg})


def pair_spearman(similarity: np.ndarray, gold: np.ndarray) -> float:
    """Spearman correlation between a method's similarity and the gold score over the scored pairs."""
    upper = np.triu_indices(len(gold), 1)
    scored = ~np.isnan(gold[upper])
    return float(sps.spearmanr(similarity[upper][scored], gold[upper][scored]).statistic)


def icc2(ratings: np.ndarray) -> dict:
    """ICC(2,1) and ICC(2,k): two-way random effects, absolute agreement. `ratings`: (items, raters)."""
    x = np.asarray(ratings, dtype=np.float64)
    n, k = x.shape
    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_error = (residual ** 2).sum() / ((n - 1) * (k - 1))
    single = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    average = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    return {"icc_2_1": float(single), "icc_2_k": float(average), "raters": k, "items": n}
