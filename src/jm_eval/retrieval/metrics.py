"""Leave-one-out retrieval metrics from a similarity matrix and a relevance matrix (Phase 4c).

Each decision is a query against all the others. A decision is never ranked for itself or for a
decision of the same process. Queries with no relevant decision are not evaluated, so the query set
depends only on the labels and is the same for every method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["P@5", "P@10", "R@5", "R@10", "MAP", "MRR", "nDCG@10", "mAR"]


def candidate_mask(groups: np.ndarray) -> np.ndarray:
    """allowed[i, j]: decision j may be ranked for query i (another decision of another process)."""
    groups = np.asarray(groups)
    return groups[:, None] != groups[None, :]


def rank_candidates(similarity: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Candidates of every query, most similar first; ties keep the corpus order. Excluded ones go last."""
    scores = np.where(allowed, similarity, -np.inf)
    return np.argsort(-scores, axis=1, kind="stable")


def per_query_metrics(similarity: np.ndarray, relevant: np.ndarray, allowed: np.ndarray) -> pd.DataFrame:
    """P@5, P@10, R@5, R@10, average precision, reciprocal rank, nDCG@10 and mAR for every query that
    has a relevant decision. Indexed by the query's row number.

    mAR is the query's Recall@k averaged over k = 1..(number of candidates): the mean Average Recall
    of Mentzingen et al. (2024) on the whole ranking. It rewards placing relevant decisions high
    anywhere in the ranking, unlike the top-k metrics, and its value depends on the corpus size.
    """
    relevant = relevant & allowed
    queries = np.flatnonzero(relevant.any(axis=1))
    order = rank_candidates(similarity[queries], allowed[queries])
    hits = np.take_along_axis(relevant[queries], order, axis=1).astype(np.float64)
    if hits.shape[1] < 10:                                            # fewer than 10 candidates: empty ranks
        hits = np.pad(hits, ((0, 0), (0, 10 - hits.shape[1])))
    n_relevant = hits.sum(axis=1)
    ranks = np.arange(1, hits.shape[1] + 1)
    precision_at = np.cumsum(hits, axis=1) / ranks
    recall_at = np.cumsum(hits, axis=1) / n_relevant[:, None]
    n_candidates = allowed[queries].sum(axis=1)
    # ranks after the last candidate hold the excluded decisions, where Recall is already 1
    recall_sum = recall_at.sum(axis=1) - (hits.shape[1] - n_candidates)
    discounts = 1.0 / np.log2(np.arange(2, 12))                       # ranks 1..10
    ideal = np.array([discounts[:int(min(n, 10))].sum() for n in n_relevant])
    return pd.DataFrame({
        "P@5": hits[:, :5].sum(axis=1) / 5,
        "P@10": hits[:, :10].sum(axis=1) / 10,
        "R@5": hits[:, :5].sum(axis=1) / n_relevant,
        "R@10": hits[:, :10].sum(axis=1) / n_relevant,
        "MAP": (precision_at * hits).sum(axis=1) / n_relevant,
        "MRR": 1.0 / (hits.argmax(axis=1) + 1),
        "nDCG@10": (hits[:, :10] * discounts).sum(axis=1) / ideal,
        "mAR": recall_sum / n_candidates,
    }, index=queries)


def neighbour_agreement(similarity: np.ndarray, allowed: np.ndarray, classes: np.ndarray, k: int = 10) -> pd.DataFrame:
    """For every query with a known class: the share of its top-k neighbours of the same class, and
    the share expected if the neighbours were drawn at random from its candidates.

    Neighbours of unknown class are left out of both shares. `classes` holds None/NaN for unknown.
    """
    classes = np.asarray(classes, dtype=object)
    known = np.array([c is not None and c == c for c in classes])
    codes = np.full(len(classes), -1)
    codes[known] = pd.factorize(classes[known])[0]
    queries = np.flatnonzero(known)
    top = rank_candidates(similarity[queries], allowed[queries])[:, :k]
    top_codes = codes[top]
    top_known = top_codes >= 0
    same_top = (top_codes == codes[queries][:, None]) & top_known
    candidates = allowed[queries] & known[None, :]
    same_all = (codes[None, :] == codes[queries][:, None]) & candidates
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = same_top.sum(axis=1) / top_known.sum(axis=1)
        expected = same_all.sum(axis=1) / candidates.sum(axis=1)
    return pd.DataFrame({"observed": observed, "expected": expected}, index=queries).dropna()


def cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v @ v.T
