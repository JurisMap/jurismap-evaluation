"""How well a 2D map keeps the neighbourhoods of the original space (Phase 5, A2).

Everything is computed from two distance matrices over the same decisions: the original space
(cosine distance between the embedding vectors) and the map (Euclidean distance between the 2D
points). A decision is never its own neighbour; ties keep the corpus order.

- Trustworthiness at k: the map's k nearest neighbours are neighbours in the original space
  (Venna & Kaski; the definition of sklearn.manifold.trustworthiness, which a test checks).
- Continuity at k: the same with the roles swapped. The original k nearest neighbours stay
  neighbours on the map.
- kNN Jaccard at k: overlap of the two sets of k nearest neighbours, averaged over decisions.
- Spearman correlation of the pairwise distances on a sample of pairs.
- Label metrics: silhouette of a labelling, and the share of a decision's nearest neighbours that
  carry its label.
- Seed stability: Procrustes disparity between two fits, and the overlap of their neighbours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

from jm_eval.retrieval import metrics as retrieval_metrics


def cosine_distances(vectors: np.ndarray) -> np.ndarray:
    d = 1.0 - retrieval_metrics.cosine_similarity(vectors)
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, None)


def euclidean_distances(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=np.float64)
    return cdist(c, c)


def neighbour_order(dist: np.ndarray) -> np.ndarray:
    """order[i]: the other decisions, nearest to i first. Shape (n, n - 1)."""
    d = np.array(dist, dtype=np.float64)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :-1]


def neighbour_ranks(order: np.ndarray) -> np.ndarray:
    """ranks[i, j]: position of j among i's neighbours, 1 for the nearest; 0 on the diagonal."""
    n = order.shape[0]
    ranks = np.zeros((n, n), dtype=np.int32)
    ranks[np.arange(n)[:, None], order] = np.arange(1, n, dtype=np.int32)
    return ranks


class Space:
    """A distance matrix with every decision's neighbours in order and their ranks."""

    def __init__(self, dist: np.ndarray):
        self.dist = dist
        self.order = neighbour_order(dist)
        self.ranks = neighbour_ranks(self.order)

    @property
    def n(self) -> int:
        return self.dist.shape[0]


def trustworthiness(reference: Space, other: Space, k: int) -> float:
    """Penalises the k nearest neighbours in `other` that rank beyond k in `reference`.

    trustworthiness(original, map, k) is the trustworthiness of the map; with the arguments
    swapped it is the continuity.
    """
    n = reference.n
    if not 0 < k < n / 2:
        raise ValueError(f"k must be below n / 2 (k = {k}, n = {n})")
    excess = reference.ranks[np.arange(n)[:, None], other.order[:, :k]] - k
    return float(1.0 - excess[excess > 0].sum() * 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)))


def nearest(space: Space, k: int) -> np.ndarray:
    """near[i, j]: j is one of i's k nearest neighbours."""
    return (space.ranks >= 1) & (space.ranks <= k)


def neighbour_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Jaccard overlap of each decision's neighbour sets, given as the matrices of `nearest`."""
    shared = (a & b).sum(axis=1)
    return float((shared / ((a | b).sum(axis=1))).mean())


def knn_jaccard(a: Space, b: Space, k: int) -> float:
    """Mean Jaccard overlap of each decision's k nearest neighbours in the two spaces."""
    return neighbour_jaccard(nearest(a, k), nearest(b, k))


def sample_pairs(n: int, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`size` distinct pairs i < j of n decisions (every pair when there are fewer)."""
    i, j = np.triu_indices(n, k=1)
    if size < len(i):
        keep = np.sort(np.random.default_rng(seed).choice(len(i), size=size, replace=False))
        i, j = i[keep], j[keep]
    return i, j


def distance_spearman(a: np.ndarray, b: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> float:
    return float(spearmanr(a[pairs], b[pairs]).statistic)


def neighbourhood_metrics(original: Space, projected: Space, ks: list[int], jaccard_k: int,
                          pairs: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """The label-free metrics of one projection, keyed "<metric>@<k>" or "<metric>"."""
    out = {}
    for k in ks:
        out[f"trustworthiness@{k}"] = trustworthiness(original, projected, k)
        out[f"continuity@{k}"] = trustworthiness(projected, original, k)
    out[f"knn_jaccard@{jaccard_k}"] = knn_jaccard(original, projected, jaccard_k)
    out["distance_spearman"] = distance_spearman(original.dist, projected.dist, pairs)
    return out


def holdout_metrics(original: np.ndarray, projected: np.ndarray, k: int) -> dict[str, float]:
    """Trustworthiness, continuity and kNN Jaccard of decisions placed on a fitted map.

    Both matrices have a row per placed decision and a column per fitted decision: its distances to
    them in the original space and on the map. The neighbours of a placed decision are fitted
    decisions only, and the penalties are scaled as in `trustworthiness` for a set of that size.
    """
    m, candidates = original.shape
    n = candidates + 1                                             # the placed decision and its candidates
    if not 0 < k < n / 2:
        raise ValueError(f"k must be below n / 2 (k = {k}, n = {n})")
    rows = np.arange(m)[:, None]
    orders = [np.argsort(d, axis=1, kind="stable") for d in (original, projected)]
    ranks = []
    for order in orders:
        r = np.empty((m, candidates), dtype=np.int32)
        r[rows, order] = np.arange(1, candidates + 1, dtype=np.int32)
        ranks.append(r)
    scale = 2.0 / (m * k * (2.0 * n - 3.0 * k - 1.0))

    def kept(reference_ranks: np.ndarray, other_order: np.ndarray) -> float:
        excess = reference_ranks[rows, other_order[:, :k]] - k
        return float(1.0 - excess[excess > 0].sum() * scale)

    shared = ((ranks[0] <= k) & (ranks[1] <= k)).sum(axis=1)
    return {f"holdout_trustworthiness@{k}": kept(ranks[0], orders[1]), f"holdout_continuity@{k}": kept(ranks[1], orders[0]),
            f"holdout_knn_jaccard@{k}": float((shared / (2 * k - shared)).mean())}


def nearest_point_gap(space: Space) -> float:
    """Median distance from a point to its nearest point on the map, as a share of the largest
    distance between two points: how far apart neighbouring points are drawn (0 = on top of each other)."""
    nearest_distance = space.dist[np.arange(space.n), space.order[:, 0]]
    return float(np.median(nearest_distance) / space.dist.max())


# ---------------------------------------------------------------------------- labels

def class_codes(values) -> np.ndarray:
    """An integer per decision for classes with at least two decisions; -1 for the rest and for missing labels."""
    values = np.asarray(values, dtype=object)
    known = np.array([v is not None and v == v for v in values])
    codes = np.full(len(values), -1)
    if known.any():
        codes[known] = pd.factorize(values[known])[0]
        sizes = np.bincount(codes[known])
        codes[known & (sizes[np.clip(codes, 0, None)] < 2)] = -1
    return codes


def silhouette(dist: np.ndarray, codes: np.ndarray) -> float:
    """Mean silhouette of the labelled decisions under a distance matrix; NaN with fewer than two classes."""
    keep = np.flatnonzero(codes >= 0)
    if len(np.unique(codes[keep])) < 2:
        return float("nan")
    return float(silhouette_score(dist[np.ix_(keep, keep)], codes[keep], metric="precomputed"))


def same_label_share(dist: np.ndarray, allowed: np.ndarray, codes_by_label: dict[str, np.ndarray], k: int) -> dict[str, float]:
    """Per labelling: the mean share of a decision's k nearest candidates that carry its label.

    Candidates are the decisions of other processes (`allowed`, as in the retrieval benchmark).
    Neighbours without a label are left out of the share, and so are decisions without one.
    """
    top = retrieval_metrics.rank_candidates(-dist, allowed)[:, :k]
    out = {}
    for name, codes in codes_by_label.items():
        queries = np.flatnonzero(codes >= 0)
        top_codes = codes[top[queries]]
        labelled = top_codes >= 0
        same = (top_codes == codes[queries][:, None]) & labelled
        with np.errstate(invalid="ignore", divide="ignore"):
            share = same.sum(axis=1) / labelled.sum(axis=1)
        out[name] = float(np.nanmean(share)) if len(share) else float("nan")
    return out


def chance_share(allowed: np.ndarray, codes: np.ndarray) -> float:
    """The same share if the neighbours were drawn at random from each decision's candidates."""
    queries = np.flatnonzero(codes >= 0)
    candidates = allowed[queries] & (codes >= 0)[None, :]
    same = (codes[None, :] == codes[queries][:, None]) & candidates
    return float((same.sum(axis=1) / candidates.sum(axis=1)).mean())


# ---------------------------------------------------------------------------- seeds

def procrustes_disparity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared difference left between two maps after the best shift, scaling, rotation and
    reflection (0 = the same map, 1 = unrelated)."""
    return float(procrustes(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))[2])
