"""Candidates for the vector version 2 retrieves with (Phase 4.1; config/a1_candidates.yaml).

A candidate is a weighted sum of parts: the compressed report's vector, the full report's vector
(one call, truncated by the model) and the full report embedded in chunks. This module cuts the
chunks, pools and combines vectors, splits the queries into a development and a held-out half, and
computes the selection score with its bootstrap interval.
"""

from __future__ import annotations

import re

import numpy as np

from jm_eval.io import load_config

CONFIG = "a1_candidates"
_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")


def settings() -> dict:
    return load_config(CONFIG)


# ---------------------------------------------------------------------------- chunks

def _pack(pieces: list[str], max_chars: int, joiner: str) -> list[str]:
    """Join consecutive pieces into chunks of at most `max_chars`; a longer piece stays alone."""
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(joiner) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}{joiner}{piece}" if current else piece
    return chunks + ([current] if current else [])


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Cut `text` into chunks of at most `max_chars`, at paragraph ends where possible, then at
    sentence ends, then anywhere. A text within the limit is returned whole, unchanged."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for paragraph in (p for p in re.split(r"\n\s*\n|\n", text) if p.strip()):
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        for sentence in _pack(_SENTENCE_END.split(paragraph), max_chars, " "):
            pieces.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
    return _pack(pieces, max_chars, "\n")


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def pool(chunk_vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """One unit-length vector for a decision: the weighted mean of its unit-length chunk vectors."""
    pooled = (unit_rows(chunk_vectors) * np.asarray(weights, dtype=np.float64)[:, None]).sum(axis=0)
    return pooled / np.linalg.norm(pooled)


def combine(parts: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Unit-length weighted sum of the unit-length vectors of the parts (rows are decisions)."""
    total = sum(w * unit_rows(parts[name]) for name, w in weights.items())
    return unit_rows(total)


# ---------------------------------------------------------------------------- split

def development_mask(groups: np.ndarray, share: float, seed: int) -> np.ndarray:
    """True for the decisions whose queries are used in the selection. Process groups are shuffled
    with `seed` and the first `share` of them are the development half, so decisions of one process
    are on the same side."""
    unique = np.unique(groups)
    shuffled = np.random.default_rng(seed).permutation(unique)
    return np.isin(groups, shuffled[:int(round(len(unique) * share))])


# ---------------------------------------------------------------------------- score

def selection_score(values: dict[str, np.ndarray], families: dict[str, list[str]], rows: np.ndarray | None = None) -> np.ndarray:
    """Mean over families of the mean over the family's definitions of the mean over queries.

    values[definition] has shape (decisions,) or (decisions, candidates), NaN where the decision is
    not a query of that definition. `rows` selects (and may repeat) decisions, for the bootstrap.
    """
    family_means = []
    for definitions in families.values():
        means = [np.nanmean(values[d] if rows is None else values[d][rows], axis=0) for d in definitions]
        family_means.append(np.mean(means, axis=0))
    return np.mean(family_means, axis=0)


def score_difference_interval(values: dict[str, np.ndarray], families: dict[str, list[str]], column: int, reference: int, *,
                              resamples: int, seed: int, level: float = 0.95) -> tuple[float, float, float]:
    """Paired bootstrap of score(column) - score(reference) over decisions: (difference, low, high)."""
    n = len(next(iter(values.values())))
    pair = {d: v[:, [column, reference]] for d, v in values.items()}
    observed = selection_score(pair, families)
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for b in range(resamples):
        scores = selection_score(pair, families, rng.integers(0, n, size=n))
        differences[b] = scores[0] - scores[1]
    tail = (1 - level) / 2
    low, high = np.quantile(differences, [tail, 1 - tail])
    return float(observed[0] - observed[1]), float(low), float(high)
