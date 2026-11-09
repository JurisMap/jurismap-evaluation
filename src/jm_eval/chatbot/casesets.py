"""Case sets of the A3 chatbot evaluation (Phase 6, task 1).

From v2.4 the app offers, as context cases, the 10 decisions most similar to the user's case by the
retrieval vectors. A set is built the same way: a seed decision plays the user's case and the set is
its 10 most similar decisions, without the seed and the other decisions of its process. One seed is
drawn at random in each of the most frequent legal-basis keys (Phase 4a, `base_key`), so the sets
are stratified by legal basis. Sets share no decision, and no seed is in a set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jm_eval.io import load_config

CONFIG = "a3_chatbot"


def settings() -> dict:
    return load_config(CONFIG)


def strata(labels: pd.DataFrame, n: int) -> list[str]:
    """The `n` most frequent legal-basis keys (ties in key order)."""
    counts = labels["base_key"].dropna().value_counts()
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [key for key, _ in ordered[:n]]


def build_sets(labels: pd.DataFrame, vectors: np.ndarray, *, n_sets: int, size: int, seed: int,
               development_sets: int = 0) -> list[dict]:
    """The evaluated sets, then the development sets, each from its own stratum.

    `labels` and `vectors` are aligned and in `acordao_hash` order; the vectors have unit length.
    """
    hashes = labels["acordao_hash"].to_numpy()
    groups = labels["process_group"].to_numpy()
    keys = strata(labels, n_sets + development_sets)
    if len(keys) < n_sets + development_sets:
        raise ValueError("not enough legal-basis keys for the sets")
    rng = np.random.default_rng(seed)
    used = np.zeros(len(labels), dtype=bool)          # seeds and members of earlier sets
    sets = []
    for number, key in enumerate(keys, 1):
        candidates = np.flatnonzero((labels["base_key"] == key).to_numpy() & ~used)
        seed_row = int(rng.choice(candidates))
        similarity = vectors @ vectors[seed_row]
        similarity[(groups == groups[seed_row]) | used] = -np.inf
        members = np.argsort(-similarity, kind="stable")[:size]
        used[seed_row] = True
        used[members] = True
        development = number > n_sets
        sets.append({
            "set_id": f"D{number - n_sets}" if development else f"S{number}",
            "development": development,
            "stratum": key,
            "stratum_decisions": int((labels["base_key"] == key).sum()),
            "seed_hash": str(hashes[seed_row]),
            "case_hashes": [str(hashes[i]) for i in members],
            "similarity": [round(float(similarity[i]), 6) for i in members],
        })
    return sets


def describe(sets: list[dict], labels: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per set: what its decisions share (for the report; no text)."""
    lab, cor = labels.set_index("acordao_hash"), corpus.set_index("acordao_hash")
    rows = []
    for s in sets:
        members = s["case_hashes"]
        outcomes = cor.loc[members, "acordao_resultado"].value_counts()
        chars = (cor.loc[members, "acordao_relatorio"].fillna("").str.len()
                 + cor.loc[members, "acordao_voto_relator"].fillna("").str.len())
        rows.append({
            "set_id": s["set_id"], "development": s["development"], "stratum": s["stratum"],
            "stratum_decisions": s["stratum_decisions"], "seed_hash": s["seed_hash"],
            "decisions": len(members),
            "same_legal_basis_as_seed": int((lab.loc[members, "base_key"] == s["stratum"]).sum()),
            "same_category_as_seed": int((lab.loc[members, "category"] == lab.at[s["seed_hash"], "category"]).sum()),
            "processes": int(lab.loc[members, "process_group"].nunique()),
            "years": f"{int(cor.loc[members, 'ano_sessoes'].min())}-{int(cor.loc[members, 'ano_sessoes'].max())}",
            "outcomes": "; ".join(f"{k}: {v}" for k, v in outcomes.items()),
            "similarity_min": min(s["similarity"]), "similarity_max": max(s["similarity"]),
            "report_and_vote_chars": int(chars.sum()),
        })
    return pd.DataFrame(rows)
