"""A5 context-size test (Phase 7): the context of each question at sizes other than Phase 6's 10.

The context at size k is the case set's k decisions most similar to its seed. Phase 6's members
are the seed's 10 most similar decisions (in order), so k = 5 is their first five; for k = 20 and
30 the seed's next most similar decisions are added by Phase 6's own rule (jm_eval.chatbot.casesets).

A frozen Phase 6 question is asked at a size only where Phase 6's rules still hold there; a count
is counted again over the context, and open questions get new reference points (scripts/72).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jm_eval.chatbot import questions as Q


def extend_sets(sets: list[dict], labels: pd.DataFrame, vectors: np.ndarray, size: int, excluded: set[str]) -> list[dict]:
    """Each set's members followed by its seed's next most similar decisions, up to `size` in all.

    `labels` and `vectors` are aligned and in `acordao_hash` order; the vectors have unit length.
    Added decisions are never the seed's process, a decision in `excluded`, or one added to an
    earlier set. Returns, per set, `ranked_hashes` (members first) and their similarity to the seed.
    """
    hashes = labels["acordao_hash"].to_numpy()
    groups = labels["process_group"].to_numpy()
    row_of = {h: i for i, h in enumerate(hashes)}
    used = np.isin(hashes, sorted(excluded))
    out = []
    for s in sets:
        seed_row = row_of[s["seed_hash"]]
        members = [row_of[h] for h in s["case_hashes"]]
        similarity = vectors @ vectors[seed_row]
        member_similarity = similarity[members].copy()
        similarity[(groups == groups[seed_row]) | used] = -np.inf
        similarity[members] = -np.inf
        added = np.argsort(-similarity, kind="stable")[:size - len(members)]
        if not np.isfinite(similarity[added]).all():
            raise ValueError(f"{s['set_id']}: not enough decisions to extend the set to {size}")
        used[added] = True
        ranked = [*members, *added.tolist()]
        values = [*member_similarity.tolist(), *similarity[added].tolist()]
        if any(b > a + 1e-9 for a, b in zip(values, values[1:])):
            raise ValueError(f"{s['set_id']}: the extended ranking is not in order of similarity")
        out.append({"set_id": s["set_id"], "seed_hash": s["seed_hash"], "members": len(members),
                    "ranked_hashes": [str(hashes[i]) for i in ranked],
                    "similarity": [round(float(v), 6) for v in values]})
    return out


def count_gold(question: dict, rows: pd.DataFrame) -> int:
    """A count question's gold value over `rows`: the decisions with the value the question names."""
    variant, value = question["variant"], question["value"]
    if variant == "outcome":
        values = rows["acordao_resultado"].astype(str)
    elif variant == "rapporteur":
        values = rows["acordao_relator"].map(lambda v: (Q.as_list(v) or [""])[0])
    else:
        values = rows["ano_sessoes"].astype(int).astype(str)
    return int((values == value).sum())


def setting_questions(questions: list[dict], ranked: dict[str, list[str]], k: int, corpus: pd.DataFrame,
                      types: list[str]) -> tuple[list[dict], list[dict]]:
    """The questions asked with a context of size k, and those dropped there (with the reason).

    `ranked` maps a set to its decisions in order of similarity to the seed; `corpus` has
    `acordao_hash`. Reference points are kept only where the context is Phase 6's.
    """
    by_hash = corpus.set_index("acordao_hash", drop=False)
    asked, dropped = [], []
    for q in questions:
        if q["type"] not in types:
            continue
        context = ranked[q["set_id"]][:k]
        rows = by_hash.loc[context]
        same = context == q["case_hashes"]
        new = {**q, "case_hashes": context, "set_size": len(context)}
        reason = None
        if q["type"] == "templated" and q["subtype"] == "count":
            new["gold"] = count_gold(q, rows)
            new["phase6_gold"] = q["gold"]
        elif q["type"] == "templated":
            if q["target_hash"] not in context:
                reason = "the decision asked about is not in the context"
            elif q["target_hash"] not in set(Q._unique_titles(rows)["acordao_hash"]):
                reason = "the title asked about does not identify one decision of the context"
        elif q["type"] == "open":
            if q["subtype"] == "compare_outcomes":
                granted = rows["acordao_resultado"].isin(Q.GRANTED)
                if not (granted.any() and (~granted).any()):
                    reason = "the context lacks one of the two groups to compare"
            if not same:
                new.pop("reference_points", None)           # written again from the context's key points
        if reason:
            dropped.append({"question_id": q["question_id"], "reason": reason})
        else:
            asked.append(new)
    return asked, dropped


def describe(extended: list[dict], sizes: list[int], corpus: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per set and size: what the context holds (for the report; no text)."""
    cor, lab = corpus.set_index("acordao_hash"), labels.set_index("acordao_hash")
    rows = []
    for s in extended:
        for k in sizes:
            context = s["ranked_hashes"][:k]
            chars = (cor.loc[context, "acordao_relatorio"].fillna("").str.len()
                     + cor.loc[context, "acordao_voto_relator"].fillna("").str.len())
            granted = cor.loc[context, "acordao_resultado"].isin(Q.GRANTED)
            rows.append({"set_id": s["set_id"], "k": k, "decisions": len(context),
                         "added_to_phase6_set": max(0, k - s["members"]),
                         "similarity_min": min(s["similarity"][:k]),
                         "same_category_as_seed": int((lab.loc[context, "category"] == lab.at[s["seed_hash"], "category"]).sum()),
                         "granted_wholly_or_partly": int(granted.sum()), "not_granted_or_other": int((~granted).sum()),
                         "report_and_vote_chars": int(chars.sum())})
    return pd.DataFrame(rows)
