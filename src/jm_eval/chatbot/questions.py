"""Question set of the A3 chatbot evaluation (Phase 6, task 2).

Per case set: four templated questions whose answer is a field of one decision (or a count over the
set), one unanswerable question (what it asks is in no field of the set, checked here) and two open
questions. Templates, the assignment of types to sets and the seed are in config/a3_chatbot.yaml.
The reference key points of the open questions are written separately (jm_eval.chatbot.material).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from jm_eval.chatbot import scoring as S
from jm_eval.retrieval.labels import OUTCOMES, process_keys
from jm_eval.retrieval.provisions import provision_ids

# The fields the chatbot receives of every context case (backend `context_fields`), plus the title.
CONTEXT_FIELDS = ["acordao_titulo", "ano_sessoes", "sessao_nr_int", "sessao_datas", "sessao_acordao", "acordao_url",
                  "acordao_recorrente", "acordao_recorrido", "acordao_relator", "acordao_advogado", "acordao_ementa",
                  "acordao_penalidade", "acordao_base", "acordao_acordao", "acordao_participantes",
                  "acordao_relatorio", "acordao_voto_relator", "acordao_resultado"]
GRANTED = {"Provido", "Parcialmente provido"}


TITLE_RE = re.compile(r"(Processo|Recurso) \S")          # the forms a question can name with "do ..."


def clean_title(title: str) -> str:
    """A decision's title as a question writes it: single spaces, no closing punctuation."""
    return re.sub(r"\s+", " ", title).strip().rstrip(".;,: ")


def as_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]


def context_text(rows: pd.DataFrame) -> str:
    """Every field the chatbot receives of these decisions, as one text."""
    parts = []
    for row in rows.itertuples(index=False):
        for field in CONTEXT_FIELDS:
            value = getattr(row, field)
            parts.append(", ".join(as_list(value)) if not isinstance(value, (int, np.integer)) else str(value))
    return "\n".join(parts)


def gold_of(kind: str, row, canon: dict | None) -> object | None:
    """The gold value of a templated question about one decision; None if the decision cannot be asked."""
    if kind == "outcome":
        return OUTCOMES.get(str(row.acordao_resultado).strip().lower())
    if kind == "rapporteur":
        return as_list(row.acordao_relator) or None
    if kind == "appellant":
        return as_list(row.acordao_recorrente) or None
    if kind == "lawyer":
        return as_list(row.acordao_advogado) or None
    if kind == "session_date":
        return S.parse_session_date(row.sessao_datas, row.ano_sessoes)
    if kind == "legal_basis":
        ids = provision_ids(row.acordao_base, canon)
        return ids if ids and all(":art" in i for i in ids) else None
    if kind == "penalty":
        gold = S.parse_penalty(row.acordao_penalidade)
        return gold if gold["types"] and len(gold["amounts"]) <= 1 else None
    raise ValueError(kind)


def _unique_titles(rows: pd.DataFrame) -> pd.DataFrame:
    """Decisions whose title, and whose process numbers, identify one decision of the set.

    Only titles of the usual forms ("Processo ...", "Recurso ...") are asked about.
    """
    titles = rows["acordao_titulo"].map(clean_title)
    keys = [process_keys(t) for t in titles]
    shared = [any(k & other for j, other in enumerate(keys) if j != i) for i, k in enumerate(keys)]
    usual = titles.map(lambda t: TITLE_RE.match(t) is not None).to_numpy()
    return rows[~titles.duplicated(keep=False).to_numpy() & ~np.array(shared) & usual]


def _count_question(variant: str, rows: pd.DataFrame, templates: dict) -> tuple[str, int, str]:
    if variant == "outcome":
        values = rows["acordao_resultado"]
    elif variant == "rapporteur":
        values = rows["acordao_relator"].map(lambda v: (as_list(v) or [""])[0])
    else:
        values = rows["ano_sessoes"].astype(int)
    counts = values[values.astype(str) != ""].value_counts()
    value = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]       # the most frequent value
    return templates[variant].format(value=value), int(counts[value]), str(value)


def outside_decision(rows: pd.DataFrame, corpus: pd.DataFrame, excluded: set[str], rng) -> pd.Series:
    """A decision of the corpus outside every set whose process numbers appear in no text of this set."""
    in_set = set(S.process_numbers(context_text(rows)))
    order = rng.permutation(len(corpus))
    for i in order:
        row = corpus.iloc[int(i)]
        title = clean_title(row["acordao_titulo"])
        keys = process_keys(title)
        if row["acordao_hash"] not in excluded and title.startswith("Processo ") and keys and not keys & in_set:
            return row
    raise ValueError("no decision outside the set")


def no_cnpj(row) -> bool:
    text = context_text(pd.DataFrame([row._asdict() if hasattr(row, "_asdict") else dict(row)]))
    return not S.has_cnpj(text) and "cnpj" not in S.normalise(text)


def build_questions(case_set: dict, set_index: int, corpus: pd.DataFrame, cfg: dict, canon: dict | None,
                    excluded: set[str], seed: int, *, every_type: bool = False) -> list[dict]:
    """The seven questions of one case set. `corpus` is in position order, with `acordao_hash`.

    `every_type` (development set only) asks every templated type, every count variant, both kinds
    of unanswerable question and the four open questions, so each scorer and prompt can be tried.
    """
    q_cfg = cfg["questions"]
    by_hash = corpus.set_index("acordao_hash", drop=False)
    rows = by_hash.loc[case_set["case_hashes"]]
    askable = _unique_titles(rows)
    rng = np.random.default_rng([seed, set_index])
    set_id, questions, used = case_set["set_id"], [], set()
    base = {"set_id": set_id, "case_hashes": case_set["case_hashes"], "set_size": len(rows)}

    def pick(plan: str):
        return q_cfg[plan][set_index % len(q_cfg[plan])]

    def add(qtype: str, subtype: str, text: str, **extra) -> None:
        number = sum(q["type"] == qtype for q in questions) + 1
        questions.append({"question_id": f"{set_id}-{qtype[0].upper()}{number}", **base, "type": qtype,
                          "subtype": subtype, "question": text, **extra})

    count_templates = q_cfg["templated"]["count"]["variants"]
    count_variants = list(count_templates) if every_type else [pick("count_variant_by_set") or "outcome"]
    for kind in (list(q_cfg["templated"]) if every_type else pick("templated_by_set")):
        if kind == "count":
            for variant in count_variants:
                text, gold, value = _count_question(variant, rows, count_templates)
                add("templated", "count", text, gold=gold, variant=variant, value=value, target_hash=None)
            continue
        golds = {r.acordao_hash: gold_of(kind, r, canon) for r in askable.itertuples(index=False)}
        eligible = [h for h, g in golds.items() if g is not None]
        fresh = [h for h in eligible if h not in used] or eligible
        if not fresh:
            raise ValueError(f"{set_id}: no decision can be asked about its {kind}")
        target = str(rng.choice(sorted(fresh)))
        used.add(target)
        title = clean_title(by_hash.at[target, "acordao_titulo"])
        add("templated", kind, q_cfg["templated"][kind]["text"].format(title=title), gold=golds[target],
            target_hash=target, target_title=title)

    for kind in (list(q_cfg["unanswerable"]) if every_type else [pick("unanswerable_by_set")]):
        candidates = [r for r in askable.itertuples(index=False) if no_cnpj(r)] if kind == "appellant_cnpj" else []
        if kind == "appellant_cnpj" and candidates:
            target = candidates[int(rng.integers(len(candidates)))]
            title = clean_title(target.acordao_titulo)
            add("unanswerable", kind, q_cfg["unanswerable"][kind]["text"].format(title=title),
                target_hash=target.acordao_hash, target_title=title,
                absence="no CNPJ and no mention of one in any field of the decision")
        else:                                                    # also the fallback when every decision has a CNPJ
            outside = outside_decision(rows, corpus, excluded, rng)
            title = clean_title(outside["acordao_titulo"])
            add("unanswerable", "outside_process", q_cfg["unanswerable"]["outside_process"]["text"].format(title=title),
                target_hash=None, target_title=title, outside_hash=outside["acordao_hash"],
                absence="the decision is in no case set and its number is in no text of this set")

    both = rows["acordao_resultado"].isin(GRANTED).any() and (~rows["acordao_resultado"].isin(GRANTED)).any()
    for kind in (list(q_cfg["open"]) if every_type else pick("open_by_set")):
        asked = "compare_penalties" if kind == "compare_outcomes" and not both else kind
        if asked in {q["subtype"] for q in questions if q["type"] == "open"}:
            continue
        add("open", asked, q_cfg["open"][asked], **({"planned_subtype": kind} if asked != kind else {}))
    return questions


def spot_check_sample(questions: list[dict], share: float, seed: int) -> list[str]:
    """Question IDs for the author's check: `share` of the set, at least one question of every type per draw order."""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(share * len(questions)))
    by_type: dict[str, list[str]] = {}
    for q in questions:
        by_type.setdefault(q["type"], []).append(q["question_id"])
    picked = []
    for qtype, ids in by_type.items():                           # proportional to the type's share
        k = max(1, round(n * len(ids) / len(questions)))
        picked += [str(i) for i in rng.choice(sorted(ids), size=min(k, len(ids)), replace=False)]
    return sorted(picked)
