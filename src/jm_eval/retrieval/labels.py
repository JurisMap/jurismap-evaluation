"""Silver labels for the A1 retrieval benchmark (Phase 4a), from fields the corpus already has.

- Legal basis: `acordao_base` parsed into provision IDs (jm_eval.retrieval.provisions). The key
  that decides relevance drops the generic provisions and the sanction regulations listed in
  config/a1_labels.yaml.
- Infringed provisions read from the report and the vote: the citations that follow a phrase such
  as "infringindo", "infração ao" or "dispositivo infringido", up to the point where the text turns
  to the penalty. A secondary label, and the fallback where `acordao_base` gives nothing.
- Penalty type from `acordao_penalidade`, outcome from `acordao_resultado`.
- Process groups: decisions of the same process (same process or appeal number in the title),
  which the leave-one-out protocol excludes from each other's rankings.
- Relevance matrices: identical key (strict) or Jaccard of the provision sets (relaxed).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np

from jm_eval.io import load_config
from jm_eval.retrieval.provisions import norm_of, parse

CONFIG = "a1_labels"

# e.g. "15414.604792/2016-77", "15414.604792 2016-77", "15414-621879-2017-90"
PROC_RE = re.compile(r"(?<![\d.])(\d{4,5})[.\-](\d{6,7})\s*[/ \-]?\s*(\d{2,4})\s*-\s*(\d{2})(?!\d)")
RECURSO_RE = re.compile(r"Rec(?:ur)?so\s*(?:n[ºo°.]*\s*)?(\d[\d.]*)", re.IGNORECASE)

PENALTY_TYPES = [
    ("multa", r"multa|R\$"),
    ("advertencia", r"advert[êe]ncia"),
    ("recomendacao", r"recomenda[çc][ãa]o"),
    ("inabilitacao", r"inabilita"),
    ("suspensao", r"suspens[ãa]o"),
    ("cancelamento_registro", r"cancelamento|cassa[çc][ãa]o"),
    ("destituicao", r"destitui"),
]
_PENALTY_RES = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in PENALTY_TYPES]

OUTCOMES = {"provido": "provido", "parcialmente provido": "parcialmente_provido",
            "não provido": "nao_provido", "não conhecido": "nao_conhecido", "extinto": "extinto"}

# Phrases that introduce the provision a party is accused of infringing.
INFRINGE_RE = re.compile(
    r"infring\w+|infra[çc][ãa]o\s+(?:a|à|ao|aos|às)\b|infra[çc][õo]es\s+(?:a|à|ao|aos|às)\b"
    r"|contrariando|em\s+afronta\s+a|desrespeito\s+a|descumprimento\s+(?:a|à|ao|aos|às|d[oa]s?)\b"
    r"|descumprindo|inobserv[âa]ncia\s+(?:a|à|ao|aos|d[oa]s?)\b|viola[çc][ãa]o\s+(?:a|à|ao|aos|d[oa]s?)\b"
    r"|violando|macula\w*|normas?\s+inobservadas?|em\s+desacordo\s+com", re.IGNORECASE)
# Where the text turns to the penalty (or to another step), the infringed provision has ended.
PENALTY_STOP_RE = re.compile(
    r"penalidade|san[çc][ãa]o|san[çc][õo]es|\bpena\b|\bpenas\b|\bmulta|dosimetria|tipifica[çc][ãa]o"
    r"|enquadra|agravante|atenuante|sujeit[oa]s?\s+[àa]|nos\s+termos|com\s+base\s+n|na\s+forma\s+d",
    re.IGNORECASE)
# A sentence ends at ". " before a capital letter, unless the dot closes an abbreviation of a citation.
_SENTENCE_END_RE = re.compile(
    r"(?<!\bart)(?<!\barts)(?<!\bn)(?<!\bres)(?<!\bcirc)(?<!\binc)(?<!\bc\.c)(?<!\bc)(?<!\bfls)(?<!\bn[ºo°])"
    r"\.\s+(?=[A-ZÀ-Ú])", re.IGNORECASE)
MAX_WINDOW = 400


def settings() -> dict:
    return load_config(CONFIG)


# ---------------------------------------------------------------------------- processes

def process_keys(title: str | None, url: str | None = None) -> set[str]:
    """Process numbers (digits) and the appeal number of a decision's title; the URL if the title has none."""
    title = title if isinstance(title, str) else ""
    keys = {"".join(m) for m in PROC_RE.findall(title)}
    if (m := RECURSO_RE.search(title)):
        keys.add("R" + m.group(1).replace(".", ""))
    if not keys and isinstance(url, str):
        keys = {"".join(m) for m in PROC_RE.findall(url)}
    return keys


def process_groups(keys: Sequence[set[str]]) -> np.ndarray:
    """A group ID per decision: decisions that share any key (directly or in a chain) share the ID."""
    parent = list(range(len(keys)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first: dict[str, int] = {}
    for i, ks in enumerate(keys):
        for k in ks:
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    roots = [find(i) for i in range(len(keys))]
    ids = {root: n for n, root in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots], dtype=np.int64)


# ---------------------------------------------------------------------------- field labels

def penalty_types(text: str | None) -> list[str]:
    """Penalty types named in `acordao_penalidade`, in the order of `PENALTY_TYPES`."""
    text = text if isinstance(text, str) else ""
    return [name for name, pattern in _PENALTY_RES if pattern.search(text)]


def outcome(text: str | None) -> str | None:
    return OUTCOMES.get(text.strip().lower()) if isinstance(text, str) else None


def specific(ids: Iterable[str], generic: Iterable[str], sanction_norms: Iterable[str]) -> list[str]:
    """`ids` without the generic provisions and the articles of the sanction regulations."""
    generic, sanction_norms = set(generic), set(sanction_norms)
    return [i for i in ids if i not in generic and norm_of(i) not in sanction_norms]


def key_of(ids: Iterable[str]) -> str | None:
    """Order-free key of a provision set; None for an empty set."""
    return "|".join(sorted(set(ids))) or None


# ---------------------------------------------------------------------------- report and vote

def infringed_provisions(text: str | None, canon: dict | None = None) -> list[str]:
    """Provisions cited as infringed in `text`, most often cited first (ties: first cited first).

    For every phrase of `INFRINGE_RE`, the citations up to the end of the sentence, to the first
    phrase of `PENALTY_STOP_RE` or to `MAX_WINDOW` characters are taken.
    """
    if not isinstance(text, str):
        return []
    text = re.sub(r"\s+", " ", text)
    counts: Counter = Counter()
    order: dict[str, int] = {}
    covered = 0                                     # windows may overlap: count each citation once
    for m in INFRINGE_RE.finditer(text):
        start = max(m.end(), covered)
        window = text[start:m.end() + MAX_WINDOW]
        cuts = [len(window)]
        for stop in (PENALTY_STOP_RE, _SENTENCE_END_RE):
            if (s := stop.search(window)):
                cuts.append(s.start())
        window = window[:min(cuts)]
        covered = start + len(window)
        for p in parse(window, canon):
            if p.norm and not p.amending and p.article:
                counts[p.id] += 1
                order.setdefault(p.id, start + p.start)
    return sorted(counts, key=lambda i: (-counts[i], order[i]))


def cited_provisions(text: str | None, canon: dict | None = None) -> list[str]:
    """Every article-level provision cited in `text`, in order of first appearance."""
    ids: dict[str, None] = {}
    for p in parse(text, canon):
        if p.norm and p.article and not p.amending:
            ids.setdefault(p.id)
    return list(ids)


# ---------------------------------------------------------------------------- relevance

def _incidence(sets: Sequence[Iterable[str]]) -> np.ndarray:
    vocabulary = {v: j for j, v in enumerate(sorted({v for s in sets for v in s}))}
    m = np.zeros((len(sets), max(len(vocabulary), 1)), dtype=np.float32)
    for i, s in enumerate(sets):
        for v in s:
            m[i, vocabulary[v]] = 1.0
    return m


def jaccard_matrix(sets: Sequence[Iterable[str]]) -> np.ndarray:
    """Pairwise Jaccard similarity of the sets (0 where both are empty)."""
    m = _incidence(sets)
    inter = m @ m.T
    size = m.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def relevance_matrix(sets: Sequence[Iterable[str]], groups: Sequence[int], *, kind: str = "exact",
                     threshold: float = 0.5) -> np.ndarray:
    """Boolean matrix R[i, j]: decision j is relevant to query i.

    kind "exact": identical non-empty sets; "jaccard": Jaccard >= threshold; "overlap": at least one
    element in common. A decision is never
    relevant to itself or to a decision of the same process group.
    """
    sets = [set(s) if hasattr(s, "__iter__") else set() for s in sets]      # a missing label is an empty set
    if kind == "exact":
        keys = [key_of(s) for s in sets]
        codes = {k: n for n, k in enumerate(dict.fromkeys(k for k in keys if k))}
        code = np.array([codes.get(k, -1) for k in keys])
        rel = (code[:, None] == code[None, :]) & (code[:, None] >= 0)
    elif kind == "overlap":                          # at least one element in common
        rel = jaccard_matrix(sets) > 0
    elif kind == "jaccard":
        rel = jaccard_matrix(sets) >= threshold
    else:
        raise ValueError(f"unknown relevance kind: {kind}")
    groups = np.asarray(groups)
    rel &= groups[:, None] != groups[None, :]       # also clears the diagonal
    return rel


def build_relevance(labels, spec: dict, threshold: float) -> np.ndarray:
    """Relevance matrix of one definition of config/a1_methods.yaml (`relevance`) over the rows of `labels`.

    spec: {source: <labels column>, kind: exact | jaccard | overlap | same, or_same: <column>}. "same" compares
    a single value per decision (the infraction category); `or_same` adds the pairs that share the
    value of another column (R-prov: the same primary infringed provision).
    """
    groups = labels["process_group"].to_numpy()
    other_process = groups[:, None] != groups[None, :]
    if spec["kind"] == "same":
        rel = _same_value(labels[spec["source"]].to_numpy()) & other_process
    else:
        rel = relevance_matrix(labels[spec["source"]], groups, kind=spec["kind"], threshold=threshold)
    if spec.get("or_same"):
        rel |= _same_value(labels[spec["or_same"]].to_numpy()) & other_process
    return rel


def _same_value(values: np.ndarray) -> np.ndarray:
    """same[i, j]: both decisions have the value and it is equal (missing values never match)."""
    known = np.array([v is not None and v == v for v in values])
    codes = np.full(len(values), -1)
    if known.any():
        codes[known] = np.unique(values[known].astype(str), return_inverse=True)[1]
    return (codes[:, None] == codes[None, :]) & known[:, None]


def relevance_summary(rel: np.ndarray) -> dict:
    """Coverage of a relevance matrix: queries with a relevant decision, relevant decisions per query."""
    per_query = rel.sum(axis=1)
    has = per_query > 0
    return {
        "decisions": int(rel.shape[0]),
        "queries_with_relevant": int(has.sum()),
        "share_queries_with_relevant": round(float(has.mean()), 4) if rel.shape[0] else None,
        "relevant_per_query_median": float(np.median(per_query[has])) if has.any() else None,
        "relevant_per_query_mean": round(float(per_query[has].mean()), 2) if has.any() else None,
        "relevant_pairs_share": round(float(rel.mean()), 5) if rel.size else None,
    }


def group_sizes(groups: Sequence[int]) -> dict:
    sizes = Counter(Counter(groups).values())
    by_size: dict[int, int] = defaultdict(int)
    for size, n in sizes.items():
        by_size[size] += n
    return {"groups": len(set(groups)), "groups_with_several_decisions": sum(n for s, n in by_size.items() if s > 1),
            "decisions_in_those_groups": sum(s * n for s, n in by_size.items() if s > 1),
            "largest_group": max(by_size) if by_size else 0}
