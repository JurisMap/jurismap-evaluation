"""Review of the A1 silver labels by a second model (the author, 2026-10-07).

A model of another family than the system under test (judge A, Grok 4.20 reasoning) checks 100
decisions per label type: the 60 that the author audited and 40 more. How each label is checked:

- legal basis and infringed provision: the reviewer sees the source text and the label and says
  whether the label is right (verification). Labels with a planted error are mixed in as controls,
  so the share of errors the reviewer catches is known. Long reports and votes are cut, so a
  provision the reviewer does not find is asked again with the passages of the full text that cite
  its act.
- penalty type and infraction category: the reviewer assigns the label itself, without seeing the
  existing one (blind), and the two are compared.

Only public CRSNSP text is sent.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from jm_eval.retrieval.labels import PENALTY_TYPES
from jm_eval.retrieval.provisions import norm_of

REVIEW_VERSION = 1
SYSTEM = ("You check labels of decisions of a Brazilian administrative appeals council for insurance (CRSNSP). "
          "Be literal and strict: judge only what the text states. Answer with JSON only.")

BASE_PROMPT = """Two tasks on fields of one CRSNSP decision, in Portuguese.

TASK 1 - legal basis. The field LEGAL BASIS cites the provisions the party was accused of infringing. A parser turned it into the list PARSED PROVISIONS. Check the list against the field.

How the list is written:
- One item per article: <act><number>/<year>:art<article>. Acts: DL = Decreto-Lei, LC = Lei Complementar, LEI = Lei, DEC = Decreto, RES-<issuer> = Resolução (CNSP, CMN, CFC...), CIRC-SUSEP = Circular SUSEP, CARTA-CIRC-SUSEP = Carta-Circular, PORT-<issuer> = Portaria, PRON-CPC = accounting pronouncement (CPC), CC = Código Civil, CPC = Código de Processo Civil, CDC = Código de Defesa do Consumidor, CF = Constituição Federal, CTN, CP, RICRSNSP = the council's internal rules.
- Article level only. Paragraphs, incisos, alíneas, items, annexes and chapters are dropped on purpose: do not report them as missing.
- An act cited without any article appears without ":art".
- An act named only because it amended another one ("com redação dada pela...", "alterada pela...") is left out on purpose.
- Contract clauses, general conditions and a plan's own regulation are left out on purpose.
- Judge the list against the field as written, even where the field itself has a typing or OCR slip. Two-digit years are expanded; a year or issuer the field omits may have been completed.

The list is "correct" when it has every article the field cites, each with the right act, number and year, and nothing else. Otherwise it is "wrong".

TASK 2 - penalty type. From the field PENALTY alone, list every penalty type it states (original or final), choosing from: {penalty_types}. A money amount is a fine ("multa"). Use an empty list if the field states none.

LEGAL BASIS:
{base}

PARSED PROVISIONS:
{provisions}

PENALTY:
{penalty}

Return JSON only:
{{"base_verdict": "correct" or "wrong", "base_missing": [<provisions of the field that the list lacks, as written in the field>], "base_spurious": [<items of the list that the field does not support>], "base_comment": "<one sentence, or empty>", "penalty_types": [<types from the list above>]}}"""

INFRINGED_PROMPT = """Below are the headnote (ementa) and the rapporteur's report and vote of one CRSNSP decision, in Portuguese (long texts are cut).

A parser read from this text that the party was accused of infringing this provision:

  {provision}

The provision is written as <act><number>/<year>:art<article> (DL = Decreto-Lei, LC = Lei Complementar, RES-CNSP = Resolução CNSP, CIRC-SUSEP = Circular SUSEP, CC = Código Civil, and so on); paragraphs and incisos are dropped on purpose.

Does the text state that the party infringed, or was accused of infringing, this article of this act?
- "yes": the text says so (the provision need not be the only one infringed).
- "no": the text cites it for something else (the penalty, procedure, a precedent, an argument), cites a different article or act, or does not cite it.
- "not_in_text": the passage that would decide it seems to lie in a part that was cut.

Return JSON only:
{{"verdict": "yes" or "no" or "not_in_text", "evidence": "<at most 25 consecutive words copied from the text, or empty>", "comment": "<one sentence, or empty>"}}

EMENTA:
{ementa}

RELATÓRIO:
{report}

VOTO:
{vote}"""

PASSAGES_PROMPT = """Below are passages of the report and vote of one CRSNSP decision, in Portuguese. They are the places of the full text that cite the act of this provision:

  {provision}

The provision is written as <act><number>/<year>:art<article>; paragraphs and incisos are dropped on purpose.

Do the passages state that the party infringed, or was accused of infringing, this article of this act?
- "yes": they say so (the provision need not be the only one infringed).
- "no": they cite it for something else (the penalty, procedure, a precedent), or cite a different article.

Return JSON only:
{{"verdict": "yes" or "no", "evidence": "<at most 25 consecutive words copied from the passages, or empty>", "comment": "<one sentence, or empty>"}}

PASSAGES:
{passages}"""

# How the text names an act that the provision IDs write as a code.
CODE_PATTERNS = {"CC": r"c[óo]digo\s+civil|\bCC\b|10\.?406", "CPC": r"processo\s+civil|\bN?CPC\b", "CF": r"constitui[çc][ãa]o|\bCF\b",
                 "CDC": r"defesa\s+do\s+consumidor|\bCDC\b|8\.?078", "CTN": r"tribut[áa]rio\s+nacional|\bCTN\b",
                 "CP": r"c[óo]digo\s+penal", "RICRSNSP": r"regimento\s+interno|RI-?CRSNSP"}


def act_passages(provision: str, *texts: str | None, window: int = 500, limit: int = 5) -> list[str]:
    """Passages of the texts that cite the act of `provision` (its number with its year, or the code's name)."""
    norm = norm_of(provision)
    if norm in CODE_PATTERNS:
        pattern = CODE_PATTERNS[norm]
    else:
        m = re.search(r"(\d+)(?:/(\d{4}))?$", norm)
        if not m:
            return []
        number = r"\.?".join(m.group(1)) if len(m.group(1)) > 3 else m.group(1)        # "3308" also as "3.308"
        year = rf"(?:{m.group(2)}|{m.group(2)[2:]})" if m.group(2) else r"\d{2,4}"
        pattern = rf"(?<!\d){number}\s*(?:/|,?\s+de\s+(?:\d{{1,2}}[º°]?\s+de\s+\w+\s+de\s+)?)\s*{year}(?!\d)"
    passages = []
    for text in texts:
        text, last_end = _clean(text), -1
        for hit in re.finditer(pattern, text, flags=re.IGNORECASE):
            if hit.start() < last_end:                              # inside the previous passage
                continue
            start, last_end = max(0, hit.start() - window), min(len(text), hit.end() + window)
            passages.append(text[start:last_end])
            if len(passages) == limit:
                return passages
    return passages


def passages_prompt(provision: str, passages: list[str]) -> str:
    return PASSAGES_PROMPT.format(provision=provision, passages="\n\n".join(f"[{i}] ...{p}..." for i, p in enumerate(passages, 1)))


def validate_passages(answer: object) -> tuple[dict | None, str | None]:
    if not isinstance(answer, dict) or answer.get("verdict") not in ("yes", "no"):
        return None, "verdict is not 'yes' or 'no'"
    return {"verdict": answer["verdict"], "evidence": str(answer.get("evidence") or ""),
            "comment": str(answer.get("comment") or "")}, None


PENALTY_NAMES = {"multa": "multa", "advertencia": "advertência", "recomendacao": "recomendação",
                 "inabilitacao": "inabilitação", "suspensao": "suspensão",
                 "cancelamento_registro": "cancelamento de registro", "destituicao": "destituição"}
assert list(PENALTY_NAMES) == [name for name, _ in PENALTY_TYPES]


def _clean(text) -> str:
    return re.sub(r"[ \t]+", " ", text if isinstance(text, str) else "").strip()


def base_prompt(base: str | None, provisions: list[str], penalty: str | None) -> str:
    return BASE_PROMPT.format(penalty_types=", ".join(f'"{v}"' for v in PENALTY_NAMES.values()),
                              base=_clean(base) or "(empty)", provisions="; ".join(provisions) or "(none)",
                              penalty=_clean(penalty) or "(empty)")


def infringed_prompt(provision: str, ementa: str | None, report: str | None, vote: str | None,
                     report_chars: int = 12000, vote_chars: int = 6000) -> str:
    return INFRINGED_PROMPT.format(provision=provision, ementa=_clean(ementa) or "(sem ementa)",
                                   report=_clean(report)[:report_chars] or "(sem relatório)",
                                   vote=_clean(vote)[:vote_chars] or "(sem voto)")


def validate_base(answer: object) -> tuple[dict | None, str | None]:
    if not isinstance(answer, dict) or answer.get("base_verdict") not in ("correct", "wrong"):
        return None, "base_verdict is not 'correct' or 'wrong'"
    names = {v: k for k, v in PENALTY_NAMES.items()} | {k: k for k in PENALTY_NAMES}
    types = answer.get("penalty_types")
    if not isinstance(types, list) or any(str(t).strip().lower() not in names for t in types):
        return None, "penalty_types holds a value outside the list"
    as_list = lambda v: [str(x) for x in v] if isinstance(v, list) else []  # noqa: E731
    order = list(PENALTY_NAMES)
    return {"base_verdict": answer["base_verdict"], "base_missing": as_list(answer.get("base_missing")),
            "base_spurious": as_list(answer.get("base_spurious")), "base_comment": str(answer.get("base_comment") or ""),
            "penalty_types": sorted({names[str(t).strip().lower()] for t in types}, key=order.index)}, None


def validate_infringed(answer: object) -> tuple[dict | None, str | None]:
    if not isinstance(answer, dict) or answer.get("verdict") not in ("yes", "no", "not_in_text"):
        return None, "verdict is not 'yes', 'no' or 'not_in_text'"
    return {"verdict": answer["verdict"], "evidence": str(answer.get("evidence") or ""),
            "comment": str(answer.get("comment") or "")}, None


# ---------------------------------------------------------------------------- sample and controls

def review_sample(labels: pd.DataFrame, audited: list[str], excluded: set[str], n: int, seed: int) -> list[str]:
    """The audited decisions plus random ones up to `n`, drawn outside the development material."""
    pool = sorted(set(labels["acordao_hash"]) - set(audited) - set(excluded))
    extra = max(0, n - len(audited))
    picks = np.random.default_rng(seed).choice(len(pool), size=min(extra, len(pool)), replace=False)
    return list(audited) + [pool[i] for i in sorted(picks)]


def corrupt_base(provisions: list[str], rng: np.random.Generator) -> tuple[list[str], str]:
    """A legal-basis label with one planted error: an article dropped, or an article number changed."""
    articles = [i for i, p in enumerate(provisions) if ":art" in p]
    i = articles[int(rng.integers(len(articles)))]
    if len(provisions) > 1 and rng.random() < 0.5:
        return provisions[:i] + provisions[i + 1:], f"dropped {provisions[i]}"
    norm, article = provisions[i].split(":art")
    number = int(re.match(r"\d+", article).group())
    wrong = number + 10
    while f"{norm}:art{wrong}" in provisions:
        wrong += 1
    changed = provisions.copy()
    changed[i] = f"{norm}:art{wrong}"
    return changed, f"changed {provisions[i]} to {changed[i]}"


def controls(labels: pd.DataFrame, sample: list[str], excluded: set[str], n_base: int, n_infringed: int,
             seed: int) -> pd.DataFrame:
    """Decisions outside the sample whose label gets a planted error (the reviewer is not told).

    Columns: acordao_hash, kind ('base' | 'infringed'), label (the corrupted label), planted.
    """
    rng = np.random.default_rng(seed)
    pool = labels[~labels["acordao_hash"].isin(set(sample) | set(excluded))].sort_values("acordao_hash")
    rows = []
    with_articles = pool[pool["base_provisions"].apply(lambda p: any(":art" in x for x in p))]
    for i in sorted(rng.choice(len(with_articles), size=min(n_base, len(with_articles)), replace=False)):
        r = with_articles.iloc[i]
        label, planted = corrupt_base(list(r["base_provisions"]), rng)
        rows.append({"acordao_hash": r["acordao_hash"], "kind": "base", "label": label, "planted": planted})
    with_primary = pool[pool["text_primary"].notna() & ~pool["acordao_hash"].isin({r["acordao_hash"] for r in rows})]
    donors = sorted(set(with_primary["text_primary"]))
    for i in sorted(rng.choice(len(with_primary), size=min(n_infringed, len(with_primary)), replace=False)):
        r = with_primary.iloc[i]
        own_norms = {norm_of(p) for p in [*r["text_cited"], *r["base_provisions"], r["text_primary"]]}
        foreign = [p for p in donors if norm_of(p) not in own_norms]     # an act the decision never cites
        wrong = foreign[int(rng.integers(len(foreign)))]
        rows.append({"acordao_hash": r["acordao_hash"], "kind": "infringed", "label": wrong,
                     "planted": f"replaced {r['text_primary']} by {wrong}"})
    return pd.DataFrame(rows, columns=["acordao_hash", "kind", "label", "planted"])


# ---------------------------------------------------------------------------- agreement

def agreement(a: pd.Series, b: pd.Series) -> dict:
    """Agreement of two label series on the rows where both have a value: share equal and Cohen's kappa."""
    from sklearn.metrics import cohen_kappa_score

    both = a.notna() & b.notna()
    x, y = a[both].astype(str), b[both].astype(str)
    if len(x) == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    same = float((x.to_numpy() == y.to_numpy()).mean())
    kappa = float(cohen_kappa_score(x, y)) if len(set(x) | set(y)) > 1 else float("nan")
    return {"n": int(len(x)), "agreement": round(same, 4), "kappa": None if kappa != kappa else round(kappa, 4)}
