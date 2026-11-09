"""Planted controls for the claim judges of the A3 evaluation: claims whose truth is known.

The judges' labels cannot be checked against a gold standard on real answers, so claims are built
from the decisions themselves and judged with the same prompt and the same evidence retrieval:

  field, true      a field of the decision stated as it is (rapporteur, outcome, appellant, session date)
  field, false     the same statement with the value of another decision of the set
  sentence, true   a sentence copied from the report or the vote of the decision
  sentence, false  such a sentence with one figure changed (a money amount, a percentage, a duration, a year)

A true control must be labelled "supported"; a false one must not, and "contradicted" is the exact
label, since the evidence states the other value. A control is kept only if its evidence passages
hold what decides it (the decision's own fields, or the copied sentence), so the judges are
measured and not the retrieval.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from jm_eval.chatbot import scoring as S
from jm_eval.chatbot.judging import field_text
from jm_eval.chatbot.questions import TITLE_RE, as_list, clean_title

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-ZÀ-Ú])")
MONEY_RE = re.compile(r"R\$\s*(\d{1,3}(?:\.\d{3})*),(\d{2})")
PERCENT_RE = re.compile(r"\b(\d{1,3})\s*%")
DAYS_RE = re.compile(r"\b(\d{1,3}) (dias|meses|anos)\b")
YEAR_RE = re.compile(r"\bde ((?:19|20)\d{2})\b")
OTHER_OUTCOME = {"Não provido": "Provido", "Provido": "Não provido", "Parcialmente provido": "Não provido"}
FIELD_TEMPLATES = {
    "rapporteur": "No {title}, o Conselheiro Relator foi {value}.",
    "outcome": "No {title}, o resultado do julgamento do recurso foi \"{value}\".",
    "appellant": "No {title}, figura como recorrente {value}.",
    "session_date": "A sessão de julgamento do {title} foi realizada em {value}.",
}


def _field_value(kind: str, row: dict) -> str | None:
    if kind == "rapporteur":
        return (as_list(row["acordao_relator"]) or [None])[0]
    if kind == "appellant":
        return (as_list(row["acordao_recorrente"]) or [None])[0]
    if kind == "outcome":
        return row["acordao_resultado"] if row["acordao_resultado"] in OTHER_OUTCOME else None
    gold = S.parse_session_date(row["sessao_datas"], row["ano_sessoes"])
    return f"{field_text(row['sessao_datas'])} de {int(row['ano_sessoes'])}" if gold else None


def field_controls(rows: pd.DataFrame, rng, n: int) -> list[dict]:
    """`n` field statements, alternately true and false, over random (decision, field) pairs."""
    records = [r for r in rows.to_dict("records") if TITLE_RE.match(clean_title(r["acordao_titulo"]))]
    values = {(r["acordao_hash"], kind): _field_value(kind, r) for r in records for kind in FIELD_TEMPLATES}
    pairs = [(r, kind) for r in records for kind in FIELD_TEMPLATES if values[(r["acordao_hash"], kind)]]
    out = []
    for i in rng.permutation(len(pairs)):
        row, kind = pairs[int(i)]
        value, truth = values[(row["acordao_hash"], kind)], len(out) % 2 == 0
        if not truth:
            if kind == "outcome":
                value = OTHER_OUTCOME[value]
            else:
                own = {S.name_key(v) for v in (as_list(row["acordao_relator"]) + as_list(row["acordao_recorrente"]) + [value])}
                others = sorted({v for (h, k), v in values.items() if k == kind and v and S.name_key(v) not in own})
                if not others:
                    continue
                value = others[int(rng.integers(len(others)))]
        title = clean_title(row["acordao_titulo"])
        out.append({"acordao_hash": row["acordao_hash"], "kind": f"field/{kind}", "truth": truth,
                    "claim": FIELD_TEMPLATES[kind].format(title=title, value=value), "source": "dados do caso"})
        if len(out) == n:
            break
    return out


def sentences(text: str | None, low: int = 12, high: int = 30) -> list[str]:
    """Sentences of `low` to `high` words: short enough to lie whole inside one evidence passage."""
    parts = SENTENCE_SPLIT_RE.split(field_text(text))
    return [p.strip() for p in parts if low <= len(p.split()) <= high]


def alter_figure(sentence: str) -> str | None:
    """The sentence with its figure changed, if it has exactly one money amount, percentage, duration or year."""
    hits = [(m, kind) for kind, pattern in (("money", MONEY_RE), ("percent", PERCENT_RE), ("days", DAYS_RE), ("year", YEAR_RE))
            for m in pattern.finditer(sentence)]
    if len(hits) != 1:
        return None
    m, kind = hits[0]
    if kind == "money":
        amount = int(m.group(1).replace(".", "")) * 3
        new = f"R$ {amount:,}".replace(",", ".") + f",{m.group(2)}"
    elif kind == "percent":
        new = f"{int(m.group(1)) + 25}%"
    elif kind == "year":
        new = f"de {int(m.group(1)) - 3}"
    else:
        new = f"{int(m.group(1)) + 15} {m.group(2)}"
    return sentence[:m.start()] + new + sentence[m.end():]


def sentence_controls(rows: pd.DataFrame, rng, n_true: int, n_false: int) -> list[dict]:
    """Copied sentences (true) and copied sentences with one figure changed (false)."""
    pool = []
    for r in rows.to_dict("records"):
        title = clean_title(r["acordao_titulo"])
        if not TITLE_RE.match(title):
            continue
        for field in ("acordao_relatorio", "acordao_voto_relator"):
            pool += [(r["acordao_hash"], title, s) for s in sentences(r[field])]
    order = [pool[int(i)] for i in rng.permutation(len(pool))]
    out, used = [], set()
    for truth, wanted in ((False, n_false), (True, n_true)):
        taken = 0
        for h, title, sentence in order:
            if taken == wanted:
                break
            if sentence in used:
                continue
            text = sentence if truth else alter_figure(sentence)
            if text is None:
                continue
            used.add(sentence)
            taken += 1
            out.append({"acordao_hash": h, "kind": "sentence", "truth": truth, "claim": f"No {title}: {text}",
                        "source": sentence})
    return out


def evidence_holds_source(control: dict, evidence: list[str], title: str) -> bool:
    """The evidence passages contain what decides the control."""
    if control["source"] == "dados do caso":
        return any(e.startswith(f"[{title} | dados do caso]") for e in evidence)
    joined = " ".join(" ".join(evidence).split())
    return " ".join(control["source"].split()) in joined


def score(controls: pd.DataFrame, rater: str) -> dict:
    """How a rater labelled the controls it judged."""
    judged = controls[controls[rater].isin(["supported", "contradicted", "not_found"])]
    true, false = judged[judged["truth"]], judged[~judged["truth"]]
    return {"true_controls": len(true), "true_accepted": int((true[rater] == "supported").sum()),
            "true_called_not_found": int((true[rater] == "not_found").sum()),
            "true_called_contradicted": int((true[rater] == "contradicted").sum()),
            "false_controls": len(false), "false_rejected": int((false[rater] != "supported").sum()),
            "false_called_contradicted": int((false[rater] == "contradicted").sum()),
            "false_accepted": int((false[rater] == "supported").sum())}
