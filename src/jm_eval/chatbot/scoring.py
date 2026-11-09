"""Deterministic scoring of chatbot answers (Phase 6, task 4): no judge is involved.

- Templated questions: the gold value comes from the decision's fields. The answer is a sentence,
  so "exact match" means that the normalised gold value is stated in the normalised answer:
  names and outcomes by normalised match, dates by day, month and year, money amounts within a
  tolerance, counts as numbers, lists (appellants, lawyers, provisions) by the share of gold items
  the answer states.
- Unanswerable questions: abstention by a closed list of patterns (config/a3_chatbot.yaml).
- Citation check: every process number in an answer must be one of the supplied decisions.

Every scorer returns {"score": 0..1, "correct": bool, ...details}; `correct` means score == 1.
"""

from __future__ import annotations

import re
import unicodedata

from jm_eval.retrieval.labels import PROC_RE, RECURSO_RE, penalty_types
from jm_eval.retrieval.provisions import provision_ids

MONTHS = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6, "julho": 7,
          "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12}
NUMBER_WORDS = {"nenhum": 0, "nenhuma": 0, "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3,
                "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
                # up to the largest context of Phase 7 (30 decisions); both Brazilian and European spellings
                "onze": 11, "doze": 12, "treze": 13, "quatorze": 14, "catorze": 14, "quinze": 15,
                "dezesseis": 16, "dezasseis": 16, "dezessete": 17, "dezassete": 17, "dezoito": 18,
                "dezenove": 19, "dezanove": 19, "vinte": 20, "trinta": 30}
UNITS = {w: v for w, v in NUMBER_WORDS.items() if 1 <= v <= 9}
COMPOUND_RE = re.compile(rf"\b(vinte|trinta)\s+e\s+({'|'.join(UNITS)})\b")      # "vinte e dois" is 22, not 20 and 2
NUMBER_WORD_ALT = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
# an amount written in words ("dez mil reais", "(cinco por cento)") is not a count
AMOUNT_WORDS_RE = re.compile(rf"\([^)]*\breais\b[^)]*\)|\b(?:{NUMBER_WORD_ALT})(?:\s+e\s+(?:{NUMBER_WORD_ALT}))*"
                             r"\s+(?:mil|milhao|milhoes|reais|por\s+cento)\b")
# Outcome categories, most specific first: a match is blanked before the next pattern is tried.
OUTCOME_PATTERNS = [
    ("parcialmente_provido", r"parcial(?:mente)?\s+provi\w+|provi\w+\s+(?:em\s+parte|parcial\w*)"
                             r"|provimento\s+parcial|parcial\s+provimento"),
    ("nao_provido", r"nao\s+(?:foi\s+|foram\s+|sendo\s+|teve\s+)?provi\w+"
                    r"|neg\w+(?:-lhes?|-se)?\s+(?:o\s+)?provimento|provimento\s+(?:foi\s+)?negado"
                    r"|nao\s+(?:lhes?\s+)?(?:dar|deu|deram|dado)(?:-lhes?)?\s+provimento|desprovi\w+|improvi\w+"),
    ("nao_conhecido", r"nao\s+(?:foi\s+|foram\s+|sendo\s+)?conhec\w+|nao\s+conhecimento"),
    ("extinto", r"extint\w+|extincao"),
    ("provido", r"\bprovid[oa]s?\b|\b(?:deu|dado|dar|dando|deram|dada)(?:-lhes?|-se)?\s+provimento"
                r"|provimento\s+(?:ao|do|aos|dos)\s+recursos?"),
]
MONEY_RE = re.compile(r"r\$\s*(\d[\d.]*(?:,\d{1,2})?)|(\d[\d.]*(?:,\d{1,2})?)\s+reais")
CNPJ_RE = re.compile(r"(?<![\d.])\d{2}\.?\d{3}\.?\d{3}\s*/\s*\d{4}\s*-?\s*\d{2}(?!\d)")
DATE_SPAN_RE = re.compile(
    r"(\d{1,2})\s*(?:o|º|°)?((?:\s*(?:,|e|a|ate)\s*\d{1,2}\s*(?:o|º|°)?)*)\s+de\s+(" + "|".join(MONTHS) + r")"
    r"(?:\s+de\s+(\d{4}))?")
NUMERIC_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
# e.g. "ACÓRDÃO CRSNSP 7334/2021", "Acórdão nº 602/2024": the council's own number of a decision
ACORDAO_RE = re.compile(r"ac[óo]rd[ãa]o\s+(?:crsnsp\s+)?(?:n[ºo°.]*\s*)?(\d{1,5})\s*/\s*(\d{4})", re.IGNORECASE)
CLEAN_SESSION_DATE_RE = re.compile(r"\d{1,2}(?:\s*(?:,|e)\s*\d{1,2})*\s+de\s+(?:" + "|".join(MONTHS) + r")")


def normalise(text: str | None) -> str:
    """Lower case, no accents, single spaces."""
    text = unicodedata.normalize("NFKD", text if isinstance(text, str) else "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", text).strip()


def name_key(text: str | None) -> str:
    """A name or a sentence reduced to its words: punctuation dropped, "S.A." and "S/A" written "sa"."""
    text = re.sub(r"\bs\s?[./]\s?a\b\.?", "sa", normalise(text))
    return " ".join(re.findall(r"[0-9a-z]+", text))


def states_name(answer: str, name: str) -> bool:
    key = name_key(name)
    return bool(key) and f" {key} " in f" {name_key(answer)} "


def _result(score: float, **details) -> dict:
    return {"score": round(float(score), 4), "correct": score >= 1.0, **details}


def score_names(answer: str, gold: list[str]) -> dict:
    """Share of the gold names that the answer states in full."""
    found = [g for g in gold if states_name(answer, g)]
    return _result(len(found) / len(gold) if gold else 0.0, gold_items=len(gold), found_items=len(found),
                   missing=[g for g in gold if g not in found])


def outcomes_stated(answer: str) -> list[str]:
    text, found = normalise(answer), []
    for name, pattern in OUTCOME_PATTERNS:
        text, n = re.subn(pattern, " ", text)
        if n:
            found.append(name)
    return found


def score_outcome(answer: str, gold: str) -> dict:
    """Correct when the gold outcome is the only outcome the answer states."""
    stated = outcomes_stated(answer)
    return _result(1.0 if stated == [gold] else 0.0, stated=stated, ambiguous=gold in stated and len(stated) > 1)


def dates_stated(answer: str) -> list[dict]:
    """Dates written as "25 e 26 de agosto de 2021" or "25/08/2021": [{"days", "month", "year"}]."""
    text, out = normalise(answer), []
    for m in DATE_SPAN_RE.finditer(text):
        days = [int(d) for d in re.findall(r"\d{1,2}", m.group(1) + " " + (m.group(2) or ""))]
        out.append({"days": days, "month": MONTHS[m.group(3)], "year": int(m.group(4)) if m.group(4) else None})
    for m in NUMERIC_DATE_RE.finditer(text):
        out.append({"days": [int(m.group(1))], "month": int(m.group(2)), "year": int(m.group(3))})
    return out


def parse_session_date(sessao_datas: str | None, year: int) -> dict | None:
    """Gold days, month and year of a session written as "25 e 26 de agosto"; None for any other form."""
    text = normalise(sessao_datas)
    if not CLEAN_SESSION_DATE_RE.fullmatch(text):
        return None
    stated = dates_stated(text)
    return {"days": stated[0]["days"], "month": stated[0]["month"], "year": int(year)} if len(stated) == 1 else None


def score_session_date(answer: str, gold: dict) -> dict:
    """Correct when the answer states the month, every day and the year of the session.

    The days of one month may be written together or as separate dates; a date written without its
    year counts if the year is stated elsewhere in the answer.
    """
    stated = dates_stated(answer)
    year_elsewhere = re.search(rf"\b{gold['year']}\b", normalise(answer)) is not None
    days = {d for s in stated if s["month"] == gold["month"]
            and (s["year"] == gold["year"] or (s["year"] is None and year_elsewhere)) for d in s["days"]}
    found = [d for d in gold["days"] if d in days]
    return _result(len(found) / len(gold["days"]), stated=stated)


def money_amounts(text: str | None) -> list[float]:
    values = []
    for m in MONEY_RE.finditer(normalise(text)):
        raw = (m.group(1) or m.group(2)).rstrip(".")
        try:
            values.append(float(raw.replace(".", "").replace(",", ".")))
        except ValueError:
            continue
    return values


def parse_penalty(text: str | None) -> dict:
    return {"types": penalty_types(text), "amounts": sorted(set(money_amounts(text)))}


def score_penalty(answer: str, gold: dict, tolerance: float = 0.005) -> dict:
    """Share of the gold penalty types and amounts that the answer states."""
    types, amounts = penalty_types(answer), money_amounts(answer)
    type_hits = [t for t in gold["types"] if t in types]
    amount_hits = [a for a in gold["amounts"] if any(abs(a - b) <= tolerance * max(a, 1.0) for b in amounts)]
    total = len(gold["types"]) + len(gold["amounts"])
    return _result((len(type_hits) + len(amount_hits)) / total if total else 0.0,
                   stated_types=types, stated_amounts=amounts)


def score_provisions(answer: str, gold: list[str], canon: dict | None = None) -> dict:
    """Share of the gold provisions that the answer cites (same parser as the Phase 4a labels)."""
    # the parser reads "nº" and "n.", as the corpus writes them; an answer may spell the word out
    stated = provision_ids(re.sub(r"\bn[úu]meros?\b", "nº", answer or "", flags=re.IGNORECASE), canon)
    found = [g for g in gold if g in stated]
    return _result(len(found) / len(gold) if gold else 0.0, gold_items=len(gold), found_items=len(found),
                   missing=[g for g in gold if g not in found], extra=[s for s in stated if s not in gold])


def numbers_stated(answer: str) -> list[int]:
    """Counts an answer states: numerals and number words, without identifiers, dates and amounts."""
    text = normalise(answer)
    text = PROC_RE.sub(" ", text)
    text = re.sub(RECURSO_RE.pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:acordao|sessao|processo|art\.?|artigos?|resolucao|circular|lei|decreto(?:-lei)?)\s*"
                  r"(?:crsnsp\s*)?(?:n[o°º.]*\s*)?\d[\d./-]*", " ", text)
    text = DATE_SPAN_RE.sub(" ", NUMERIC_DATE_RE.sub(" ", text))
    text = MONEY_RE.sub(" ", text)
    text = re.sub(r"(?m)^\s*\d+\s*[.)-]\s", " ", text)                       # list markers
    text = re.sub(r"\b\d+\s*(?:o|a|º|ª|°)\b|\b\d{4}\b|\d+[.,/]\d+", " ", text)   # ordinals, years, decimals
    numbers = [int(n) for n in re.findall(r"(?<![\w.])\d{1,3}(?![\w%])", text)]
    text = AMOUNT_WORDS_RE.sub(" ", text)
    numbers += [NUMBER_WORDS[tens] + UNITS[unit] for tens, unit in COMPOUND_RE.findall(text)]
    text = COMPOUND_RE.sub(" ", text)
    numbers += [value for word, value in NUMBER_WORDS.items() if re.search(rf"\b{word}\b", text)]
    return sorted(set(numbers))


def score_count(answer: str, gold: int, set_size: int = 10) -> dict:
    """Correct when the gold count is the only count stated, the size of the case set aside.

    "todos" stands for the set size. An answer that states the gold count next to another number is
    scored 0 and marked `ambiguous`.
    """
    stated = numbers_stated(answer)
    text = normalise(answer)
    if re.search(r"\btod[oa]s\b", text):
        stated = sorted(set(stated) | {set_size})
    if gold == 0 and re.search(r"\bnenhum", text):
        stated = sorted(set(stated) | {0})
    others = [n for n in stated if n != gold and n != set_size]
    # "um"/"uma" is also the indefinite article: alone it is not a competing count
    others = [n for n in others if not (n == 1 and not re.search(r"(?<![\w.])1(?![\w%])", text))]
    ok = gold in stated and not others
    return _result(1.0 if ok else 0.0, stated=stated, ambiguous=gold in stated and bool(others))


def has_cnpj(text: str | None) -> bool:
    """A CNPJ number is written in the text (process numbers, which look alike, are set aside)."""
    return CNPJ_RE.search(PROC_RE.sub(" ", text if isinstance(text, str) else "")) is not None


def abstains(answer: str, patterns: list[str]) -> bool:
    text = normalise(answer)
    return any(re.search(p, text) for p in patterns)


def score_unanswerable(answer: str, kind: str, patterns: list[str]) -> dict:
    """"abstained", "fabricated" (a CNPJ is given where none exists) or "unclassified" (for the judge)."""
    if kind == "appellant_cnpj" and has_cnpj(answer):
        return {"status": "fabricated", "score": 0.0, "correct": False}
    if abstains(answer, patterns):
        return {"status": "abstained", "score": 1.0, "correct": True}
    return {"status": "unclassified", "score": None, "correct": None}


def process_numbers(text: str | None) -> list[str]:
    """Process and appeal numbers written in a text, as the keys of `labels.process_keys`."""
    text = text if isinstance(text, str) else ""
    keys = ["".join(m) for m in PROC_RE.findall(text)]
    keys += ["R" + m.replace(".", "").rstrip(".") for m in RECURSO_RE.findall(text) if m.replace(".", "")]
    return list(dict.fromkeys(keys))


def acordao_numbers(text: str | None) -> list[str]:
    """Acórdão numbers written in a text, as "A<number>/<year>"."""
    found = ACORDAO_RE.findall(text if isinstance(text, str) else "")
    return list(dict.fromkeys(f"A{int(number)}/{year}" for number, year in found))


def own_acordao_number(acordao_text: str | None) -> str | None:
    """The number of a decision: the first acórdão number of its own `acordao_acordao` field."""
    numbers = acordao_numbers(acordao_text)
    return numbers[0] if numbers else None


def decision_identifiers(text: str | None) -> list[str]:
    """Everything a text uses to name a decision: process and appeal numbers, and acórdão numbers."""
    return list(dict.fromkeys(process_numbers(text) + acordao_numbers(text)))


def citation_check(answer: str, supplied: set[str], question: str, context: str) -> dict:
    """Where each decision identifier of an answer comes from (process, appeal and acórdão numbers).

    supplied: one of the decisions of the case set; in_question: named by the user;
    in_context: cited inside the supplied texts (e.g. a precedent in a vote); unknown: in none.
    """
    asked, in_context = set(decision_identifiers(question)), set(decision_identifiers(context))
    kinds = {"supplied": [], "in_question": [], "in_context": [], "unknown": []}
    for key in decision_identifiers(answer):
        kind = ("supplied" if key in supplied else "in_question" if key in asked
                else "in_context" if key in in_context else "unknown")
        kinds[kind].append(key)
    cited = sum(len(v) for v in kinds.values())
    return {"cited": cited, **{k: len(v) for k, v in kinds.items()}, "unknown_numbers": kinds["unknown"],
            "all_supplied": cited == len(kinds["supplied"]) + len(kinds["in_question"]),
            "none_unknown": not kinds["unknown"]}


SCORERS = {"outcome", "rapporteur", "appellant", "lawyer", "session_date", "legal_basis", "penalty", "count"}


def score_templated(question: dict, answer: str, *, canon: dict | None = None, tolerance: float = 0.005) -> dict:
    kind, gold = question["subtype"], question["gold"]
    if kind == "outcome":
        return score_outcome(answer, gold)
    if kind in ("rapporteur", "appellant", "lawyer"):
        return score_names(answer, gold)
    if kind == "session_date":
        return score_session_date(answer, gold)
    if kind == "legal_basis":
        return score_provisions(answer, gold, canon)
    if kind == "penalty":
        return score_penalty(answer, gold, tolerance)
    if kind == "count":
        return score_count(answer, gold, question.get("set_size", 10))
    raise ValueError(f"no scorer for {kind!r}")
