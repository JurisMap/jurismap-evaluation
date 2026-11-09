"""Automatic metrics of the A4 drafts (Phase 8, task 4) that need no model and no outside source.

- Template section coverage: each section of the template is matched to the draft's headings by
  fuzzy comparison of their normalised words (difflib ratio, or every content word of the section
  name found in one heading).
- Precedent citations: process numbers (SUSEP's NNNNN.NNNNNN/YYYY-DD) and CRSNSP ruling numbers
  ("Acórdão CRSNSP 7038/2021") found in the draft, each resolved to the corpus: one of the supplied
  precedents, another decision of the corpus, the source decision, or not found. Ruling numbers are
  read from the dispositives (1,832 of 2,390 carry one), so a ruling not found is "unresolved", while
  a process number not found is not in the corpus.
"""

from __future__ import annotations

import difflib
import re

import pandas as pd

from jm_eval.drafting.notices import normalise
from jm_eval.retrieval.labels import PROC_RE

ROMAN = re.compile(r"^(?:[ivxlc]+|\d+)\s*[-–.)]?\s+", re.I)
STOP = {"da", "do", "das", "dos", "de", "e", "a", "o", "ao", "aos", "as", "os"}
RULING_IN_DRAFT = re.compile(r"ac[óo]rd[ãa]os?\b(?:\s+(?:do\s+)?crsnsp)?(?:\s+n[º°o.]*)?\s*((?:\d{1,2}\.)?\d{3,5}\s*/\s*\d{2,4}"
                             r"(?:\s*(?:,|e)\s*(?:n[º°o.]*\s*)?(?:\d{1,2}\.)?\d{3,5}\s*/\s*\d{2,4})*)", re.I)
RULING_NUMBER = re.compile(r"((?:\d{1,2}\.)?\d{3,5})\s*/\s*(\d{2,4})")
RULING_IN_CORPUS = re.compile(r"AC[ÓO]RD[ÃA]O\s+(?:CRSNSP\s*)?(?:N[º°o.]*\s*)?((?:\d{1,2}\.)?\d{3,5})(?:\s*/\s*(\d{2,4}))?", re.I)


# ---------------------------------------------------------------------------- template sections

def _words(text: str) -> list[str]:
    return [w for w in normalise(ROMAN.sub("", text.strip().strip("#*_ ").strip())).split() if w not in STOP]


def headings(draft: str) -> list[str]:
    """Heading lines of a draft: markdown headings, lines wholly in bold, and short lines in capitals
    that start with a roman or arabic numeral."""
    out = []
    for line in draft.splitlines():
        s = line.strip()
        if not s:
            continue
        bare = s.strip("#*_ ").strip()
        if (s.startswith("#") or (s.startswith("**") and s.rstrip(":").endswith("**") and len(bare) < 120)
                or (ROMAN.match(bare) and len(bare) < 100 and bare.upper() == bare)):
            out.append(bare)
    return out


def section_coverage(draft: str, sections: dict[str, list[str]], threshold: float = 0.8) -> dict:
    """Which template sections have a matching heading in the draft: a heading that contains one of
    the section's names (whole words, normalised), or is close to the section's own name."""
    heads = [" ".join(_words(h)) for h in headings(draft)]
    found = {}
    for section, names in sections.items():
        target = " ".join(_words(section))
        hit = any(re.search(rf"\b{re.escape(normalise(n))}\b", h) for h in heads for n in names)
        hit = hit or any(difflib.SequenceMatcher(None, target, h).ratio() >= threshold for h in heads)
        found[section] = hit
    return {"sections_found": found, "coverage": sum(found.values()) / len(sections) if sections else None}


# ---------------------------------------------------------------------------- citations

def _ruling_key(number: str) -> str:
    return str(int(number.replace(".", "")))


class CorpusIndex:
    """Process keys and ruling numbers of the corpus decisions."""

    def __init__(self, corpus: pd.DataFrame, labels: pd.DataFrame, sources_dir=None):
        """`sources_dir`: the decisions' full texts (<acordao_hash>.txt), whose header gives the ruling
        number of decisions whose dispositive does not (e.g. 15414.626623/2017-79: ACÓRDÃO CRSNSP 7357/2021)."""
        self.by_process: dict[str, set[str]] = {}
        for h, keys in zip(labels["acordao_hash"], labels["process_keys"]):
            for k in keys:
                self.by_process.setdefault(k, set()).add(h)
        self.by_ruling: dict[str, set[str]] = {}
        for h, text in zip(corpus["acordao_hash"], corpus["acordao_acordao"].fillna("").astype(str)):
            if (m := RULING_IN_CORPUS.search(text[:300])):
                self.by_ruling.setdefault(_ruling_key(m.group(1)), set()).add(h)
        if sources_dir is not None:
            from pathlib import Path

            for h in corpus["acordao_hash"]:
                path = Path(sources_dir) / f"{h}.txt"
                if path.exists() and (m := re.search(r"AC[ÓO]RD[ÃA]O\s+CRSNSP\s*(?:N[º°o.]*\s*)?((?:\d{1,2}\.)?\d{3,5})",
                                                     path.read_text(encoding="utf-8", errors="ignore"), re.I)):
                    self.by_ruling.setdefault(_ruling_key(m.group(1)), set()).add(h)

    def resolve(self, kind: str, key: str) -> set[str]:
        return (self.by_process if kind == "process" else self.by_ruling).get(key, set())


def citations(draft: str) -> list[dict]:
    """Process numbers and CRSNSP ruling numbers cited in a draft, each once."""
    out, seen = [], set()
    for m in PROC_RE.finditer(draft):
        key = "".join(m.groups())
        if ("process", key) not in seen:
            seen.add(("process", key))
            out.append({"kind": "process", "key": key, "text": m.group(0), "pos": m.start()})
    for m in RULING_IN_DRAFT.finditer(draft):
        for n in RULING_NUMBER.finditer(m.group(1)):
            key = _ruling_key(n.group(1))
            if ("ruling", key) not in seen:
                seen.add(("ruling", key))
                out.append({"kind": "ruling", "key": key, "text": n.group(0), "pos": m.start(1) + n.start()})
    return out


_PAIR_BETWEEN = re.compile(r"\W*(?:processo|proc\.?)\s*(?:susep\s*|crsnsp\s*)?(?:n[º°o.]*\s*)?", re.I)


def pairs(draft: str, cites: list[dict]) -> list[tuple[dict, dict]]:
    """A ruling number followed directly by its process number: only brackets, dashes or markup and
    "Processo nº" / "Proc." between them ("Acórdão CRSNSP 7357/2021 (Processo nº 15414.626623/2017-79)",
    "Acórdão CRSNSP 6839/2020 - Processo nº ..."). A process followed by the next ruling of a list, or a
    ruling covering several processes, is not a pair."""
    ordered = sorted(cites, key=lambda c: c["pos"])
    out = []
    for a, b in zip(ordered, ordered[1:]):
        if a["kind"] == "ruling" and b["kind"] == "process" \
                and _PAIR_BETWEEN.fullmatch(draft[a["pos"] + len(a["text"]):b["pos"]]):
            out.append((a, b))
    return out


def classify_citations(draft: str, index: CorpusIndex, supplied: list[str], source: str, notice: str = "") -> dict:
    """Each citation as supplied, source, corpus (another decision), from_notice (a number the case
    document states: not a precedent) or not_found; pairs of a ruling and a process cited together
    whose numbers belong to different decisions (or to none) are mismatches; and the counts.

    A cited decision is counted once even when the draft cites both its process and its ruling."""
    rows = []
    notice_digits = re.sub(r"\D", "", notice)
    for c in citations(draft):
        hashes = index.resolve(c["kind"], c["key"])
        if c["kind"] == "process" and not hashes and c["key"] in notice_digits:
            status = "from_notice"
        elif source in hashes:
            status = "source"
        elif hashes & set(supplied):
            status = "supplied"
        elif hashes:
            status = "corpus"
        else:
            status = "not_found"
        rows.append({**c, "status": status, "hashes": sorted(hashes)})
    mismatches = []
    for ruling, process in pairs(draft, rows):
        rh, ph = set(ruling["hashes"]), set(process["hashes"])
        if ph and not (rh & ph):                         # the process is known, the ruling number is not its own
            mismatches.append({"ruling": ruling["text"], "process": process["text"], "ruling_found": bool(rh)})
    supplied_cited = {h for r in rows if r["status"] == "supplied" for h in r["hashes"] if h in set(supplied)}
    corpus_cited = {h for r in rows if r["status"] == "corpus" for h in r["hashes"]} - supplied_cited
    return {
        "citations": rows,
        "n_citations": len(rows),
        "cited_supplied": len(supplied_cited),               # distinct supplied precedents cited
        "cited_corpus_not_supplied": len(corpus_cited),     # distinct other corpus decisions cited
        "cited_source": any(r["status"] == "source" for r in rows),
        "process_not_in_corpus": sum(1 for r in rows if r["status"] == "not_found" and r["kind"] == "process"),
        "ruling_unresolved": sum(1 for r in rows if r["status"] == "not_found" and r["kind"] == "ruling"),
        "supplied_used_share": len(supplied_cited) / len(supplied) if supplied else None,
        "ruling_process_mismatch": len(mismatches), "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------- statute validity

_NUM = r"(?:\d{1,3}\.\d{3}|\d{1,4})"
_ART_ITEM = rf"{_NUM}\s*[º°o]?(?:\s*-\s*[A-Z](?![a-z]))?"
ARTICLE_CITE = re.compile(rf"\bart(?:igo)?s?\s*\.?\s*({_ART_ITEM}(?:\s*(?:,|e)\s*(?:art(?:igo)?s?\.?\s*)?{_ART_ITEM})*)", re.I)
_MONTHS = "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
ACT_CITE = re.compile(
    r"(?P<kind>decreto[- ]lei|\bDL\b|lei\s+complementar|\bLC\b|lei(?:\s+federal)?|resolu[cç][aã]o\s+(?:do\s+)?(?:cnsp|cmn)|res\.\s*(?:cnsp|cmn)"
    r"|circular\s+(?:da\s+)?susep|circ\.\s*susep|decreto)"
    rf"\s*(?:n[º°o.]*\s*)?(?P<num>{_NUM})(?:\s*/\s*(?P<y1>\d{{4}}|\d{{2}})(?!\d)|,?\s*de\s+\d{{1,2}}[º°o]?\s+de\s+(?:{_MONTHS})\s+de\s+(?P<y2>\d{{4}}))?"
    r"|(?P<named>constitui[cç][aã]o\s+(?:da\s+rep[uú]blica|federal)|\bCF(?:/88|/1988)?\b|c[oó]digo\s+civil|\bCC(?:/2002)?\b"
    r"|LINDB|lei\s+de\s+introdu[cç][aã]o|c[oó]digo\s+de\s+defesa\s+do\s+consumidor|\bCDC\b|regimento\s+interno)", re.I)
_KIND_PREFIX = {"decreto-lei": "DL", "decreto lei": "DL", "dl": "DL", "lei complementar": "LC", "lc": "LC", "lei": "L",
                "lei federal": "L", "decreto": "DEC", "res cnsp": "RES-CNSP", "res cmn": "RES-CMN", "circ susep": "CIRC-SUSEP",
                "resolucao cnsp": "RES-CNSP", "resolucao do cnsp": "RES-CNSP", "resolucao cmn": "RES-CMN",
                "resolucao do cmn": "RES-CMN", "circular susep": "CIRC-SUSEP", "circular da susep": "CIRC-SUSEP"}
_NAMED = [(r"constitui|^cf", "CF/1988"), (r"c[oó]digo\s+civil|^cc", "L10406/2002"), (r"lindb|introdu", "DL4657/1942"),
          (r"consumidor|^cdc", "L8078/1990"), (r"regimento", "REGIMENTO-INTERNO")]


def act_key(m: re.Match, known: set[str]) -> str:
    """The config key of a cited act ("DL73/1966"); a number without a year is matched to the one known act of
    that kind and number; an unmatched act keeps its own key and is unknown to the table."""
    if m.group("named"):
        text = normalise(m.group("named"))
        return next(key for pattern, key in _NAMED if re.search(pattern, text))
    prefix = _KIND_PREFIX[re.sub(r"\s+", " ", normalise(m.group("kind")))]
    number = str(int(m.group("num").replace(".", "")))
    year = m.group("y1") or m.group("y2")
    if year and len(year) == 2:
        year = ("19" if int(year) > 30 else "20") + year
    if year:
        return f"{prefix}{number}/{year}"
    same = [k for k in known if k.startswith(f"{prefix}{number}/")]
    return same[0] if len(same) == 1 else f"{prefix}{number}/?"


def statute_citations(draft: str, known: set[str], window: int = 140) -> list[dict]:
    """(act, article) pairs cited in a draft: an article mention tied to the act joined to it right after
    ("art. 133 da Resolução ..."), or else to the act named just before it ("Decreto-Lei nº 73/1966 (arts.
    36 e 88)"). Articles with neither are returned with act None."""
    arts = list(ARTICLE_CITE.finditer(draft))
    acts = list(ACT_CITE.finditer(draft))
    out = []
    for i, a in enumerate(arts):
        stop = min(a.end() + window, arts[i + 1].start() if i + 1 < len(arts) else len(draft))
        after = next((m for m in acts if a.end() <= m.start() < stop), None)
        before = next((m for m in reversed(acts) if 0 <= a.start() - m.end() <= 15), None)
        # 1. an act right after the article, joined by "da/do/de" with no semicolon or bracket between
        #    ("art. 5º, incisos LIV e LV, da Constituição", "art. 133 da Resolução ..."); 2. an act named
        #    just before it with only punctuation or markup between ("Decreto-Lei nº 73/1966 (arts. 36 e 88)",
        #    'Código Civil: > *"Art. 265.'). Otherwise no act: a back-reference ("do mesmo normativo") or a
        #    quotation of a statute named further away is not graded.
        joined = after is not None and after.start() - a.end() <= 60 and re.fullmatch(
            r"[^;()\n\"]*\bd[aoe]s?\s+", draft[a.end():after.start()]) is not None
        tight = before is not None and not re.search(r"[A-Za-zÀ-ú0-9]", draft[before.end():a.start()])
        m = after if joined else (before if tight else None)
        key = act_key(m, known) if m else None
        for item in re.finditer(_ART_ITEM, a.group(1)):
            number = re.match(r"(\d[\d.]*)\s*[º°o]?\s*(?:-\s*([A-Z]))?", item.group(0))
            art = str(int(number.group(1).replace(".", ""))) + (f"-{number.group(2)}" if number.group(2) else "")
            out.append({"act": key, "article": art, "text": a.group(0)[:60]})
    return out


def classify_statutes(draft: str, table: dict) -> dict:
    """Each cited (act, article) as found, in_range (manual review), beyond, unknown_act or no_act."""
    rows = []
    for c in statute_citations(draft, set(table)):
        act = table.get(c["act"]) if c["act"] else None
        if c["act"] is None:
            status = "no_act"
        elif act is None:
            status = "unknown_act"
        elif c["article"] in set(act["articles"]):
            status = "found"
        elif act["last_article"] and int(c["article"].split("-")[0]) <= act["last_article"]:
            status = "in_range"
        else:
            status = "beyond"
        rows.append({**c, "status": status})
    unique = {(r["act"], r["article"]): r["status"] for r in rows}
    counts = {s: sum(1 for v in unique.values() if v == s) for s in ("found", "in_range", "beyond", "unknown_act", "no_act")}
    graded = counts["found"] + counts["in_range"] + counts["beyond"]
    return {"statutes": rows, **{f"statute_{k}": v for k, v in counts.items()},
            "statute_invalid_share": counts["beyond"] / graded if graded else None}
