"""Normative citations in Portuguese legal text -> normalised provision IDs (Phase 4a).

    parse("art. 113 do Decreto-Lei nº 73/1966")  ->  [Provision("DL73/1966", "113", ...)]
    provision.id                                  ->  "DL73/1966:art113"

A citation has a norm (kind, issuer, number, year) and usually one or more articles. Articles are
kept at article level: paragraphs, incisos, alíneas, items and annexes are dropped. A norm cited
without an article gives a norm-level ID ("CIRC-SUSEP364/2008").

How an article finds its norm, in this order:
1. a reference back ("ambos da mesma norma", "da citada Resolução"): the previous norm;
2. a norm placed just before it, with at most a comma between, that no earlier article has taken
   ("Circular SUSEP nº 74/1999, art. 5º"), and the articles that continue such a list;
3. the first norm mentioned after it, within `MAX_GAP` characters ("arts. 24 e 113 do Decreto-Lei
   nº 73/66").
Articles that find no norm are returned with norm None.

Two-digit years are expanded, and codes are named ("CC" for Lei 10.406/2002). A norm written
without year or issuer ("Lei 7492", "Resolução nº 321/2015") is completed from `KNOWN_YEARS` and
from an optional `canon` table built on the corpus (`build_canon`).

Development: see config/a1_labels.yaml (`development`).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

MAX_GAP = 160          # characters between an article and the norm that follows it
MAX_CHAIN_GAP = 20     # characters between two article lists that share a norm placed before them

# Norms that are usually cited without their year.
KNOWN_YEARS = {
    ("DL", 73): 1966, ("DL", 261): 1967, ("DL", 2848): 1940, ("DL", 2063): 1940,
    ("LC", 109): 2001, ("LC", 126): 2007, ("LC", 213): 2025,
    ("LEI", 4594): 1964, ("LEI", 5172): 1966, ("LEI", 6194): 1974, ("LEI", 6404): 1976,
    ("LEI", 6435): 1977, ("LEI", 6830): 1980, ("LEI", 7492): 1986, ("LEI", 8078): 1990,
    ("LEI", 9613): 1998, ("LEI", 9656): 1998, ("LEI", 9784): 1999, ("LEI", 9873): 1999,
    ("LEI", 10406): 2002, ("LEI", 13105): 2015, ("DEC", 60459): 1967,
}
# Laws better known as codes.
CODE_ALIASES = {
    "LEI10406/2002": "CC", "LEI13105/2015": "CPC", "LEI8078/1990": "CDC", "LEI5172/1966": "CTN",
    "DL2848/1940": "CP",
}
DEFAULT_ISSUER = {"CIRC": "SUSEP", "CARTA-CIRC": "SUSEP"}


def _default_issuer(kind: str, number: int) -> str:
    """Issuer of a norm cited without one: CNSP resolutions are numbered below 1000, CMN's above 2000."""
    if kind == "RES":
        return "CNSP" if number < 1000 else "CMN" if number >= 2000 else ""
    return DEFAULT_ISSUER.get(kind, "")

_KINDS = [
    # (prefix, pattern, takes an issuer); the patterns also accept common OCR forms
    ("DL", r"decreto[\s\-‐–]*le[il1]\b|\bDL\b", False),              # "Decreto-Lel"
    ("LC", r"lei\s+co[a-z]{6,12}ntar\b|\bLC\b", False),             # "Lei Cornplernentar"
    ("CARTA-CIRC", r"carta[\s\-]*circular", True),
    # accounting pronouncements ("CPC nº 38"), not the Código de Processo Civil
    ("PRON-CPC", r"(?:pronunciamento\s+t[ée]cnico\s+)?\bCPC\b(?=\s*(?:n\s?[º°o.]*\s*)?\d)", False),
    ("RES", r"reso[li1]u[^\s\d]{1,5}|\bres\b\.?(?=\s*(?:n|[A-Z]{2}))", True),   # "Resoluco", "Res. CNSP"
    ("CIRC", r"circular|\bcirc\b\.?(?=\s*(?:susep|n\b|n[º°.]|\d))", True),    # "Circ. SUSEP", "Circ SUSEP 365"
    ("PORT", r"portaria", True),
    ("INST", r"instru[çc][ãa]o(?:\s+normativa)?", True),
    ("DELIB", r"delibera[çc][ãa]o", True),
    ("MP", r"medida\s+provis[óo]ria", False),
    ("LEI", r"lei(?:\s+federal)?", False),
    ("DEC", r"decreto", False),
]
_ISSUER = r"(?:CNSP|SU\s?SEP|SUSEF|CMN|CFC|CVM|BACEN|BCB|MF|ME|CGPC|CNPC|PREVIC|ANS|CRSNSP|COAF)"
# "nº", "n.", "No" and OCR forms of the mark ("n2", "n9", 'n"', "na", a lone "0") before a number
_NUM_MARK = (r"(?:n\s?[º°ªo]s?\.?|n\.\s?[º°ªo]?|n[29\"”a](?=\s*\d)|n(?=\s*\d)|[º°](?=\s*\d)"
             r"|(?<=\s)[0o](?=\s+\d+\s*/))")
_NUMBER = r"(?P<num>\d{1,3}(?:\.\d{3})+|\d{1,6})"
_YEAR = (r"(?:\s*/\s*(?P<y1>\d{4}|\d{2})(?!\d)"
         r"|\s*,?\s+de\s+(?:\d{1,2}[º°]?\s+de\s+[a-zç]+\s+de\s+)?(?P<y2>\d{4})(?!\d))?")

NORM_RE = re.compile(
    r"(?P<kind>" + "|".join(f"(?P<k{i}>{p})" for i, (_, p, _) in enumerate(_KINDS)) + r")"
    r"(?:\s*(?:" + _NUM_MARK + r")\s*)?"                       # "Resolução nº CNSP nº 321/2015"
    r"(?:\s*(?:d[oa]\s+)?(?P<issuer>" + _ISSUER + r")(?:/[A-Z]+)*/?)?"
    r"\s*(?:" + _NUM_MARK + r")?\s*" + _NUMBER + _YEAR,
    re.IGNORECASE)

CODE_RE = re.compile(
    r"(?P<CC>c[óo]digo\s+civil|\bCC(?:/\d{2,4})?\b)"
    r"|(?P<CPC>c[óo]digo\s+de\s+processo\s+civil|\bN?CPC(?:/\d{2,4})?\b)"
    r"|(?P<CDC>c[óo]digo\s+de\s+(?:prote[çc][ãa]o\s+e\s+)?defesa\s+do\s+consumidor|\bCDC\b)"
    r"|(?P<CF>constitui[çc][ãa]o(?:\s+federal|\s+da\s+rep[úu]blica(?:\s+federativa\s+do\s+brasil)?)?"
    r"|\bCF(?:/(?:19)?88)?\b|\bCRFB(?:/(?:19)?88)?\b)"
    r"|(?P<CP>c[óo]digo\s+penal\b)"
    r"|(?P<CTN>c[óo]digo\s+tribut[áa]rio\s+nacional|\bCTN\b)"
    r"|(?P<RICRSNSP>regimento\s+interno(?:\s+do\s+CRSNSP)?|\bRI-?CRSNSP\b)",
    re.IGNORECASE)

# A later act named only because it amended the cited one is not part of the citation.
AMENDED_RE = re.compile(
    r"(?:reda[çc][ãa]o(?:\s+(?:dada|conferida|determinada|atual))?|alterad[oa]s?|alter(?:a[çc][ãa]o|a[çc][õo]es)"
    r"(?:\s+\w+){0,2}|modificad[oa]s?|revogad[oa]s?|substitu[íi]d[oa]s?|inclu[íi]d[oa]s?|acrescid[oa]s?)"
    r"\s+(?:pel[oa]s?|d[oa]s?|n[oa]s?)\s*(?:art\w*\.?\s*\d+\S*\s+d[oa]\s+)?$", re.IGNORECASE)
BACK_REF_RE = re.compile(
    r"^[\s,]*(?:(?:tod[oa]s|amb[oa]s)\s+)?d[oa]s?\s+(?:mesm[oa]s?|citad[oa]s?|referid[oa]s?|aludid[oa]s?"
    r"|mencionad[oa]s?|supracitad[oa]s?|j[áa]\s+citad[oa]s?)\b", re.IGNORECASE)

# Sub-article units, removed before articles are read so that their numbers are not taken as articles.
_ORD = r"[\s.]*[º°ªo]?"


def _list(item: str) -> str:
    return rf"{item}(?:\s*(?:,|e|a|ao)\s*{item})*"


_SUBUNIT_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"§§+\s*" + _list(r"\d+" + _ORD),                           # "§§ 1º, 2º e 13"
    # "§ 1º e 3º": after a single sign only numbers below 10 continue the paragraphs, so that the
    # articles of "arts. 37, § 3º, 38, 40" are kept
    r"§\s*(?:[úu]nico|\d+" + _ORD + r"(?:\s*(?:,|e|a|ao)\s*\d(?!\d)" + _ORD + r")*)",
    r"\bpar[áa]grafos\s+" + _list(r"\d+" + _ORD),
    r"\bpar[áa]grafo\s+(?:[úu]nico|primeiro|segundo|terceiro|\d+" + _ORD + r")",
    r"\b(?:sub)?ite(?:m|ns)\s+" + _list(r"\d+(?:\.\d+)*"),
    r"\bincisos?\s+" + _list(r"\d+" + _ORD),
    r"\b(?:anexo|cap[íi]tulo|t[íi]tulo|se[çc][ãa]o|tabela|quadro)\s+(?:\d+|[IVXLC]+\b)",
)]

ART_HEAD_RE = re.compile(r"\b(?:arts?\b\.?|artigos?\b)\s*(?:art\.\s*)?", re.IGNORECASE)
_ART_NUM_RE = re.compile(r"(\d{1,2}\.\d{3}|\d{1,4})(?:\s?[-‐–]\s?([A-Z])\b(?!\w))?[\s.]*[º°ªo]?(?![\d/])")
# Words that may stand between two articles of one list: separators, "c/c", "caput", incisos, alíneas.
_ART_FILLER_RE = re.compile(
    r"\s*(?:,|;|\be\b|\bc\s?[/.]\s?c\b\.?|\bcaput\b|\bincisos?\b|\binc\b\.?|\bal[íi]neas?\b|[\"“”'‘’`´]\w[\"“”'‘’`´]"
    r"|\b[IVXLC]+\b|\([^)]{0,40}\))\s*", re.IGNORECASE)


@dataclass(frozen=True)
class Provision:
    norm: str | None          # "DL73/1966"; None when an article found no norm
    article: str | None       # "113", "35B"; None for a norm cited without an article
    start: int                # position of the article (or of the norm) in the text
    amending: bool = False    # the norm is named only as the act that amended another

    @property
    def id(self) -> str:
        norm = self.norm or "?"
        return f"{norm}:art{self.article}" if self.article else norm


@dataclass(frozen=True)
class _Norm:
    prefix: str
    number: int | None
    year: int | None
    start: int
    end: int
    amending: bool

    @property
    def id(self) -> str:
        if self.number is None:
            return self.prefix
        raw = f"{self.prefix}{self.number}" + (f"/{self.year}" if self.year else "")
        return CODE_ALIASES.get(raw, raw)


def _year(text: str | None) -> int | None:
    if not text:
        return None
    y = int(text)
    if y >= 100:
        return y
    return 2000 + y if y <= 30 else 1900 + y


def _issuer(text: str | None) -> str:
    return re.sub(r"\s", "", text or "").upper().replace("SUSEF", "SUSEP")


def _kind(match: re.Match) -> tuple[str, bool]:
    return next((prefix, issuer) for i, (prefix, _, issuer) in enumerate(_KINDS) if match.group(f"k{i}"))


def _blank(match: re.Match) -> str:
    return " " * len(match.group(0))


def _find_norms(text: str, canon: dict | None) -> list[_Norm]:
    norms = []
    for m in NORM_RE.finditer(text):
        kind, takes_issuer = _kind(m)
        number = int(m.group("num").replace(".", ""))
        year = _year(m.group("y1") or m.group("y2"))
        issuer = _issuer(m.group("issuer")) if takes_issuer else ""
        known_issuer, known_year = (canon or {}).get((kind, number), (None, None))
        if takes_issuer:
            issuer = issuer or known_issuer or _default_issuer(kind, number)
        year = year or known_year or KNOWN_YEARS.get((kind, number))
        amending = bool(AMENDED_RE.search(text[max(0, m.start() - 60):m.start()]))
        norms.append(_Norm(f"{kind}-{issuer}" if issuer else kind, number, year, m.start(), m.end(), amending))
    taken = [(n.start, n.end) for n in norms]
    for m in CODE_RE.finditer(text):
        if not any(s <= m.start() < e for s, e in taken):
            norms.append(_Norm(m.lastgroup, None, None, m.start(), m.end(), False))
    return sorted(norms, key=lambda n: n.start)


def _strip_subunits(text: str) -> str:
    """`text` with paragraphs, items, annexes etc. blanked (same length)."""
    for pattern in _SUBUNIT_RES:
        text = pattern.sub(_blank, text)
    return text


def _find_article_groups(clean: str) -> list[tuple[int, int, list[str]]]:
    """(start, end, article numbers) of every "art. ..." list in a text without sub-article units."""
    groups = []
    for head in ART_HEAD_RE.finditer(clean):
        pos, articles = head.end(), []
        while True:
            m = _ART_NUM_RE.match(clean, pos)
            if not m:
                break
            articles.append(m.group(1).replace(".", "") + (m.group(2) or ""))
            pos = m.end()
            nxt = pos
            while (f := _ART_FILLER_RE.match(clean, nxt)) and f.end() > nxt:
                nxt = f.end()
            if not _ART_NUM_RE.match(clean, nxt) or NORM_RE.match(clean, nxt):
                break
            pos = nxt
        if articles:
            groups.append((head.start(), pos, articles))
    return groups


def parse(text: str | None, canon: dict | None = None) -> list[Provision]:
    """Every provision cited in `text`, in order of appearance (duplicates kept)."""
    if not isinstance(text, str) or not text:
        return []
    norms = _find_norms(text, canon)
    clean = _strip_subunits(text)
    out, used = [], set()
    chain: tuple[_Norm, int] | None = None      # (norm placed before its articles, end of the last list)
    for start, end, articles in _find_article_groups(clean):
        before = [n for n in norms if n.end <= start and not n.amending]
        after = [n for n in norms if n.start >= end and not n.amending]
        norm, backward = None, False
        tail = end                                  # past the incisos that close the list
        while (f := _ART_FILLER_RE.match(clean, tail)) and f.end() > tail:
            tail = f.end()
        if before and BACK_REF_RE.match(text[tail:tail + 60]):
            norm = before[-1]
        elif before and before[-1].start not in used and not re.search(r"[\w;.]", clean[before[-1].end:start]):
            norm, backward = before[-1], True
        elif chain and start - chain[1] <= MAX_CHAIN_GAP and not (after and after[0].start - end <= 10):
            norm, backward = chain[0], True
        elif after and after[0].start - end <= MAX_GAP:
            norm = after[0]
        chain = (norm, end) if backward else None
        if norm is not None:
            used.add(norm.start)
        out.extend(Provision(norm.id if norm else None, a, start) for a in articles)
    out.extend(Provision(n.id, None, n.start, n.amending) for n in norms if n.start not in used)
    return sorted(out, key=lambda p: p.start)


def provision_ids(text: str | None, canon: dict | None = None, *, norm_only: bool = True) -> list[str]:
    """Distinct IDs cited in `text`, in order of first appearance.

    Articles without a norm and acts named only as amendments are left out. A norm cited without an
    article is left out when one of its articles is cited, and always when `norm_only` is False.
    """
    provisions = [p for p in parse(text, canon) if p.norm and not p.amending]
    with_article = {p.norm for p in provisions if p.article}
    ids: dict[str, None] = {}
    for p in provisions:
        if p.article or (norm_only and p.norm not in with_article):
            ids.setdefault(p.id)
    return list(ids)


def norm_of(provision_id: str) -> str:
    return provision_id.split(":art")[0]


def build_canon(texts: Iterable[str | None]) -> dict[tuple[str, int], tuple[str | None, int | None]]:
    """(kind, number) -> the issuer and year most often written with it in `texts`.

    Used to complete citations that omit the year or the issuer. A value is kept only when it
    accounts for at least two thirds of the citations of that number that state one.
    """
    issuers: dict[tuple[str, int], Counter] = defaultdict(Counter)
    years: dict[tuple[str, int], Counter] = defaultdict(Counter)
    for text in texts:
        for m in NORM_RE.finditer(text if isinstance(text, str) else ""):
            kind, takes_issuer = _kind(m)
            key = (kind, int(m.group("num").replace(".", "")))
            if takes_issuer and m.group("issuer"):
                issuers[key][_issuer(m.group("issuer"))] += 1
            if (year := _year(m.group("y1") or m.group("y2"))):
                years[key][year] += 1

    def top(counter: Counter):
        if not counter:
            return None
        value, n = counter.most_common(1)[0]
        return value if n * 3 >= sum(counter.values()) * 2 else None

    return {key: (top(issuers.get(key, Counter())), top(years.get(key, Counter())))
            for key in set(issuers) | set(years)}
