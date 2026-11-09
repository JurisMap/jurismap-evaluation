"""Test notices of the A4 drafting evaluation (Phase 8, task 1) and the material of task 5.

A test notice is an infraction notice reconstructed from a real decision's report, so that a draft
written from it can be compared with what the real appellant argued. Mistral Medium 3 writes the
notice, the list of the appellant's arguments and, for granted appeals, which of them the board
accepted (config/a4_drafting.yaml). Every step that reads what the board decided is given the
dispositive and the outcome label: in some decisions the rapporteur was outvoted, and the vote then
argues the losing position (DECISIONS.md, 2026-10-18).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from jm_eval.chatbot.judging import cut_middle, field_text
from jm_eval.io import load_config, resolve

CONFIG = "a4_drafting"
PROMPTS_VERSION = 1           # argument and accepted-argument prompts
NOTICE_PROMPT_VERSION = 3     # 2: the case document adds SUSEP's first-instance decision (the author, 2026-10-18);
                              # 3: its final penalty, not an earlier proposal (N14 had taken a first proposal)


def settings() -> dict:
    return load_config(CONFIG)


# ---------------------------------------------------------------------------- what may not be used

def excluded_decisions(cfg: dict | None = None) -> set[str]:
    """Every decision used in Phases 6 and 7: case-set seeds and members, and out-of-set question subjects."""
    spec = (cfg or settings())["excluded_from"]
    out: set[str] = set()
    for name in spec["case_sets"]:
        for path in all_versions(resolve(name)):
            for s in json.loads(path.read_text(encoding="utf-8"))["sets"]:
                out.add(s["seed_hash"])
                out.update(s.get("case_hashes") or s.get("ranked_hashes") or [])
    for name in spec["outside_questions"]:
        for path in all_versions(resolve(name)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and (h := json.loads(line).get("outside_hash")):
                    out.add(h)
    return out


def all_versions(path: Path) -> list[Path]:
    """An artefact and its numbered versions `<stem>_<n><suffix>` that exist."""
    pattern = re.compile(re.escape(path.stem) + r"(_\d+)?" + re.escape(path.suffix))
    return sorted(p for p in path.parent.glob(f"{path.stem}*{path.suffix}") if pattern.fullmatch(p.name))


_OUTVOTED_CLAUSE = re.compile(r"\bvencid[oa]s?\b(.*?)(?:\bque\b|\.\s|;|$)", re.I | re.S)
_RAPPORTEUR = re.compile(r"\brelator[a]?\b", re.I)


def rapporteur_outvoted(dispositive: str | None) -> bool:
    """True when a "vencido(s)" clause of the dispositive names the rapporteur, on any point."""
    return any(_RAPPORTEUR.search(m.group(1)) for m in _OUTVOTED_CLAUSE.finditer(field_text(dispositive)))


# ---------------------------------------------------------------------------- selection

def strata(labels: pd.DataFrame, n: int) -> list[str]:
    """The `n` most frequent legal-basis keys (ties in key order)."""
    counts = labels["base_key"].dropna().value_counts()
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def eligible(corpus: pd.DataFrame, excluded: set[str], spec: dict, outvoted_excluded: bool) -> pd.Series:
    """Decisions that may be drawn: not used before, enough text, an outcome on the merits."""
    sel = spec["selection"]
    ok = ~corpus["acordao_hash"].isin(excluded)
    ok &= corpus["acordao_resultado"].isin(sel["outcomes"])
    ok &= corpus["acordao_relatorio"].fillna("").str.len() >= sel["min_report_chars"]
    ok &= corpus["acordao_voto_relator"].fillna("").str.len() >= sel["min_vote_chars"]
    if outvoted_excluded:
        ok &= ~corpus["acordao_acordao"].map(rapporteur_outvoted)
    return ok


def select(corpus: pd.DataFrame, labels: pd.DataFrame, excluded: set[str], spec: dict) -> pd.DataFrame:
    """One decision drawn at random in each cell of legal-basis key x outcome, one per process.

    `corpus` and `labels` are aligned (same `acordao_hash` order).
    """
    if corpus["acordao_hash"].tolist() != labels["acordao_hash"].tolist():
        raise ValueError("corpus and labels are not aligned")
    sel = spec["selection"]
    ok = eligible(corpus, excluded, spec, spec["outvoted"]["exclude_from_test_notices"]).to_numpy()
    groups = labels["process_group"].to_numpy()
    replaced = sel.get("replaced") or {}
    hashes = corpus["acordao_hash"].to_numpy()
    rng = np.random.default_rng(sel["seed"])
    used_groups: set = set()
    rows = []
    for key in strata(labels, sel["n_strata"]):
        for outcome in sel["outcomes"]:
            cell = np.flatnonzero(ok & (labels["base_key"] == key).to_numpy()
                                  & (corpus["acordao_resultado"] == outcome).to_numpy())
            cell = np.array([i for i in cell if groups[i] not in used_groups])
            if not len(cell):
                raise ValueError(f"no eligible decision for {key} / {outcome}")
            i = int(rng.choice(cell))
            replaces = None
            if hashes[i] in replaced:
                # A drawn decision found unusable is replaced by a draw of its own (seeded by the cell),
                # so that the main draw, and every other cell, stays as it was.
                rest = np.array([j for j in cell if hashes[j] not in replaced])
                if not len(rest):
                    raise ValueError(f"no replacement for {hashes[i]} in {key} / {outcome}")
                replaces, i = str(hashes[i]), int(np.random.default_rng([sel["seed"], len(rows)]).choice(rest))
            used_groups.add(groups[i])
            rows.append({"notice_id": f"N{len(rows) + 1:02d}", "acordao_hash": hashes[i],
                         "title": corpus["acordao_titulo"].iat[i], "base_key": key, "outcome": outcome,
                         "process_group": groups[i], "cell_size": len(cell), "replaces": replaces})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- prompts

SYSTEM = ("You work on decisions of the CRSNSP, the Brazilian administrative appeals council for sanctions in "
          "insurance. Be literal and strict: use only the text you are given. Answer with JSON only.")

NOTICE_PROMPT = """Below is the rapporteur's REPORT of a CRSNSP decision, in Portuguese (a long text is cut in the middle, marked [...]), and the legal provisions the decision lists.

The report tells the history of a sanction case: what SUSEP found, the provisions it charged, the party's defence, SUSEP's first-instance decision and the appeal to the council. Reconstruct from it, in Portuguese, the CASE DOCUMENT a party appealing to the council would hold: the infraction notice (auto de infração or termo de comunicação de irregularidades) that started the case, followed by SUSEP's first-instance decision as far as the penalty goes.

The document contains ONLY:
- a header: "Superintendência de Seguros Privados – SUSEP", the kind of document, "nº [número]", the party charged ("Autuada") by name as the report gives it, and the date of the notice if the report states it;
- "Dos Fatos Apurados": the facts SUSEP found, with their dates, amounts and documents, as the report states them;
- "Dos Dispositivos Legais Aplicáveis": the provisions the party was charged with infringing;
- "Do Enquadramento da Infração": the infraction those facts amount to, in one paragraph;
- "Da Decisão da SUSEP": the penalty SUSEP applied at first instance (type and amount) and how it was set, as the report states it: the base penalty, each aggravating factor (gravity, effects, antecedents, recidivism, continuity) and each mitigating factor with its effect on the amount, and the provisions applied to set it. If the report does not say how the penalty was set, give only the penalty.

The document must NOT contain: anything the party argued or explained (its defence, appeal, justifications or corrective measures); SUSEP's answers to those arguments; the opinions of SUSEP's legal staff or the Procuradoria beyond the penalty and how it was set; anything that happened after SUSEP's first-instance decision (the appeal, the council's decision, its outcome or vote, any change to the penalty); process or appeal numbers (write "[número]"); names of SUSEP staff (write "[Agente de Fiscalização]").

Rules:
- State only what the report states. Do not invent facts, dates, amounts, provisions or penalties. Where the report does not give a detail, leave it out.
- Charge only provisions that the report or the listed provisions say the party infringed. Provisions on penalties, appeals or procedure (such as those of Resolução CNSP 243/2011) are not charges; those on penalties belong in "Da Decisão da SUSEP".
- Do not mention the right to appeal or the council.
- The report may describe several stages before the appeal (a first proposal of penalty, a new analysis, a review). "Da Decisão da SUSEP" gives the penalty of SUSEP's final first-instance decision (the "termo de julgamento" or "decisão de primeira instância" the party appealed against), not an earlier proposal.
{avoid}
Return JSON only: {{"notice": "<the full text of the document, with the section headings, sections separated by blank lines>"}}

LISTED PROVISIONS:
{base}

REPORT:
{report}"""

NOTICE_AVOID = """- The previous version stated these arguments of the party, or something close to them. Leave them out entirely:
{arguments}
"""

ARGUMENTS_PROMPT = """Below are the rapporteur's REPORT of a CRSNSP decision (a long text is cut in the middle, marked [...]), the DISPOSITIVE (what the council decided, by unanimity or by majority) and the OUTCOME label, in Portuguese.

List the arguments the appellant raised in its appeal (recurso) to the council, as the report states them. Include arguments repeated from the first-instance defence when the report says the appeal repeats them.

Rules:
- {min} to {max} arguments. One argument per item: split a passage that raises several.
- Each item is one self-contained sentence in Portuguese stating what the appellant argued (e.g. "A recorrente alega que ..."). Do not say whether the argument succeeded.
- State only what the report states. No outside knowledge. Do not name the decision (no process or appeal number).

Return JSON only: {{"arguments": ["...", "..."]}}

OUTCOME: {outcome}

DISPOSITIVE:
{dispositive}

REPORT:
{report}"""

ACCEPTED_PROMPT = """Below are the numbered ARGUMENTS the appellant raised in an appeal to the CRSNSP, the DISPOSITIVE (what the council decided), the OUTCOME label and the rapporteur's VOTE (a long text is cut in the middle, marked [...]), in Portuguese.

The appeal was granted in full or in part. Say which of the numbered arguments the council accepted, that is, which argument the part of the appeal that was granted rests on.

Important: the dispositive states what the majority decided and prevails over the vote. The rapporteur may have been outvoted on some point ("Vencido o Relator ..."): where the dispositive and the vote disagree, follow the dispositive, and read the vote only for the reasons of the points on which it agrees with the dispositive.

Rules:
- Give the numbers of the accepted arguments. Give an empty list if the granted part rests on none of them (for instance, a point the council raised by itself).
- "granted": one sentence in Portuguese stating what was granted (e.g. the fine reduced and why), taken from the dispositive and the vote.

Return JSON only: {{"accepted": [<argument numbers>], "granted": "..."}}

OUTCOME: {outcome}

DISPOSITIVE:
{dispositive}

ARGUMENTS:
{arguments}

VOTE:
{vote}"""


def _dispositive(row, chars: int) -> str:
    text = field_text(row["acordao_acordao"])
    return text if len(text) <= chars else text[:chars].rstrip() + " [...]"


NOTICE_REVIEW = """- A reviewer rejected a previous version of this document. Correct what the reviewer points out, keep what the reviewer does not question, and follow every rule above.
  REVIEWER'S COMMENT: {comment}
  REJECTED VERSION:
{previous}
"""


def notice_prompt(row, spec: dict, avoid: list[str] | None = None, review: dict | None = None) -> str:
    """`review`: {"previous": <rejected document>, "comment": <the reviewer's comment>}."""
    m = spec["material"]
    avoid_text = NOTICE_AVOID.format(arguments="\n".join(f"  - {a}" for a in avoid)) if avoid else ""
    if review:
        previous = "\n".join(f"    {line}" for line in review["previous"].splitlines())
        avoid_text += NOTICE_REVIEW.format(comment=review["comment"].strip(), previous=previous)
    return NOTICE_PROMPT.format(avoid=avoid_text, base=field_text(row["acordao_base"]) or "(none listed)",
                                report=cut_middle(field_text(row["acordao_relatorio"]), **m["notice_report_chars"]))


ARGUMENTS_REVIEW = """
A reviewer rejected a previous list. Correct what the reviewer points out and follow every rule above.
REVIEWER'S COMMENT: {comment}
REJECTED LIST:
{previous}
"""


ARGUMENTS_VOTE = """
The rapporteur's VOTE is given below the report (a long text is cut in the middle, marked [...]). Use it only to find arguments the appellant raised in the appeal that the report leaves out: the vote often lists the appeal's points and answers them. Do not list as arguments the rapporteur's own reasons, or the positions of SUSEP, its legal staff or the Procuradoria. The dispositive states what the majority decided and prevails over the vote.
"""


def arguments_prompt(row, spec: dict, review: dict | None = None, with_vote: bool = False) -> str:
    """`review`: {"previous": <rejected list>, "comment": <the reviewer's comment>}. `with_vote`: the
    rapporteur's vote is read too, for appeal points the report leaves out (the author, 2026-10-18: N09, N12),
    with shorter cuts so the prompt stays under the limit (config `material.arguments_with_vote`)."""
    m = spec["material"]
    prompt = ARGUMENTS_PROMPT
    if with_vote:
        cuts = m["arguments_with_vote"]
        prompt = prompt.replace("\nReturn JSON only:", ARGUMENTS_VOTE + "\nReturn JSON only:", 1)
        prompt += "\n\nVOTE:\n" + cut_middle(field_text(row["acordao_voto_relator"]), **cuts["vote_chars"]).replace("{", "{{").replace("}", "}}")
        m = {**m, "arguments_report_chars": cuts["report_chars"]}
    if review:
        block = ARGUMENTS_REVIEW.format(comment=review["comment"].strip().replace("{", "{{").replace("}", "}}"),
                                        previous="\n".join(f"{i}. {a}" for i, a in enumerate(review["previous"], 1))
                                        .replace("{", "{{").replace("}", "}}"))
        prompt = prompt.replace("\nReturn JSON only:", block + "\nReturn JSON only:", 1)
    return prompt.format(min=m["arguments"]["min"], max=m["arguments"]["max"],
                                   outcome=field_text(row["acordao_resultado"]),
                                   dispositive=_dispositive(row, m["dispositive_chars"]),
                                   report=cut_middle(field_text(row["acordao_relatorio"]), **m["arguments_report_chars"]))


def accepted_prompt(row, arguments: list[str], spec: dict) -> str:
    m = spec["material"]
    return ACCEPTED_PROMPT.format(outcome=field_text(row["acordao_resultado"]),
                                  dispositive=_dispositive(row, m["dispositive_chars"]),
                                  arguments="\n".join(f"{i}. {a}" for i, a in enumerate(arguments, 1)),
                                  vote=cut_middle(field_text(row["acordao_voto_relator"]), **m["accepted_vote_chars"]))


# ---------------------------------------------------------------------------- answer checks

def validate_notice(parsed):
    text = parsed.get("notice") if isinstance(parsed, dict) else None
    if not isinstance(text, str) or len(text.strip()) < 300:
        return None, "expected a notice of at least 300 characters"
    missing = [h for h in ("Dos Fatos", "Dispositivos", "Da Decisão") if h.lower() not in text.lower()]
    return (None, f"missing sections: {missing}") if missing else ({"notice": text.strip()}, None)


def validate_arguments(parsed, low: int, high: int):
    items = parsed.get("arguments") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return None, "expected a list of arguments"
    items = [a.strip() for a in items if isinstance(a, str) and a.strip()]
    return ({"arguments": items}, None) if low <= len(items) <= high else (None, f"expected {low} to {high} arguments")


def validate_accepted(parsed, n: int):
    if not isinstance(parsed, dict) or not isinstance(parsed.get("accepted"), list):
        return None, "expected a list of argument numbers"
    numbers = parsed["accepted"]
    if not all(isinstance(i, int) and 1 <= i <= n for i in numbers) or len(set(numbers)) != len(numbers):
        return None, f"argument numbers must be distinct integers from 1 to {n}"
    granted = parsed.get("granted")
    if not isinstance(granted, str) or not granted.strip():
        return None, "expected the granted part"
    return {"accepted": sorted(numbers), "granted": granted.strip()}, None


# ---------------------------------------------------------------------------- leakage

def normalise(text: str) -> str:
    """Lower case, no accents, punctuation as spaces."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def ngram_containment(argument: str, notice: str, n: int) -> float:
    """Share of the argument's word n-grams found in the notice (0 for an argument shorter than n words)."""
    a, b = normalise(argument).split(), normalise(notice).split()
    grams = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    if not grams:
        return 0.0
    found = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(grams & found) / len(grams)


def forbidden_terms(notice: str, patterns: list[str]) -> list[str]:
    """The forbidden terms the notice contains, as found."""
    text = normalise(notice)
    return sorted({m.group(0) for p in patterns for m in re.finditer(rf"\b(?:{p})\b", text)})


def sentences(text: str) -> list[str]:
    """Sentences and list items of a text, at least 4 words each."""
    parts = re.split(r"(?<=[.;:!?])\s+|\n+|•", text or "")
    return [p.strip() for p in parts if len(p.split()) >= 4]


_AMOUNT = re.compile(r"R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\d+(?:,\d{2})?)")


def amounts(text: str | None) -> set[str]:
    """Money amounts of a text, as digits with two decimals ("R$ 25.720,00" -> "25720.00")."""
    out = set()
    for m in _AMOUNT.finditer(field_text(text)):
        whole, _, cents = m.group(1).partition(",")
        out.add(f"{int(whole.replace('.', ''))}.{cents or '00'}")
    return out


def penalty_field_amount(penalty: str | None) -> str | None:
    """The first amount of the decision's penalty field: usually the fine SUSEP applied (not always: N09, N13)."""
    m = _AMOUNT.search(field_text(penalty))
    return next(iter(amounts(m.group(0)))) if m else None


_VERB = re.compile(r"\b(?:(?P<kept>mant(?:er|endo|id[oa]s?|em|enh\w*))|reduz\w*|fix\w*|convol\w*|minor\w*|diminu\w*"
                   r"|afast\w*|aplic(?!ad[oa]s?\b)\w*|substitu\w*|redimension\w*)\b", re.I)   # "aplicada" (the fine SUSEP applied) decides nothing


def council_amounts(dispositive: str | None) -> set[str]:
    """Amounts the council's dispositive sets (a reduced fine, the minority's proposal), not those it keeps.
    The nearest verb in the 80 characters before an amount decides: "manter", "mantendo", "mantida" mark
    SUSEP's own fine; any other verb, or none, an amount the council set."""
    text = field_text(dispositive)
    out = set()
    for m in _AMOUNT.finditer(text):
        verbs = list(_VERB.finditer(text[max(0, m.start() - 80):m.start()]))
        if not (verbs and verbs[-1].group("kept")):
            out |= amounts(m.group(0))
    return out


def outcome_amounts(notice: str, dispositive: str | None, report: str | None) -> list[str]:
    """Amounts of the case document that can give away the appeal's outcome, whatever their source:
    those the council's dispositive sets (version 2 of the rule, after N14: SUSEP's early figure of
    R$ 20.000,00 was also the amount the appeal granted), and, as in version 1, any amount of the
    dispositive that the report does not state."""
    return sorted(amounts(notice) & (council_amounts(dispositive) | (amounts(dispositive) - amounts(report))))


# ---------------------------------------------------------------------------- leakage check

def argument_content(argument: str) -> str:
    """The argument without its "A recorrente alega que" opening, so the NLI model judges what is argued."""
    words = argument.split()
    for i, w in enumerate(words[:10]):
        if w.lower() == "que":
            return " ".join(words[i + 1:]) or argument
    return argument


def leakage(notice: str, arguments: list[str], row: dict, spec: dict, nli) -> dict:
    """Per argument: the highest entailment by a sentence of the notice and the n-gram containment;
    forbidden terms; amounts that can give away the outcome. `nli` is jm_eval.chatbot.nli.load()."""
    from jm_eval.chatbot import nli as NLI

    lk = spec["leakage"]
    premises = sentences(notice) or [notice]
    rows = []
    for k, argument in enumerate(arguments, 1):
        probs = NLI.probabilities(nli, premises, [argument_content(argument)] * len(premises))
        best = int(np.argmax(probs[:, 0]))
        rows.append({"argument": k, "entailment": round(float(probs[best, 0]), 4), "sentence": premises[best],
                     "containment": round(ngram_containment(argument, notice, lk["ngram"]), 4)})
    flagged = [r["argument"] for r in rows
               if r["entailment"] > lk["nli_threshold"] or r["containment"] > lk["ngram_threshold"]]
    terms = forbidden_terms(notice, lk["forbidden_terms"])
    outcome = outcome_amounts(notice, row["acordao_acordao"], row["acordao_relatorio"])
    return {"per_argument": rows, "flagged_arguments": flagged, "forbidden_terms": terms,
            "outcome_amounts": outcome, "flagged": bool(flagged or terms or outcome)}


def load_frozen(cfg: dict | None = None) -> list[dict]:
    """The frozen test notices (scripts/82), checked against the hash in the freeze record."""
    from jm_eval.io import latest_artefact_path, sha256_file

    record_path = latest_artefact_path(resolve((cfg or settings())["freeze"]["record"]))
    if not record_path.exists():
        raise FileNotFoundError(f"{record_path}: the notices are not frozen (scripts/82_a4_freeze_notices.py)")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    path = resolve(record["file"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{path} does not match its freeze record")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------- evaluated version

VERSIONS = {"v2.6": ("a4", "a4"), "v2.7": ("a4_v27", "a4_v27")}   # version -> (data directory, results prefix)


def a4_version() -> str:
    """The version of JurisMap whose Phase 8 outputs a script reads and writes: JM_A4_VERSION, default v2.6."""
    import os

    version = os.getenv("JM_A4_VERSION", "v2.6")
    if version not in VERSIONS:
        raise ValueError(f"JM_A4_VERSION must be one of {tuple(VERSIONS)}, got {version!r}")
    return version


def work_dir():
    """data/v2/a4 (v2.6) or data/v2/a4_v27 (v2.7): drafts, judgments and working files of the version."""
    path = resolve(load_config("paths")["data"]["v2"]) / VERSIONS[a4_version()][0]
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(name: str):
    """results/<name> for v2.6; results/a4_v27_<rest> for v2.7 (name starts with "a4_")."""
    prefix = VERSIONS[a4_version()][1]
    if name.startswith("a4_") and prefix != "a4":
        name = prefix + name[2:]
    elif name.startswith("figures/a4_") and prefix != "a4":
        name = "figures/" + prefix + name[len("figures/a4"):]
    return resolve(load_config("paths")["results"]) / name
