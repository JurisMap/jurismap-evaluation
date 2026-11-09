"""Judge prompts of the A4 drafting evaluation (Phase 8, tasks 4 to 6) and their answer checks.

Two calls per draft and judge (config/a4_drafting.yaml `judging`):
- RUBRIC: the five criteria of the plan, 1 to 5, from the draft, the notice and the headnotes of the
  supplied precedents; for "fidelity to facts" the judge also lists each fact about the case that the
  draft states and the notice does not (dates, amounts, events, documents, names, numbers). Those
  facts, minus the ones the fact guard replaced or flagged, are the invented facts the guard missed.
- ARGUMENTS: for each of the real appellant's arguments (material of task 5), whether the draft
  raises it: "raised", "partly" or "absent".
The judge is never told the drafting condition. Prompts stay under the step's token limit.
"""

from __future__ import annotations

import re

from jm_eval.chatbot.judging import field_text

PROMPTS_VERSION = 1
CRITERIA = ("legal_accuracy", "fidelity_to_facts", "use_of_precedents", "structure", "persuasiveness")
RECALL = ("raised", "partly", "absent")

SYSTEM = ("You evaluate draft appeals to the CRSNSP, the Brazilian administrative appeals council for sanctions in "
          "insurance. Be strict and literal: judge only from the texts you are given. Answer with JSON only.")

RUBRIC_PROMPT = """Below are a CASE DOCUMENT (the infraction notice and SUSEP's first-instance decision), the HEADNOTES of the CRSNSP precedents the drafting assistant was given (possibly none), and a DRAFT APPEAL to the CRSNSP written for the party charged, all in Portuguese. Placeholders such as "[A PREENCHER: ...]" mark details the drafter left for the lawyer; they are neither errors nor facts.

Score the draft from 1 (very poor) to 5 (excellent) on each criterion:
- legal_accuracy: the legal provisions, institutes and procedural rules it invokes exist and are applied correctly; no misstatement of the law.
- fidelity_to_facts: it states the facts of the case as the case document gives them; it neither contradicts nor adds facts about the case.
- use_of_precedents: it cites CRSNSP precedents that fit the case and uses them to support specific points; citing none where precedents would help, or citing ill-fitting ones, scores low.
- structure: clear, complete and orderly as an appeal (qualification, admissibility, facts, merits, law, case law, requests).
- persuasiveness: the strength of its arguments for the party, as a CRSNSP counsellor would read them.

Then list in "added_facts" each fact about THIS case that the draft states and the case document does not state: a date, an amount, an event, a document, a name or a number concerning the party or its case. Do not list legal provisions, precedents, general statements of law, arguments, requests, or figures the draft computes from amounts in the case document. Quote the draft's words for each (at most 20 words).

Return JSON only: {{"scores": {{"legal_accuracy": <1-5>, "fidelity_to_facts": <1-5>, "use_of_precedents": <1-5>, "structure": <1-5>, "persuasiveness": <1-5>}}, "added_facts": ["...", "..."], "reason": "<one sentence on the main weakness>"}}

CASE DOCUMENT:
{notice}

PRECEDENT HEADNOTES:
{precedents}

DRAFT APPEAL:
{draft}"""

ARGUMENTS_PROMPT = """Below are the numbered ARGUMENTS that the party's real appeal to the CRSNSP raised in this case, and a DRAFT APPEAL written by an assistant for the same case without seeing the real appeal, both in Portuguese.

For each argument, say whether the draft raises it:
- "raised": the draft makes the same argument in substance (the same legal or factual point), whatever the wording.
- "partly": the draft touches the point but in a different or much weaker form, or only as a passing mention.
- "absent": the draft does not make the point.
Judge each argument on its own. A generic request (e.g. "reform the decision") does not raise a specific argument.

Return JSON only: {{"arguments": [{{"id": <argument number>, "label": "raised" or "partly" or "absent"}}]}}

ARGUMENTS OF THE REAL APPEAL:
{arguments}

DRAFT APPEAL:
{draft}"""


def precedent_headnotes(rows: list[dict]) -> str:
    if not rows:
        return "(none were given)"
    return "\n".join(f"- {field_text(r['acordao_titulo'])}: {field_text(r['acordao_ementa'])}" for r in rows)


def rubric_prompt(notice: str, precedents: list[dict], draft: str) -> str:
    return RUBRIC_PROMPT.format(notice=notice.strip(), precedents=precedent_headnotes(precedents), draft=draft.strip())


def arguments_prompt(arguments: list[str], draft: str) -> str:
    return ARGUMENTS_PROMPT.format(arguments="\n".join(f"{i}. {a}" for i, a in enumerate(arguments, 1)), draft=draft.strip())


def validate_rubric(parsed):
    if not isinstance(parsed, dict) or not isinstance(parsed.get("scores"), dict):
        return None, "expected scores"
    scores = parsed["scores"]
    if set(scores) != set(CRITERIA) or not all(isinstance(v, int) and 1 <= v <= 5 for v in scores.values()):
        return None, f"expected integer scores 1-5 for {CRITERIA}"
    facts = parsed.get("added_facts")
    if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
        return None, "expected a list of added facts"
    return {"scores": scores, "added_facts": [f.strip() for f in facts if f.strip()],
            "reason": str(parsed.get("reason") or "").strip()}, None


def validate_arguments(parsed, n: int):
    items = parsed.get("arguments") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return None, "expected a list of arguments"
    labels = {}
    for x in items:
        if not isinstance(x, dict) or not isinstance(x.get("id"), int) or x.get("label") not in RECALL:
            return None, "each argument needs an integer id and a label raised/partly/absent"
        labels[x["id"]] = x["label"]
    if sorted(labels) != list(range(1, n + 1)):
        return None, f"expected labels for arguments 1 to {n}"
    return {"labels": [labels[i] for i in range(1, n + 1)]}, None


PLACEHOLDER = re.compile(r"\[\s*A\s+PREENCHER|\bX{3,}\b|\[[^\]]*(?:nome|n[úu]mero|data|endere[çc]o|inserir)[^\]]*\]", re.I)


def added_facts(answer: dict | None) -> list[str]:
    """The judge's added facts without placeholders, which the prompt excludes but Mistral lists anyway
    (development run on the fictitious notice)."""
    return [f for f in (answer or {}).get("added_facts", []) if not PLACEHOLDER.search(f)]


def missed_by_guard(facts: list[str], fact_guard: dict | None) -> list[str]:
    """Added facts in which no text the fact guard replaced or flagged occurs."""
    caught = [f["text"] for f in (fact_guard or {}).get("findings", []) if f.get("text")]
    return [fact for fact in facts if not any(c in fact for c in caught)]


# ---------------------------------------------------------------------------- second pass: what the added facts are

FACT_KINDS = ("invented", "assertion", "procedural", "in_notice")

FACT_KINDS_PROMPT = """Below are a CASE DOCUMENT (the infraction notice and SUSEP's first-instance decision, in Portuguese) and numbered ITEMS: passages of a draft appeal for the party charged, which a reviewer listed as stating something about the case that the case document does not state.

Classify each item:
- "invented": a specific fact about the party or its case (an event, a date, an amount, a document, a name, a number, a procedural act, a circumstance of the facts) that the case document does not state and the item presents as a fact.
- "assertion": the party's own characterisation of its conduct, intent or situation (good faith, no intent, diligence, cooperation, no harm, no profit, regular history), or an argument or legal qualification, without a specific checkable fact.
- "procedural": what any appeal implies and the case document need not state: that the party was notified of the decision, that the appeal is filed in time, that it is represented by lawyers.
- "in_notice": the case document states it, or it follows directly from what the case document states.
An item that restates a fact of the case document in other words, or gives it a slant favourable to the party, is "in_notice" (or "assertion" for the slant), not "invented". When an item mixes kinds, choose "invented" if it contains any specific invented fact.

Return JSON only: {{"items": [{{"id": <item number>, "kind": "invented" or "assertion" or "procedural" or "in_notice"}}]}}

CASE DOCUMENT:
{notice}

ITEMS:
{items}"""


def fact_kinds_prompt(notice: str, items: list[str]) -> str:
    return FACT_KINDS_PROMPT.format(notice=notice.strip(), items="\n".join(f"{i}. {t}" for i, t in enumerate(items, 1)))


def validate_fact_kinds(parsed, n: int):
    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return None, "expected a list of items"
    kinds = {}
    for x in items:
        if not isinstance(x, dict) or not isinstance(x.get("id"), int) or x.get("kind") not in FACT_KINDS:
            return None, f"each item needs an integer id and a kind among {FACT_KINDS}"
        kinds[x["id"]] = x["kind"]
    if sorted(kinds) != list(range(1, n + 1)):
        return None, f"expected kinds for items 1 to {n}"
    return {"kinds": [kinds[i] for i in range(1, n + 1)]}, None
