"""Prompts, answer checks and the call log of every model call of the A3 evaluation (Phase 6).

Roles (config/a3_chatbot.yaml): Mistral Medium 3 writes the test material (key points of each
decision, reference points of each open question) and splits answers into claims; GPT-5.4 and
Mistral judge claims against evidence, coverage of the reference points and the similarity of
pairs of reports; Gemini 3.8 Flash judges the same items and is reported separately.

No prompt may exceed `max_prompt_tokens` (the judges' per-minute quotas): `Asker` refuses a prompt
whose estimate is above the limit and logs the API's own token count of every call.
Only public CRSNSP text and generated text is sent (ground rule 1).
"""

from __future__ import annotations

import re
import threading

from jm_eval.cache import cache_key
from jm_eval.io import load_config
from jm_eval.llm import Judge, get_model, parse_json

PROMPTS_VERSION = 1
MAX_FORMAT_RETRIES = 2
LABELS = ("supported", "contradicted", "not_found")
COVERAGE = ("covered", "partly", "missing")

SYSTEM = ("You work on decisions of the CRSNSP, the Brazilian administrative appeals council for sanctions in "
          "insurance. Be literal and strict: use only the text you are given. Answer with JSON only.")

KEY_POINTS_PROMPT = """Below are fields of one CRSNSP decision, in Portuguese: the headnote, the outcome, the penalty, and the rapporteur's report and vote (a long text is cut in the middle, marked [...]).

List the {min} to {max} key points someone must know about this decision. Cover, where the text states them: (a) the conduct the party was accused of; (b) the appellant's main arguments; (c) the grounds on which the rapporteur decided them; (d) the outcome and what happened to the penalty.

Rules:
- Each key point is one self-contained sentence in Portuguese.
- State only what the text states. No opinions, no outside knowledge.
- Do not name the decision (no process or appeal number).

Return JSON only: {{"key_points": ["...", "..."]}}

{decision}"""

REFERENCE_PROMPT = """A user asked a QUESTION about a set of {n} CRSNSP decisions. Below are the key points of each decision, in Portuguese, under the decision's title. They were written beforehand from the decision's text.

Write the reference for the answer: the {min} to {max} points that a complete answer to the question should cover.

Rules:
- Use only the key points below. Do not add knowledge.
- Each reference point is one sentence in Portuguese that answers part of the question.
- Where a point concerns particular decisions, name them by their title.
- Prefer points that hold for several decisions. Keep a point about a single decision only where the question needs it.
- Leave out any part of the question that the key points cannot answer.

Return JSON only: {{"reference_points": ["...", "..."]}}

QUESTION:
{question}

KEY POINTS OF THE DECISIONS:
{key_points}"""

DECOMPOSE_PROMPT = """Below is an ANSWER that an assistant gave about decisions of the CRSNSP, in Portuguese. Split it into atomic factual claims.

Rules:
- One claim is one fact that can be checked against the text of the decisions. Split a sentence that states several facts.
- Each claim must be understandable alone: replace pronouns and references such as "o primeiro caso" or "nesse processo" by what they refer to, and keep the decision the claim is about.
- Keep the answer's own wording. Do not add, correct or interpret.
- Leave out greetings, offers of help, statements about what the assistant can or cannot do, and sentences that only announce or wrap up without stating a fact.
- "processes": the process or appeal numbers, as written in the answer, of the decisions the claim is about. Use an empty list where the claim names none.

Return JSON only: {{"claims": [{{"claim": "...", "processes": ["..."]}}]}}

ANSWER:
{answer}"""

VERIFY_PROMPT = """Each item below has a CLAIM, taken from an assistant's answer about decisions of the CRSNSP, and EVIDENCE: numbered passages retrieved from those decisions, in Portuguese. Every passage starts with the title of its decision and the field it comes from.

Judge each claim against its own evidence only:
- "supported": the evidence states the claim, or the claim follows directly from it. Every part of the claim must be covered, and the passage must belong to the decision the claim is about.
- "contradicted": the evidence states something incompatible with the claim (another value, name, date or outcome, or the opposite).
- "not_found": the evidence neither states nor contradicts the claim, or covers only part of it.

Do not use outside knowledge.

Return JSON only: {{"verdicts": [{{"id": <item number>, "label": "supported" or "contradicted" or "not_found", "passage": <number of the passage that decides it, or null>}}]}}

{items}"""

FULL_TEXT_PROMPT = """Below are a CLAIM, taken from an assistant's answer about decisions of the CRSNSP, and the text of one decision, in Portuguese{part_note}.{scope_note}

Judge the claim against this text only:
- "supported": the text states the claim, or the claim follows directly from it. Every part of the claim must be covered.
- "contradicted": the text states something incompatible with the claim.
- "not_found": the text neither states nor contradicts the claim, or covers only part of it.

Do not use outside knowledge.

Return JSON only: {{"label": "supported" or "contradicted" or "not_found", "quote": "<at most 25 consecutive words of the text that decide it, or empty>"}}

CLAIM:
{claim}

DECISION:
{decision}"""

COMPLETENESS_PROMPT = """Below are a QUESTION about a set of CRSNSP decisions, the REFERENCE POINTS that a complete answer should cover, and the ANSWER an assistant gave, in Portuguese.

For each reference point, say how far the answer covers it:
- "covered": the answer states the point. The substance must be the same; the wording may differ.
- "partly": the answer states part of the point, or states it only vaguely.
- "missing": the answer does not state it.

Judge coverage only. Do not judge whether the answer is right, and ignore what it adds beyond the points.

Return JSON only: {{"points": [{{"id": <point number>, "coverage": "covered" or "partly" or "missing"}}]}}

QUESTION:
{question}

REFERENCE POINTS:
{points}

ANSWER:
{answer}"""

ABSTENTION_PROMPT = """A user asked an assistant a QUESTION about a set of decisions. What the question asks for is not in the data the assistant has. Below is the ANSWER, in Portuguese. Did the assistant abstain?
- "abstained": it says that it does not have, or cannot find, what was asked, and gives no value for it.
- "answered": it gives a value for what was asked, whatever the caveats.

Return JSON only: {{"label": "abstained" or "answered"}}

QUESTION:
{question}

ANSWER:
{answer}"""

PAIR_PROMPT = """Below are the rapporteur's reports of two CRSNSP decisions, A and B, in Portuguese (a long report is cut). Each decision judges an appeal against a sanction for an infraction of insurance regulation.

Rate how similar the infractions of the two decisions are, on a scale from 0 to 5:
0 = not similar at all; 1 = very little similar; 2 = little similar; 3 = moderately similar; 4 = similar; 5 = very similar.

Rate the infractions themselves (the conduct and the rule infringed), not the parties, the procedure, the arguments or the outcome.

Return JSON only: {{"similarity": <integer from 0 to 5>, "reason": "<one sentence>"}}

REPORT A:
{a}

REPORT B:
{b}"""


SEVERAL_DECISIONS_NOTE = (" The claim is about several decisions: judge only what it says about this one, as if the "
                          "others were not mentioned.")


def cut_middle(text: str | None, head: int, tail: int) -> str:
    """A text longer than head + tail characters keeps its head and its tail."""
    text = (text or "").strip()
    return text if len(text) <= head + tail else f"{text[:head].rstrip()}\n[...]\n{text[-tail:].lstrip()}"


def field_text(value) -> str:
    """A field as text: lists joined, missing values empty."""
    if isinstance(value, str):
        return re.sub(r"[ \t\xa0]+", " ", value).strip()
    if value is None or isinstance(value, (float, int)):
        return "" if value is None or value != value else str(value)
    return ", ".join(str(v).strip() for v in value if str(v).strip())


def decision_text(row, report_chars: dict | None = None, vote_chars: dict | None = None) -> str:
    """Title, short fields, report and vote of a decision, as a judge reads them."""
    report, vote = field_text(row["acordao_relatorio"]), field_text(row["acordao_voto_relator"])
    if report_chars:
        report = cut_middle(report, **report_chars)
    if vote_chars:
        vote = cut_middle(vote, **vote_chars)
    return "\n".join([
        f"TÍTULO: {field_text(row['acordao_titulo'])}",
        f"EMENTA: {field_text(row['acordao_ementa'])}",
        f"RESULTADO: {field_text(row['acordao_resultado'])}",
        f"PENALIDADE: {field_text(row['acordao_penalidade'])}",
        f"BASE LEGAL: {field_text(row['acordao_base'])}",
        f"RELATÓRIO:\n{report}",
        f"VOTO DO RELATOR:\n{vote}",
    ])


# ---------------------------------------------------------------------------- answer checks

def _strings(value, low: int, high: int) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [v.strip() for v in value if isinstance(v, str) and v.strip()]
    return items if low <= len(items) <= high and len(items) == len(value) else None


def validate_key_points(parsed, low: int, high: int):
    points = _strings(parsed.get("key_points") if isinstance(parsed, dict) else None, low, high)
    return ({"key_points": points}, None) if points else (None, f"expected {low} to {high} key points")


def validate_reference(parsed, low: int, high: int):
    points = _strings(parsed.get("reference_points") if isinstance(parsed, dict) else None, low, high)
    return ({"reference_points": points}, None) if points else (None, f"expected {low} to {high} reference points")


def validate_claims(parsed):
    claims = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(claims, list):
        return None, "no list of claims"
    out = []
    for c in claims:
        if not isinstance(c, dict) or not isinstance(c.get("claim"), str) or not c["claim"].strip():
            return None, "a claim has no text"
        processes = c.get("processes") or []
        if not isinstance(processes, list):
            return None, "processes is not a list"
        out.append({"claim": c["claim"].strip(), "processes": [str(p).strip() for p in processes if str(p).strip()]})
    return {"claims": out}, None                    # an answer may hold no factual claim


def validate_verdicts(parsed, n: int):
    verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
    if not isinstance(verdicts, list):
        return None, "no list of verdicts"
    by_id = {}
    for v in verdicts:
        if not isinstance(v, dict) or v.get("label") not in LABELS:
            return None, "a verdict has no valid label"
        try:
            by_id[int(v.get("id"))] = v
        except (TypeError, ValueError):
            return None, "a verdict has no item number"
    if sorted(by_id) != list(range(1, n + 1)):
        return None, f"expected verdicts for items 1 to {n}"
    return {"verdicts": [{"label": by_id[i]["label"],
                          "passage": by_id[i].get("passage") if isinstance(by_id[i].get("passage"), int) else None}
                         for i in range(1, n + 1)]}, None


def validate_label(parsed, allowed=LABELS):
    if not isinstance(parsed, dict) or parsed.get("label") not in allowed:
        return None, "no valid label"
    return {"label": parsed["label"], "quote": str(parsed.get("quote") or "")}, None


def validate_coverage(parsed, n: int):
    points = parsed.get("points") if isinstance(parsed, dict) else None
    if not isinstance(points, list):
        return None, "no list of points"
    by_id = {}
    for p in points:
        if not isinstance(p, dict) or p.get("coverage") not in COVERAGE:
            return None, "a point has no valid coverage"
        try:
            by_id[int(p.get("id"))] = p["coverage"]
        except (TypeError, ValueError):
            return None, "a point has no number"
    if sorted(by_id) != list(range(1, n + 1)):
        return None, f"expected points 1 to {n}"
    return {"coverage": [by_id[i] for i in range(1, n + 1)]}, None


def validate_similarity(parsed):
    value = parsed.get("similarity") if isinstance(parsed, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 5 or int(value) != value:
        return None, "similarity is not an integer from 0 to 5"
    return {"similarity": int(value), "reason": str(parsed.get("reason") or "")}, None


# ---------------------------------------------------------------------------- calls

class PromptTooLong(ValueError):
    pass


def estimated_tokens(prompt: str, system: str | None = SYSTEM) -> int:
    cfg = load_config("a3_chatbot")
    return int((len(prompt) + len(system or "")) / cfg["chars_per_token"])


class Asker:
    """One model in one step: paced calls, checked answers, format retries and a call log."""

    def __init__(self, role: str, step: str, *, model: Judge | None = None, dev: bool = False, log: list | None = None,
                 max_tokens: int | None = None, rpm: float | None = None):
        """`max_tokens` and `rpm` replace the config's prompt limit and pacing for this step (Phase 7)."""
        cfg = load_config("a3_chatbot")
        pacing = (cfg.get("pacing") or {}).get(role, {})
        rpm = rpm or pacing.get(step, pacing.get("default"))
        self.role, self.step, self.dev = role, step, dev
        self.model = model or get_model("judges", role, **({"quota": {"rpm": rpm}} if rpm else {}))
        self.max_tokens = max_tokens or cfg["max_prompt_tokens"]
        self._slots = threading.Semaphore(pacing.get("concurrency", 8))      # calls of this model in flight at once
        self.calls: list[dict] = []
        self.log = log                               # a run's shared call log: it reaches the manifest even if the run fails
        self._lock = threading.Lock()

    def ask(self, prompt: str, validate, *, item: str, extra: dict | None = None, system: str | None = SYSTEM,
            as_json: bool = True):
        """The checked answer, or None when every attempt was unusable. `as_json=False` checks the raw text."""
        estimate = estimated_tokens(prompt, system)
        if estimate > self.max_tokens:
            raise PromptTooLong(f"{self.step} {item}: about {estimate} tokens, above the limit of {self.max_tokens}")
        for attempt in range(MAX_FORMAT_RETRIES + 1):
            key = {"a3": PROMPTS_VERSION, "step": self.step, **(extra or {}), **({"format_retry": attempt} if attempt else {})}
            problem, answer = None, None
            with self._slots:
                result = self.model.complete(prompt, system=system, cache_extra=key)
            try:
                answer, problem = validate(parse_json(result.text) if as_json else result.text)
            except ValueError as e:                      # not JSON
                problem = f"not JSON: {e}"
            params = getattr(self.model, "params", dict)()
            record = {
                "step": self.step, "role": self.role, "model": self.model.model, "item": item, "attempt": attempt,
                "cached": result.cached, "estimated_tokens": estimate, "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens, "reasoning_tokens": result.reasoning_tokens,
                "cost_usd": result.cost_usd, "latency_s": result.latency_s, "credentials": result.credentials,
                "problem": problem, "dev": self.dev,
                "cache_entry": cache_key(self.model.model, {"system": system, "prompt": prompt}, {**params, **key})}
            with self._lock:
                self.calls.append(record)
                if self.log is not None:
                    self.log.append(record)
            if answer is not None:
                return answer
        return None


def calls_summary(calls: list[dict]) -> dict:
    """Totals of a call log: what was paid in this run, and the largest prompt the API counted."""
    paid = [c for c in calls if c["cached"] is False]
    tokens = [c["input_tokens"] for c in calls if c["input_tokens"]]
    by_credentials: dict[str, int] = {}
    for c in paid:
        by_credentials[c["credentials"] or "unknown"] = by_credentials.get(c["credentials"] or "unknown", 0) + 1
    return {"calls": len(calls), "calls_not_cached": len(paid),
            "unusable_answers": sum(c["problem"] is not None for c in calls),
            "input_tokens": sum(c["input_tokens"] or 0 for c in paid),
            "output_tokens": sum(c["output_tokens"] or 0 for c in paid),
            "cost_usd": round(sum(c["cost_usd"] or 0 for c in paid), 4),
            "max_input_tokens": max(tokens) if tokens else None,
            "served_by": by_credentials}
