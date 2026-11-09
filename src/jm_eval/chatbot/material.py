"""Reference material of the open questions (Phase 6, task 2; reused by Phase 7), by Mistral Medium 3.

Two small steps (never Gemini; ground rule 7): the key points of each decision, one decision per
call, then the reference points of an open question from the key points of its context's decisions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from jm_eval.chatbot import judging as J


def key_points(asker: J.Asker, rows: pd.DataFrame, cfg: dict, workers: int = 3) -> dict[str, list[str]]:
    """Key points of each decision, one call per decision."""
    spec = cfg["material"]

    def one(row: dict) -> tuple[str, list[str]]:
        text = J.decision_text(row, spec["report_chars"], spec["vote_chars"])
        prompt = J.KEY_POINTS_PROMPT.format(decision=text, **spec["key_points"])
        answer = asker.ask(prompt, lambda p: J.validate_key_points(p, spec["key_points"]["min"], spec["key_points"]["max"]),
                           item=row["acordao_hash"])
        if answer is None:
            raise RuntimeError(f"no usable key points for {row['acordao_hash']}")
        return row["acordao_hash"], answer["key_points"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, rows.to_dict("records")))


def reference_points(asker: J.Asker, question: dict, points: dict[str, list[str]], titles: dict[str, str], cfg: dict) -> list[str]:
    """Reference points of one open question from the key points of the decisions in `question["case_hashes"]`."""
    spec = cfg["material"]["reference_points"]
    blocks = "\n\n".join(f"{titles[h]}:\n" + "\n".join(f"- {p}" for p in points[h]) for h in question["case_hashes"])
    prompt = J.REFERENCE_PROMPT.format(n=len(question["case_hashes"]), question=question["question"],
                                       key_points=blocks, **spec)
    answer = asker.ask(prompt, lambda p: J.validate_reference(p, spec["min"], spec["max"]), item=question["question_id"])
    if answer is None:
        raise RuntimeError(f"no usable reference for {question['question_id']}")
    return answer["reference_points"]
