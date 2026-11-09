"""Phase 6, A3 (task 4): deterministic scoring of the answers. No judge scores a known answer.

- Templated questions: the gold value against the answer, by code (jm_eval.chatbot.scoring).
- Unanswerable questions: abstention by the closed list of patterns of config/a3_chatbot.yaml. Only
  an answer that the patterns cannot classify goes to Mistral Medium 3, with the question and the
  answer alone.
- Citation check, every answer: each decision the answer names (by process, appeal or acórdão
  number) must be one of the 10 supplied decisions. A number that is in the question, or that the supplied texts themselves cite, is
  counted apart; a number found nowhere is unknown.

Outputs (never overwritten):
  data/v2/a3/deterministic.parquet      one row per answer
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from jm_eval.chatbot import judging as J
from jm_eval.chatbot import questions as Q
from jm_eval.chatbot import scoring as S
from jm_eval.chatbot import store
from jm_eval.io import load_config, new_artefact_path
from jm_eval.manifest import RunManifest
from jm_eval.retrieval.labels import process_keys
from jm_eval.retrieval.provisions import build_canon

SCRIPT = "63_a3_deterministic"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--candidate", help="Phase 6.1: a strategy of config/a3_candidates.yaml, on the development sets")
    ap.add_argument("--setting", help="Phase 7: a context size of config/a5_context_size.yaml, on the evaluated questions")
    args = ap.parse_args()
    args.dev = args.dev or bool(args.candidate)
    store.use_candidate(args.candidate)
    store.use_setting(args.setting)
    cfg = load_config("a3_chatbot")
    scoring, work = cfg["scoring"], store.work_dir(args.dev)
    with RunManifest(SCRIPT, work, models={"abstention_fallback": load_config("models")["judges"]["judge_b"]["model"]},
                     params={"dev": args.dev, "setting": args.setting, "scoring": scoring,
                             "prompts_version": J.PROMPTS_VERSION}) as run:
        questions = {q["question_id"]: q for q in store.load_questions(args.dev)}
        answers = store.load_table("answers.parquet", args.dev)
        corpus = store.load_corpus()
        canon = build_canon(corpus["acordao_base"])
        by_hash = corpus.set_index("acordao_hash", drop=False)
        asker = J.Asker("judge_b", "abstention", dev=args.dev)
        run.data["calls"] = asker.calls
        context: dict[str, tuple[set[str], str]] = {}
        rows = []
        for a in answers.to_dict("records"):
            q = questions[a["question_id"]]
            if q["set_id"] not in context:
                members = by_hash.loc[q["case_hashes"]]
                supplied = set().union(*(process_keys(Q.clean_title(t), u) for t, u in zip(members["acordao_titulo"], members["acordao_url"])))
                supplied |= {n for n in members["acordao_acordao"].map(S.own_acordao_number) if n}
                context[q["set_id"]] = (supplied, Q.context_text(members))
            supplied, text = context[q["set_id"]]
            row = {"answer_id": a["answer_id"], "question_id": a["question_id"], "set_id": a["set_id"], "type": a["type"],
                   "subtype": a["subtype"], "run": a["run"], "score": None, "correct": None, "scored_by": None, "details": None}
            if q["type"] == "templated":
                result = S.score_templated(q, a["answer"], canon=canon, tolerance=scoring["amount_tolerance"])
                row.update(score=result.pop("score"), correct=result.pop("correct"), scored_by="code",
                           details=json.dumps(result, ensure_ascii=False))
            elif q["type"] == "unanswerable":
                result = S.score_unanswerable(a["answer"], q["subtype"], scoring["abstention_patterns"])
                row.update(scored_by="code")
                if result["status"] == "unclassified":
                    verdict = asker.ask(J.ABSTENTION_PROMPT.format(question=q["question"], answer=a["answer"]),
                                        lambda p: J.validate_label(p, ("abstained", "answered")), item=a["answer_id"])
                    status = verdict["label"] if verdict else None
                    result = {"status": status, "score": None if status is None else float(status == "abstained"),
                              "correct": None if status is None else status == "abstained"}
                    row.update(scored_by="judge_b (patterns could not classify)")
                row.update(score=result["score"], correct=result["correct"], details=json.dumps({"status": result["status"]}))
            citations = S.citation_check(a["answer"], supplied, q["question"], text)
            row.update({f"cited_{k}": citations[k] for k in ("supplied", "in_question", "in_context", "unknown")},
                       cited=citations["cited"], citations_all_supplied=citations["all_supplied"],
                       citations_none_unknown=citations["none_unknown"],
                       unknown_numbers=json.dumps(citations["unknown_numbers"]))
            rows.append(row)
        table = pd.DataFrame(rows)
        path = new_artefact_path(work / "deterministic.parquet")
        table.to_parquet(path, index=False)
        run.add_output(path, "A3: deterministic scores and citation check" + (" (development set)" if args.dev else ""))
        templated, unanswerable = table[table["type"] == "templated"], table[table["type"] == "unanswerable"]
        summary = {
            "templated": {"answers": len(templated), "correct": int(templated["correct"].sum()),
                          "mean_score": round(float(templated["score"].mean()), 4) if len(templated) else None,
                          "by_subtype": {k: f"{int(v['correct'].sum())}/{len(v)}" for k, v in templated.groupby("subtype")}},
            "unanswerable": {"answers": len(unanswerable), "abstained": int((unanswerable["correct"] == True).sum()),  # noqa: E712
                             "scored_by_code": int((unanswerable["scored_by"] == "code").sum())},
            "citations": {"answers": len(table), "answers_citing": int((table["cited"] > 0).sum()),
                          "numbers_cited": int(table["cited"].sum()), "supplied": int(table["cited_supplied"].sum()),
                          "in_question": int(table["cited_in_question"].sum()),
                          "in_context": int(table["cited_in_context"].sum()), "unknown": int(table["cited_unknown"].sum()),
                          "answers_with_an_unknown_number": int((~table["citations_none_unknown"]).sum())},
            "abstention_calls": J.calls_summary(asker.calls)}
        run.data["summary"] = summary
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
