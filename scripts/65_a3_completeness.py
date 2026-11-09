"""Phase 6, A3 (task 6): completeness of the answers to the open questions.

Each judge reads the question, the reference points (written by Mistral Medium 3 from the key
points of the set's decisions, scripts/61) and the answer, and says for each point whether the
answer covers it: covered, partly or missing. Coverage of an answer = (covered + 0.5 x partly) /
points; the strict figure counts only "covered". Judges: GPT-5.4 and Mistral Medium 3; Gemini 3.8
Flash is reported separately. Mistral wrote the references and also judges them: the report
compares its scores with GPT-5.4's.

Outputs (never overwritten):
  data/v2/a3/completeness.parquet      one row per answer, judge and reference point
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from jm_eval.chatbot import judging as J
from jm_eval.chatbot import metrics as M
from jm_eval.chatbot import store
from jm_eval.io import load_config, new_artefact_path
from jm_eval.manifest import RunManifest

SCRIPT = "65_a3_completeness"


def judge_answers(asker: J.Asker, answers: list[dict], questions: dict, workers: int = 3) -> list[dict]:
    def one(a: dict) -> list[dict]:
        q = questions[a["question_id"]]
        points = q["reference_points"]
        prompt = J.COMPLETENESS_PROMPT.format(question=q["question"], answer=a["answer"],
                                              points="\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)))
        result = asker.ask(prompt, lambda p: J.validate_coverage(p, len(points)), item=a["answer_id"])
        coverage = result["coverage"] if result else [None] * len(points)
        return [{"answer_id": a["answer_id"], "question_id": a["question_id"], "set_id": a["set_id"], "run": a["run"],
                 "subtype": a["subtype"], "judge": asker.role, "point": i, "reference_point": p, "coverage": c}
                for i, (p, c) in enumerate(zip(points, coverage), 1)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for rows in pool.map(one, answers) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--candidate", help="Phase 6.1: a strategy of config/a3_candidates.yaml, on the development sets")
    ap.add_argument("--setting", help="Phase 7: a context size of config/a5_context_size.yaml, on the evaluated questions")
    ap.add_argument("--judges", nargs="+", help="only these judges of the config (a later run adds the others from the cache)")
    args = ap.parse_args()
    args.dev = args.dev or bool(args.candidate)
    store.use_candidate(args.candidate)
    store.use_setting(args.setting)
    cfg = load_config("a3_chatbot")
    spec, work, judges = cfg["completeness"], store.work_dir(args.dev), load_config("models")["judges"]
    names = spec["judges"] + ([] if args.candidate else spec["separate"])      # the selection uses the two judges only
    if args.setting:                         # Phase 7: the judges of config/a5_context_size.yaml
        names = load_config("a5_context_size")["judges"]["completeness"]
    names = [n for n in names if n in args.judges] if args.judges else names
    with RunManifest(SCRIPT, work, models={n: judges[n]["model"] for n in names},
                     params={"dev": args.dev, "setting": args.setting, "completeness": spec,
                             "prompts_version": J.PROMPTS_VERSION}) as run:
        questions = {q["question_id"]: q for q in store.load_questions(args.dev)}
        answers = store.load_table("answers.parquet", args.dev)
        answers = answers[answers["type"] == "open"].to_dict("records")
        calls: list = []
        run.data["calls"] = calls
        limits = load_config("a5_context_size")["prompt_limits"]["completeness"] if args.setting else {}
        askers = {name: J.Asker(name, "completeness", dev=args.dev, log=calls, **limits) for name in names}
        with ThreadPoolExecutor(max_workers=len(askers)) as pool:
            futures = [pool.submit(judge_answers, asker, answers, questions) for asker in askers.values()]
        table = pd.DataFrame([r for f in futures for r in f.result()])
        path = new_artefact_path(work / "completeness.parquet")
        table.to_parquet(path, index=False)
        run.add_output(path, "A3: coverage of the reference points" + (" (development set)" if args.dev else ""))

        per_answer = (table.dropna(subset=["coverage"]).groupby(["judge", "answer_id"])["coverage"]
                      .apply(lambda c: M.coverage_score(list(c))["coverage"]).reset_index())
        summary = {"answers": len(answers), "reference_points": int(table[table["judge"] == names[0]].shape[0]),
                   "unusable_answers": int(table["coverage"].isna().sum()),
                   "mean_coverage": {j: round(float(v["coverage"].mean()), 4) for j, v in per_answer.groupby("judge")},
                   "calls": {name: J.calls_summary(a.calls) for name, a in askers.items()},
                   "max_input_tokens": max((c["input_tokens"] or 0 for c in calls), default=0)}
        run.data["summary"] = summary
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
