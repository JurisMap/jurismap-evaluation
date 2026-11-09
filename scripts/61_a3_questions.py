"""Phase 6, A3 (task 2): the question set and its reference material.

Per case set: 4 templated questions (the answer is a field of a decision, or a count), 1
unanswerable question and 2 open questions; 42 in all (config/a3_chatbot.yaml).

Reference material for the open questions, in two small steps, by Mistral Medium 3 (never Gemini;
ground rule 7): the key points of each decision of a set, one decision per call, then the reference
points of each open question from the key points of its set.

Steps:
  (default)   build the draft and the sheet for the author's check of 20% of the questions
  --freeze    after the check: write the frozen set and its hash. No chatbot run reads a draft.
  --dev       the same questions for the development set (data/v2/a3_dev/); nothing is frozen

Outputs (never overwritten):
  data/v2/a3/key_points.jsonl, data/v2/a3/questions_draft.jsonl
  results/a3_questions_spot_check.xlsx      the sample to check (`ok`, `comment`), every question,
                                            the decisions of each set and their key points
  --freeze: data/v2/a3_questions.jsonl, results/a3_questions.csv, results/a3_questions_freeze.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from jm_eval.chatbot import judging as J
from jm_eval.chatbot.material import key_points, reference_points
from jm_eval.chatbot import questions as Q
from jm_eval.chatbot import store
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, sha256_file
from jm_eval.manifest import RunManifest
from jm_eval.retrieval.provisions import build_canon

SCRIPT = "61_a3_questions"


def gold_text(question: dict, corpus: pd.DataFrame) -> str:
    """The gold value as a person reads it in the sheet."""
    if question["type"] == "open":
        return "\n".join(f"{i}. {p}" for i, p in enumerate(question["reference_points"], 1))
    if question["type"] == "unanswerable":
        return f"(abstention expected: {question['absence']})"
    gold = question["gold"]
    if question["subtype"] == "session_date":
        return f"days {gold['days']}, month {gold['month']}, year {gold['year']}"
    if question["subtype"] == "penalty":
        return f"types {gold['types']}, amounts {gold['amounts']}"
    return "; ".join(gold) if isinstance(gold, list) else str(gold)


SOURCE_FIELD = {"outcome": "acordao_resultado", "rapporteur": "acordao_relator", "appellant": "acordao_recorrente",
                "lawyer": "acordao_advogado", "session_date": "sessao_datas", "legal_basis": "acordao_base",
                "penalty": "acordao_penalidade"}


def sheet_rows(questions: list[dict], corpus: pd.DataFrame) -> pd.DataFrame:
    by_hash = corpus.set_index("acordao_hash")
    rows = []
    for q in questions:
        target, field = q.get("target_hash"), SOURCE_FIELD.get(q["subtype"])
        rows.append({
            "question_id": q["question_id"], "set_id": q["set_id"], "type": q["type"], "subtype": q["subtype"],
            "question": q["question"], "gold": gold_text(q, corpus),
            "source_field_value": J.field_text(by_hash.at[target, field]) if target and field else "",
            "session_year": int(by_hash.at[target, "ano_sessoes"]) if target and q["subtype"] == "session_date" else "",
            "decision_url": by_hash.at[target, "acordao_url"] if target else "",
        })
    return pd.DataFrame(rows)


def build(args, cfg: dict, run: RunManifest) -> None:
    dev = args.dev
    work, results = store.root_dir(dev), store.results_dir()
    corpus = store.load_corpus()
    canon = build_canon(corpus["acordao_base"])
    sets = store.load_case_sets(dev)
    excluded = store.all_set_decisions()
    questions = []
    for index, case_set in enumerate(sets):
        questions += Q.build_questions(case_set, index, corpus, cfg, canon, excluded, cfg["questions"]["seed"],
                                       every_type=dev)

    log: list = []
    run.data["calls"] = log
    asker = J.Asker(cfg["material"]["model"], "key_points", dev=dev, log=log)
    members = corpus[corpus["acordao_hash"].isin({h for s in sets for h in s["case_hashes"]})]
    points = key_points(asker, members, cfg)
    reference_asker = J.Asker(cfg["material"]["model"], "reference", dev=dev, log=log)
    titles = {h: Q.clean_title(t) for h, t in zip(corpus["acordao_hash"], corpus["acordao_titulo"])}
    for q in questions:
        if q["type"] == "open":
            q["reference_points"] = reference_points(reference_asker, q, points, titles, cfg)
    calls = asker.calls + reference_asker.calls

    points_path = new_artefact_path(work / "key_points.jsonl")
    store.write_jsonl(points_path, [{"acordao_hash": h, "title": titles[h], "key_points": p} for h, p in sorted(points.items())])
    run.add_output(points_path, "A3: key points of each decision of the case sets (Mistral Medium 3)")
    name = "questions.jsonl" if dev else "questions_draft.jsonl"
    questions_path = new_artefact_path(work / name)
    store.write_jsonl(questions_path, questions)
    run.add_output(questions_path, "A3: question set" + (" of the development set" if dev else " (draft, before the author's check)"))

    table = sheet_rows(questions, corpus)
    summary = {"questions": len(questions), "by_type": table["type"].value_counts().to_dict(),
               "templated_by_subtype": table[table["type"] == "templated"]["subtype"].value_counts().to_dict(),
               "material_calls": J.calls_summary(calls)}
    if not dev:
        sample = Q.spot_check_sample(questions, cfg["questions"]["spot_check_share"], cfg["questions"]["seed"])
        check = table[table["question_id"].isin(sample)].copy()
        check["ok"], check["comment"] = None, None
        set_rows = [{"set_id": s["set_id"], "title": titles[h], "outcome": corpus.set_index("acordao_hash").at[h, "acordao_resultado"],
                     "decision_url": corpus.set_index("acordao_hash").at[h, "acordao_url"], "acordao_hash": h,
                     "key_points": "\n".join(f"- {p}" for p in points[h])} for s in sets for h in s["case_hashes"]]
        sheet_path = new_artefact_path(results / "a3_questions_spot_check.xlsx")
        with pd.ExcelWriter(sheet_path, engine="openpyxl") as writer:
            check.to_excel(writer, sheet_name="check", index=False)
            table.to_excel(writer, sheet_name="all_questions", index=False)
            pd.DataFrame(set_rows).to_excel(writer, sheet_name="case_sets", index=False)
        run.add_output(sheet_path, "A3: sample of the question set for the author's check")
        summary["spot_check"] = sample
    run.data["summary"] = summary
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(table[["question_id", "subtype", "question", "gold"]].to_string(index=False, max_colwidth=90))


def freeze(args, cfg: dict, run: RunManifest) -> None:
    work, results = store.work_dir(), store.results_dir()
    if store.frozen_questions_path().exists():
        raise SystemExit(f"{store.frozen_questions_path()} exists: the question set is already frozen")
    draft_path = latest_artefact_path(work / "questions_draft.jsonl")
    questions = store.read_jsonl(draft_path)
    sheet_path = latest_artefact_path(results / "a3_questions_spot_check.xlsx")
    check = pd.read_excel(args.checked or sheet_path, sheet_name="check")
    run.add_input(draft_path)
    run.add_input(args.checked or sheet_path)
    marks = check["ok"].map(lambda v: str(v).strip().lower() if pd.notna(v) else "")
    unmarked = check.loc[marks == "", "question_id"].tolist()
    rejected = check.loc[marks.isin(["0", "0.0", "no", "n"]), "question_id"].tolist()
    if unmarked or rejected:
        raise SystemExit(f"not frozen: unmarked {unmarked}, rejected {rejected} (column `ok` of the sheet `check`)")
    known = {q["question_id"] for q in questions}
    if not set(check["question_id"]) <= known:
        raise SystemExit("the checked sheet does not belong to the newest draft")

    frozen = store.frozen_questions_path()
    store.write_jsonl(frozen, questions)
    run.add_output(frozen, "A3: frozen question set")
    table = sheet_rows(questions, store.load_corpus())
    table_path = new_artefact_path(results / "a3_questions.csv")
    table.to_csv(table_path, index=False)
    run.add_output(table_path, "A3: frozen question set (readable)")
    record = {"sha256": sha256_file(frozen), "questions": len(questions),
              "by_type": table["type"].value_counts().to_dict(), "draft": draft_path.name,
              "checked_by_hugo": check["question_id"].tolist(),
              "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    store.freeze_record_path().write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    run.add_output(store.freeze_record_path(), "A3: hash of the frozen question set")
    run.data["summary"] = record
    print(json.dumps(record, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true", help="questions of the development set")
    ap.add_argument("--freeze", action="store_true", help="freeze the newest draft after the author's check")
    ap.add_argument("--checked", help="the sheet the author marked (default: the newest spot-check sheet)")
    args = ap.parse_args()
    cfg = load_config("a3_chatbot")
    models = {"material": load_config("models")["judges"][cfg["material"]["model"]]["model"]}
    out_dir = store.work_dir(True) if args.dev else store.results_dir()
    with RunManifest(SCRIPT, out_dir, models=models, params={"dev": args.dev, "freeze": args.freeze,
                                                             "questions": cfg["questions"], "material": cfg["material"],
                                                             "prompts_version": J.PROMPTS_VERSION},
                     seeds={"questions": cfg["questions"]["seed"]}) as run:
        (freeze if args.freeze else build)(args, cfg, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
