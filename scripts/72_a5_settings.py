"""Phase 7, A5 (task 1): the contexts and questions of the context-size test.

The context of a question at size k is its case set's k decisions most similar to the seed
(config/a5_context_size.yaml): k = 10 is Phase 6's set, k = 5 its first five, and for k = 20 and 30
the seed's next most similar decisions are added by Phase 6's rule. The frozen Phase 6 templated and
open questions are asked at every size where Phase 6's rules still hold; counts are counted again.

Reference points of the open questions at k = 5, 20 and 30 are written as in Phase 6 (Mistral Medium
3, from the key points of the context's decisions). Phase 6's key points are reused; the decisions
added for k = 20 and 30 get theirs first, one call per decision. A sample of the new references goes
to the author's check. The k = 10 setting reuses Phase 6's answers (first run of each question); they are
scored again by scripts/63 to 65 like every setting, from the cache except for the retrieval-miss
control, which draws its own sample.

The decisions added here, like Phase 6's, are not to be used as Phase 8 test notices.

Outputs (never overwritten):
  data/v2/a5/case_sets.json                 each set's decisions in order of similarity to the seed
  data/v2/a5/key_points.jsonl               key points of the added decisions (Mistral Medium 3)
  data/v2/a5/<setting>/questions.jsonl      the questions of each setting, with their contexts
  data/v2/a5/k10/answers.parquet            Phase 6's answers, first runs only
  results/a5_case_sets.csv                  one row per set and size
  results/a5_references_spot_check.xlsx     the sample of new references to check (`ok`, `comment`)
  results/a5_questions_freeze.json          hash of each setting's question file, and the dropped questions
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from jm_eval.chatbot import context_size as CS
from jm_eval.chatbot import judging as J
from jm_eval.chatbot import questions as Q
from jm_eval.chatbot import store
from jm_eval.chatbot.material import key_points, reference_points
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, sha256_file
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C

SCRIPT = "72_a5_settings"


def reuse_phase6(spec: dict, phase6_dir, asked: list[dict], run: RunManifest) -> dict:
    """Phase 6's answers to the first run of the questions asked at k = 10."""
    ids = {q["question_id"] for q in asked}
    out_dir, counts = store.a5_dir() / "k10", {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("answers",):
        source = phase6_dir / spec[name]
        table = pd.read_parquet(source)
        table = table[(table["run"] == 1) & table["question_id"].isin(ids)].reset_index(drop=True)
        path = new_artefact_path(out_dir / f"{name}.parquet")
        table.to_parquet(path, index=False)
        run.add_input(source)
        run.add_output(path, "A5: Phase 6's answers, first run of each templated and open question (k = 10)")
        counts[name] = len(table)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build the contexts and questions, print them and stop (no model call)")
    args = ap.parse_args()
    cfg, cfg5 = load_config("a3_chatbot"), load_config("a5_context_size")
    spec = cfg5["extension"]
    models = {"material": load_config("models")["judges"][cfg["material"]["model"]]["model"]}
    with RunManifest(SCRIPT, store.results_dir(), models=models,
                     params={"a5": cfg5, "material": cfg["material"], "prompts_version": J.PROMPTS_VERSION},
                     seeds={"references_spot_check": cfg5["references"]["seed"]}) as run:
        labels = C.load_labels()
        vectors = C.method_vectors(spec["vector_method"]).astype(np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        corpus = store.load_corpus()
        assert labels["acordao_hash"].tolist() == corpus["acordao_hash"].tolist()
        sets = store.load_case_sets(False)
        frozen = store.load_questions(False)
        run.add_input(store.frozen_questions_path())

        # 1. Contexts: each set's members, then the seed's next most similar decisions.
        excluded = store.all_set_decisions() | {q["outside_hash"] for q in frozen if q.get("outside_hash")}
        extended = CS.extend_sets(sets, labels, vectors, spec["size"], excluded)
        for s, e in zip(sets, extended):                  # the members' similarity is Phase 6's
            assert np.allclose(s["similarity"], e["similarity"][:len(s["similarity"])], atol=1e-5), s["set_id"]
        added = [h for e in extended for h in e["ranked_hashes"][e["members"]:]]
        assert len(added) == len(set(added)) and not set(added) & excluded
        sets_path = new_artefact_path(store.a5_dir() / "case_sets.json")
        sets_path.write_text(json.dumps({"vector_method": spec["vector_method"], "sets": extended,
                                         "note": "not to be used as Phase 8 test notices"}, indent=1), encoding="utf-8")
        run.add_output(sets_path, "A5: each set's decisions in order of similarity to the seed")
        sizes = sorted({s["top_k"] for s in cfg5["settings"].values()})
        table = CS.describe(extended, sizes, corpus, labels)
        table_path = new_artefact_path(store.results_dir() / "a5_case_sets.csv")
        table.to_csv(table_path, index=False)
        run.add_output(table_path, "A5: the contexts, one row per set and size")

        # 2. Questions of each setting.
        ranked = {e["set_id"]: e["ranked_hashes"] for e in extended}
        by_setting, dropped = {}, {}
        for name, setting in cfg5["settings"].items():
            if setting.get("same_questions_as"):
                continue
            by_setting[name], dropped[name] = CS.setting_questions(frozen, ranked, setting["top_k"], corpus,
                                                                   cfg5["question_types"])

        if args.dry_run:
            print(table.to_string(index=False))
            for name, items in dropped.items():
                print(name, {t: sum(q["type"] == t for q in by_setting[name]) for t in cfg5["question_types"]},
                      [f"{d['question_id']}: {d['reason']}" for d in items])
            print("count golds:", {n: [(q["question_id"], q["gold"]) for q in qs if q["subtype"] == "count"] for n, qs in by_setting.items()})
            raise SystemExit(0)

        # 3. Key points: Phase 6's for its members, new ones for the added decisions.
        phase6_points = {r["acordao_hash"]: r["key_points"]
                         for r in store.read_jsonl(latest_artefact_path(store.root_dir() / "key_points.jsonl"))}
        log: list = []
        run.data["calls"] = log
        asker = J.Asker(cfg["material"]["model"], "key_points", log=log)
        new_points = key_points(asker, corpus[corpus["acordao_hash"].isin(added)], cfg)
        points = {**phase6_points, **new_points}
        titles = {h: Q.clean_title(t) for h, t in zip(corpus["acordao_hash"], corpus["acordao_titulo"])}
        points_path = new_artefact_path(store.a5_dir() / "key_points.jsonl")
        store.write_jsonl(points_path, [{"acordao_hash": h, "title": titles[h], "key_points": p} for h, p in sorted(new_points.items())])
        run.add_output(points_path, "A5: key points of the decisions added for k = 20 and 30 (Mistral Medium 3)")

        # 4. Reference points where the context is not Phase 6's.
        reference_asker = J.Asker(cfg["material"]["model"], "reference", log=log, **cfg5["prompt_limits"]["reference"])
        new_references = []
        for name, questions in by_setting.items():
            for q in questions:
                if q["type"] == "open" and "reference_points" not in q:
                    q["reference_points"] = reference_points(reference_asker, q, points, titles, cfg)
                    new_references.append((name, q))
        for name, setting in cfg5["settings"].items():
            if setting.get("same_questions_as"):
                by_setting[name] = by_setting[setting["same_questions_as"]]
                dropped[name] = dropped[setting["same_questions_as"]]

        # 5. The author's check of a sample of the new references.
        rng = np.random.default_rng(cfg5["references"]["seed"])
        n_check = int(np.ceil(cfg5["references"]["spot_check_share"] * len(new_references)))
        sample = sorted(rng.choice(len(new_references), size=n_check, replace=False).tolist()) if new_references else []
        by_hash = corpus.set_index("acordao_hash")
        check_rows, context_rows = [], []
        for i in sample:
            name, q = new_references[i]
            check_rows.append({"setting": name, "question_id": q["question_id"], "k": q["set_size"], "question": q["question"],
                               "reference_points": "\n".join(f"{j}. {p}" for j, p in enumerate(q["reference_points"], 1)),
                               "ok": None, "comment": None})
            for h in q["case_hashes"]:
                context_rows.append({"setting": name, "question_id": q["question_id"], "title": titles[h],
                                     "outcome": by_hash.at[h, "acordao_resultado"], "decision_url": by_hash.at[h, "acordao_url"],
                                     "added_for_phase7": h in set(added), "acordao_hash": h,
                                     "key_points": "\n".join(f"- {p}" for p in points[h])})
        sheet_path = new_artefact_path(store.results_dir() / "a5_references_spot_check.xlsx")
        with pd.ExcelWriter(sheet_path, engine="openpyxl") as writer:
            pd.DataFrame(check_rows).to_excel(writer, sheet_name="check", index=False)
            pd.DataFrame(context_rows).to_excel(writer, sheet_name="contexts", index=False)
        run.add_output(sheet_path, "A5: sample of the new reference points for the author's check")

        # 6. The question file of each setting and their hashes.
        record = {"settings": {}, "phase6_questions_sha256": sha256_file(store.frozen_questions_path()),
                  "references_checked_by_hugo": False, "spot_check": [f"{r['setting']}/{r['question_id']}" for r in check_rows]}
        for name, questions in by_setting.items():
            store.use_setting(name)
            path = new_artefact_path(store.root_dir() / "questions.jsonl")
            store.write_jsonl(path, [{**q, "setting": name} for q in questions])
            run.add_output(path, f"A5: questions of setting {name}")
            record["settings"][name] = {
                "file": path.name, "sha256": sha256_file(path), "k": cfg5["settings"][name]["top_k"],
                "context_mode": cfg5["settings"][name]["context_mode"],
                "asked": {t: sum(q["type"] == t for q in questions) for t in cfg5["question_types"]},
                "dropped": dropped[name]}
        store.use_setting(None)
        record_path = new_artefact_path(store.a5_freeze_record_path())
        record_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        run.add_output(record_path, "A5: hash of each setting's question file")

        phase6_dir = store.root_dir()
        store.use_setting("k10")
        k10 = reuse_phase6(cfg5["phase6"], phase6_dir, by_setting["k10"], run)
        store.use_setting(None)

        summary = {"added_decisions": len(added), "new_key_points": len(new_points), "new_references": len(new_references),
                   "spot_check": record["spot_check"], "k10_reused_rows": k10,
                   "settings": {n: {"asked": r["asked"], "dropped": len(r["dropped"])} for n, r in record["settings"].items()},
                   "material_calls": J.calls_summary(log)}
        run.data["summary"] = summary
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        print(table.to_string(index=False))
        for name, items in dropped.items():
            for d in items:
                print(f"{name}: {d['question_id']} dropped: {d['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
