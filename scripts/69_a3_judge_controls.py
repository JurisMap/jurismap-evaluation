"""Phase 6, A3: planted controls for the claim judges (added 2026-10-11).

Claims whose truth is known are built from the decisions of the case sets (jm_eval.chatbot.controls)
and labelled by every verifier with the prompt, the evidence retrieval and the batches of
scripts/64. A true control (a field as it is, a copied sentence) must come back "supported"; a
false one (the value of another decision, a changed figure) must not. This measures, per verifier,
the share of true claims it rejects and the share of false claims it accepts. No chatbot answer is
involved, so the controls do not depend on the question set.

`--dev` uses the development set. Outputs (never overwritten):
  data/v2/a3/controls.parquet           every control with its evidence and labels
  results/a3_judge_controls.csv         per verifier and kind of control (evaluated sets only)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from jm_eval.chatbot import controls as K
from jm_eval.chatbot import judging as J
from jm_eval.chatbot import store
from jm_eval.chatbot.questions import clean_title
from jm_eval.io import load_config, new_artefact_path
from jm_eval.manifest import RunManifest
from jm_eval.stats import wilson_interval

SCRIPT = "69_a3_judge_controls"


def _faithfulness():
    spec = importlib.util.spec_from_file_location("a3_faithfulness", Path(__file__).with_name("64_a3_faithfulness.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(sets: list[dict], corpus: pd.DataFrame, per_set: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    by_hash = corpus.set_index("acordao_hash", drop=False)
    rows, questions = [], {}
    for index, case_set in enumerate(sets):
        rng = np.random.default_rng([seed, index])
        members = by_hash.loc[case_set["case_hashes"]]
        made = K.field_controls(members, rng, per_set["field"]) + K.sentence_controls(
            members, rng, per_set["sentence_true"], per_set["sentence_false"])
        group = f"controls-{case_set['set_id']}"
        questions[group] = {"case_hashes": case_set["case_hashes"]}
        for i in rng.permutation(len(made)):                      # true and false controls mixed in a batch
            c = made[int(i)]
            rows.append({"answer_id": group, "question_id": group, "set_id": case_set["set_id"], "run": 0,
                         "claim_id": f"{group}/k{len(rows) + 1}", "processes": [], **c})
    return pd.DataFrame(rows), questions


def table(controls: pd.DataFrame, raters: list[str]) -> pd.DataFrame:
    rows = []
    for rater in raters:
        for kind, sub in [("all", controls), ("field", controls[controls["kind"].str.startswith("field")]),
                          ("sentence", controls[controls["kind"] == "sentence"])]:
            s = K.score(sub, rater)
            accepted = wilson_interval(s["true_accepted"], s["true_controls"])
            rejected = wilson_interval(s["false_rejected"], s["false_controls"])
            rows.append({"rater": rater, "controls": kind, **s,
                         "true_accepted_share": round(s["true_accepted"] / s["true_controls"], 4) if s["true_controls"] else None,
                         "true_accepted_ci": f"{accepted[0]:.3f} to {accepted[1]:.3f}" if s["true_controls"] else None,
                         "false_rejected_share": round(s["false_rejected"] / s["false_controls"], 4) if s["false_controls"] else None,
                         "false_rejected_ci": f"{rejected[0]:.3f} to {rejected[1]:.3f}" if s["false_controls"] else None})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--factor", type=int, default=1, help="multiply the controls per set (the development set is one set)")
    ap.add_argument("--judges", nargs="+", help="only these verifiers (default: every verifier of the config)")
    args = ap.parse_args()
    cfg = load_config("a3_chatbot")
    spec, per_set = cfg["faithfulness"], {k: v * args.factor for k, v in cfg["controls"]["per_set"].items()}
    judges, work = load_config("models")["judges"], store.work_dir(args.dev)
    names = args.judges or spec["verifiers"] + spec["separate"]
    F = _faithfulness()
    with RunManifest(SCRIPT, work, models={"nli": spec["nli_model"], **{n: judges[n]["model"] for n in names}},
                     params={"dev": args.dev, "controls": cfg["controls"], "per_set": per_set, "prompts_version": J.PROMPTS_VERSION},
                     seeds={"controls": cfg["controls"]["seed"]}) as run:
        calls: list = []
        run.data["calls"] = calls
        corpus = store.load_corpus()
        controls, questions = build(store.load_case_sets(args.dev), corpus, per_set, cfg["controls"]["seed"])
        controls, _ = F.retrieve(controls, questions, corpus, spec)
        titles = {h: clean_title(t) for h, t in zip(corpus["acordao_hash"], corpus["acordao_titulo"])}
        held = [K.evidence_holds_source(c, list(c["evidence"]), titles[c["acordao_hash"]]) for c in controls.to_dict("records")]
        built, controls = len(controls), controls[held].reset_index(drop=True)
        controls = F.nli_labels(controls, spec)
        claim_batches = F.batches(controls, spec["batch_claims"])
        askers = {name: J.Asker(name, "controls", dev=args.dev, log=calls) for name in names}
        with ThreadPoolExecutor(max_workers=len(askers)) as pool:
            futures = {name: pool.submit(F.verify, asker, claim_batches) for name, asker in askers.items()}
        for name, future in futures.items():
            controls[name] = controls["claim_id"].map(future.result())

        path = new_artefact_path(work / "controls.parquet")
        controls.to_parquet(path, index=False)
        run.add_output(path, "A3: planted controls with evidence and labels" + (" (development set)" if args.dev else ""))
        result = table(controls, ["nli", *names])
        out = store.results_dir() if not args.dev else work / "report"
        out.mkdir(parents=True, exist_ok=True)
        table_path = new_artefact_path(out / "a3_judge_controls.csv")
        result.to_csv(table_path, index=False)
        run.add_output(table_path, "A3: the verifiers on planted controls")
        run.data["summary"] = {"built": built, "kept": len(controls), "dropped_evidence_without_the_source": built - len(controls),
                               "by_kind": controls.groupby(["kind", "truth"]).size().rename("n").reset_index().to_dict("records"),
                               "batches": len(claim_batches),
                               "calls": {name: J.calls_summary(a.calls) for name, a in askers.items()}}
        print(json.dumps(run.data["summary"], indent=1, ensure_ascii=False))
        print(result.drop(columns=["true_accepted_ci", "false_rejected_ci"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
