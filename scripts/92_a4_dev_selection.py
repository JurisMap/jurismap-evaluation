"""Phase 8 improvement round: score the fact-control arms on the development notices and apply the rule.

For each development draft (data/v2/a4_dev/drafts.jsonl): GPT-5.4's rubric (with its added facts),
whose added facts GPT-5.4 classifies as in scripts/89; the argument check by GPT-5.4 and Mistral and
the NLI vote (an argument is raised when two of the three say so); the [VERIFICAR: ...] markers; the
past-offences claim asserted as a fact (outside a [VERIFICAR] marker). Then the selection rule of
config/a4_drafting.yaml `improvement.selection`, fixed before this script ran, picks the arm.

Outputs (never overwritten):
  data/v2/a4_dev/judgments.jsonl        the judges' answers
  results/a4_dev_arms.csv               per arm: estimated invented facts, recall, rubric, markers, cost,
                                        eligibility and the selection
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from jm_eval.chatbot import judging as CJ
from jm_eval.chatbot import nli as NLI
from jm_eval.drafting import judging as DJ
from jm_eval.drafting import notices as N
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "92_a4_dev_selection"
VERIFY = re.compile(r"\[VERIFICAR:[^\]]*\]")


def load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, resolve(f"scripts/{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude", nargs="*", default=[], help="development notices to leave out (e.g. one the author rejected)")
    args = ap.parse_args()
    cfg = N.settings()
    imp, spec = cfg["improvement"], cfg["judging"]
    sel = imp["selection"]
    dev = resolve(load_config("paths")["data"]["v2"]) / "a4_dev"
    notices = {r["notice_id"]: r for r in map(json.loads, latest_artefact_path(dev / "notices.jsonl").read_text(encoding="utf-8").splitlines())}
    precedents = json.loads((dev / "precedents.json").read_text(encoding="utf-8"))
    drafts = [d for d in map(json.loads, (dev / "drafts.jsonl").read_text(encoding="utf-8").splitlines())
              if d["ok"] and d["notice_id"] not in args.exclude]
    corpus = pd.read_parquet(resolve(cfg["corpus"]), columns=["acordao_hash", "acordao_titulo", "acordao_ementa"]).set_index("acordao_hash")
    past = load_script("90_a4_invented_facts").PAST_OFFENCES
    log: list[dict] = []
    askers = {j: CJ.Asker(j, f"a4dev_{j}", log=log, max_tokens=spec["max_prompt_tokens"], rpm=spec["rpm"][j]) for j in spec["judges"]}

    def text_of(d):
        return (dev / "drafts" / f"{d['notice_id']}_{d['arm']}_{d['mode']}_r{d['run']}.md").read_text(encoding="utf-8")

    def judge_a(d):
        n, draft = notices[d["notice_id"]], text_of(d)
        extra = {"a4_judging": DJ.PROMPTS_VERSION, "dev_arm": d["arm"], "draft": f"{d['notice_id']}_{d['arm']}"}
        supplied = [corpus.loc[h].to_dict() for h in precedents[d["notice_id"]]["precedents"]]
        rubric = askers["judge_a"].ask(DJ.rubric_prompt(n["notice"], supplied, draft), DJ.validate_rubric, item=extra["draft"],
                                       system=DJ.SYSTEM, extra={**extra, "task": "rubric"})
        items = DJ.added_facts(rubric)
        kinds = askers["judge_a"].ask(DJ.fact_kinds_prompt(n["notice"], items), lambda p, k=len(items): DJ.validate_fact_kinds(p, k),
                                      item=extra["draft"], system=DJ.SYSTEM, extra={**extra, "task": "fact_kinds"}) if items else {"kinds": []}
        args = askers["judge_a"].ask(DJ.arguments_prompt(n["arguments"], draft), lambda p, k=len(n["arguments"]): DJ.validate_arguments(p, k),
                                     item=extra["draft"], system=DJ.SYSTEM, extra={**extra, "task": "arguments"})
        return {"rubric": rubric, "items": items, "kinds": kinds["kinds"] if kinds else None, "arguments": args}

    def judge_b(d):
        n = notices[d["notice_id"]]
        extra = {"a4_judging": DJ.PROMPTS_VERSION, "dev_arm": d["arm"], "draft": f"{d['notice_id']}_{d['arm']}", "task": "arguments"}
        return askers["judge_b"].ask(DJ.arguments_prompt(n["arguments"], text_of(d)), lambda p, k=len(n["arguments"]): DJ.validate_arguments(p, k),
                                     item=extra["draft"], system=DJ.SYSTEM, extra=extra)

    with RunManifest(SCRIPT, dev, params={"improvement": imp, "prompts_version": DJ.PROMPTS_VERSION, "exclude": args.exclude}) as run:
        run.data["calls"] = log
        with ThreadPoolExecutor(max_workers=8) as pool:
            fa = pool.map(judge_a, drafts)
            fb = list(ThreadPoolExecutor(max_workers=1).map(judge_b, drafts))
            answers_a = list(fa)
        nli = NLI.load(cfg["leakage"]["nli_model"])
        rows, saved = [], []
        for d, a, b in zip(drafts, answers_a, fb):
            n, draft = notices[d["notice_id"]], text_of(d)
            plain = VERIFY.sub("", draft)                      # a premise under a marker is not asserted
            votes = [v for v in (N.leakage(plain, n["arguments"], {"acordao_acordao": "", "acordao_relatorio": ""}, cfg, nli)["per_argument"])]
            nli_raised = [v["entailment"] > spec["nli_threshold"] for v in votes]
            la, lb = a["arguments"]["labels"], b["labels"]
            raised = [(x == "raised") + (y == "raised") + z >= 2 for x, y, z in zip(la, lb, nli_raised)]
            kinds = a["kinds"] or []
            rows.append({"notice_id": d["notice_id"], "arm": d["arm"],
                         "invented_estimate": sum(sel["audited_share"][k] for k in kinds),
                         "invented_classified": kinds.count("invented"),
                         "recall": sum(raised) / len(raised),
                         **{f"gpt_{c}": a["rubric"]["scores"][c] for c in DJ.CRITERIA},
                         "verify_markers": len(VERIFY.findall(draft)),
                         "past_offences_asserted": any(p.search(plain) for p in past),
                         "cost_usd": d["cost_usd"], "latency_s": (d.get("latency_ms") or 0) / 1000})
            saved.append({"notice_id": d["notice_id"], "arm": d["arm"], "judge_a": a, "judge_b": b, "nli": votes})
        path = new_artefact_path(dev / "judgments.jsonl")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in saved) + "\n", encoding="utf-8")
        run.add_output(path, "A4 improvement round: judges' answers on the development drafts")

        df = pd.DataFrame(rows)
        arms = df.groupby("arm").mean(numeric_only=True).reindex(list(imp["arms"]))
        base = arms.loc["baseline"]
        arms["controls"] = [len(imp["arms"][a]) for a in arms.index]
        arms["eligible"] = ((base["recall"] - arms["recall"] <= sel["max_recall_drop"])
                            & (base["gpt_use_of_precedents"] - arms["gpt_use_of_precedents"] <= sel["max_rubric_drop"]["use_of_precedents"])
                            & (base["gpt_legal_accuracy"] - arms["gpt_legal_accuracy"] <= sel["max_rubric_drop"]["legal_accuracy"])
                            & (arms["verify_markers"] <= sel["max_verify_markers"]))
        eligible = arms[arms["eligible"]]
        best = eligible["invented_estimate"].min()
        close = eligible[eligible["invented_estimate"] <= best + sel["tie_margin"]].sort_values(["controls", "cost_usd"])
        arms["selected"] = arms.index == close.index[0]
        out = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_dev_arms.csv")
        arms.round(3).to_csv(out)
        run.add_output(out, "A4 improvement round: arms on the development notices and the selection")
        run.data["summary"] = {"selected": close.index[0], "calls": CJ.calls_summary(log)}
        print(arms.round(2).to_string())
        print(json.dumps(run.data["summary"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
