"""Phase 8, A4 (tasks 4 to 6): the judges on every evaluated draft, blind to the condition.

For each draft that was written (data/v2/a4/drafts.jsonl, ok), in random order: the rubric by every
judge of config/a4_drafting.yaml `judging` (GPT-5.4 and Mistral, and Gemini 3.8 Flash reported
separately), from the draft, the frozen case document and the headnotes of the supplied precedents
(none in the no_precedents mode); the argument check by GPT-5.4 and Mistral against the real
appellant's arguments; and the local NLI model's vote on each argument (a sentence of the draft
entails it above the threshold). Each judge works through the drafts in its own thread at its own
pace. Answers come from the cache when a call was made before, so the script resumes.

Output: data/v2/a4/judgments.jsonl (git-ignored, never overwritten): one row per draft, judge and task.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from jm_eval.chatbot import judging as CJ
from jm_eval.chatbot import nli as NLI
from jm_eval.drafting import judging as DJ
from jm_eval.drafting import notices as N
from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "87_a4_judging"
SEED = 20261018


def nli_votes(nli, draft: str, arguments: list[str], threshold: float) -> list[dict]:
    premises = N.sentences(draft) or [draft]
    out = []
    for argument in arguments:
        probs = NLI.probabilities(nli, premises, [N.argument_content(argument)] * len(premises))
        best = int(np.argmax(probs[:, 0]))
        out.append({"entailment": round(float(probs[best, 0]), 4), "raised": bool(probs[best, 0] > threshold)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="judge only the first n drafts of the random order (a trial)")
    ap.add_argument("--judges", nargs="*", help="only these judges (e.g. while one is unavailable); the rest later, from the cache")
    args = ap.parse_args()
    cfg = N.settings()
    spec = cfg["judging"]
    a4 = N.work_dir()                                      # JM_A4_VERSION selects v2.6 (default) or v2.7
    frozen = {r["notice_id"]: r for r in N.load_frozen(cfg)}
    precedents = json.loads((a4 / "precedents.json").read_text(encoding="utf-8"))
    corpus = pd.read_parquet(resolve(cfg["corpus"]), columns=["acordao_hash", "acordao_titulo", "acordao_ementa"]).set_index("acordao_hash")
    latest = {}
    for d in map(json.loads, (a4 / "drafts.jsonl").read_text(encoding="utf-8").splitlines()):
        latest[(d["notice_id"], d["mode"], d["run"])] = d
    items = sorted(k for k, d in latest.items() if d["ok"])
    order = [items[i] for i in np.random.default_rng(SEED).permutation(len(items))][: args.limit]
    out_path = new_artefact_path(a4 / "judgments.jsonl")
    log: list[dict] = []

    def draft_text(key):
        return (a4 / "drafts" / f"{key[0]}_{key[1]}_r{key[2]}.md").read_text(encoding="utf-8")

    judges = [j for j in spec["judges"] + spec["separate"] if not args.judges or j in args.judges]
    askers = {j: CJ.Asker(j, f"a4_{j}", log=log, max_tokens=spec["max_prompt_tokens"], rpm=spec["rpm"][j]) for j in judges}

    def work(judge: str, keys: list) -> list[dict]:
        asker = askers[judge]                            # one paced asker per judge, shared by its threads
        rows = []
        for key in keys:
            nid, mode, k = key
            draft, notice = draft_text(key), frozen[nid]
            supplied = [] if mode == "no_precedents" else [corpus.loc[h].to_dict() for h in precedents[nid]["precedents"]]
            extra = {"a4_judging": DJ.PROMPTS_VERSION, "draft": f"{nid}_{mode}_r{k}", "draft_sha256": latest[key]["draft_sha256"]}
            got = asker.ask(DJ.rubric_prompt(notice["notice"], supplied, draft), DJ.validate_rubric,
                            item=extra["draft"], system=DJ.SYSTEM, extra={**extra, "task": "rubric"})
            rows.append({"notice_id": nid, "mode": mode, "run": k, "judge": judge, "task": "rubric", "answer": got})
            if judge in spec["judges"]:
                got = asker.ask(DJ.arguments_prompt(notice["arguments"], draft),
                                lambda p, n=len(notice["arguments"]): DJ.validate_arguments(p, n),
                                item=extra["draft"], system=DJ.SYSTEM, extra={**extra, "task": "arguments"})
                rows.append({"notice_id": nid, "mode": mode, "run": k, "judge": judge, "task": "arguments", "answer": got})
            print(judge, extra["draft"], "ok" if got is not None else "UNUSABLE", flush=True)
        return rows

    with RunManifest(SCRIPT, a4, params={"judging": spec, "prompts_version": DJ.PROMPTS_VERSION, "drafts": len(order),
                                         "seed": SEED, "judges": judges}) as run:
        run.data["calls"] = log
        threads = spec.get("threads", {})
        jobs = [(j, order[t::threads.get(j, 1)]) for j in judges for t in range(threads.get(j, 1))]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(work, j, keys) for j, keys in jobs]
            nli = NLI.load(cfg["leakage"]["nli_model"])
            nli_rows = [{"notice_id": n, "mode": m, "run": k, "judge": "nli", "task": "arguments",
                         "answer": {"votes": nli_votes(nli, draft_text((n, m, k)), frozen[n]["arguments"], spec["nli_threshold"])}}
                        for n, m, k in order]
            rows = [r for f in futures for r in f.result()] + nli_rows
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        run.data["summary"] = {"drafts": len(order), "rows": len(rows), "unusable": sum(r["answer"] is None for r in rows),
                               "calls": CJ.calls_summary(log)}
        print(json.dumps(run.data["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
