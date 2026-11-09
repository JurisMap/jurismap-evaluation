"""Phase 8, A4: the judge prompts tried on development material only (ground rule 8).

The smoke-test drafts of the fictitious notice (config/a4_drafting.yaml `judging.development`), one
per mode, are judged by every judge with the rubric prompt, and by GPT-5.4 and Mistral with the
argument prompt against an argument list written for the fictitious notice. The precedents shown to
the rubric are the corpus decisions each draft cites (the smoke run did not record its case set). It
checks that every answer parses and that every prompt stays under the limit, and prints the answers.
Nothing here is reported.

Output: data/v2/a4/judging_dev.jsonl (git-ignored), one row per call.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile

import pandas as pd

from jm_eval.chatbot import judging as CJ
from jm_eval.drafting import judging as DJ
from jm_eval.drafting import metrics as M
from jm_eval.drafting import notices as N
from jm_eval.io import load_config, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C

SCRIPT = "86_a4_judging_dev"


def docx_text(path) -> str:
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    return "\n".join(t for t in (re.sub(r"<[^>]+>", "", p) for p in xml.split("</w:p>")) if t.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = N.settings()
    spec, dev = cfg["judging"], cfg["judging"]["development"]
    a4 = resolve(load_config("paths")["data"]["v2"]) / "a4"
    notice = docx_text(resolve(load_config("paths")["repos"]["frontend_v1"]) / dev["notice"])
    corpus = pd.read_parquet(resolve(cfg["corpus"]))
    index = M.CorpusIndex(corpus, C.load_labels())
    by_hash = corpus.set_index("acordao_hash")
    drafts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((resolve(dev["smoke_run"]) / "drafts").glob("*.md"))}
    log: list[dict] = []
    with RunManifest(SCRIPT, a4, params={"judging": spec, "prompts_version": DJ.PROMPTS_VERSION}) as run:
        run.data["calls"] = log
        askers = {j: CJ.Asker(j, f"a4_{j}", log=log, max_tokens=spec["max_prompt_tokens"], rpm=spec["rpm"][j])
                  for j in spec["judges"] + spec["separate"]}
        rows = []
        for name, draft in drafts.items():
            cited = sorted({h for c in M.citations(draft) for h in index.resolve(c["kind"], c["key"])})
            precedents = [by_hash.loc[h].to_dict() | {"acordao_hash": h} for h in cited][:10]
            for judge, asker in askers.items():
                prompt = DJ.rubric_prompt(notice, precedents, draft)
                got = asker.ask(prompt, DJ.validate_rubric, item=name, system=DJ.SYSTEM,
                                extra={"a4_judging": DJ.PROMPTS_VERSION, "task": "rubric", "dev": True})
                rows.append({"draft": name, "judge": judge, "task": "rubric", "tokens": CJ.estimated_tokens(prompt, DJ.SYSTEM), "answer": got})
                if judge in spec["judges"]:
                    prompt = DJ.arguments_prompt(dev["arguments"], draft)
                    got = asker.ask(prompt, lambda p: DJ.validate_arguments(p, len(dev["arguments"])), item=name,
                                    system=DJ.SYSTEM, extra={"a4_judging": DJ.PROMPTS_VERSION, "task": "arguments", "dev": True})
                    rows.append({"draft": name, "judge": judge, "task": "arguments",
                                 "tokens": CJ.estimated_tokens(prompt, DJ.SYSTEM), "answer": got})
                print(name, judge, json.dumps(rows[-1]["answer"], ensure_ascii=False)[:300], flush=True)
        out = a4 / "judging_dev.jsonl"
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        run.data["summary"] = {"calls": CJ.calls_summary(log), "unusable": sum(r["answer"] is None for r in rows),
                               "max_estimated_tokens": max(r["tokens"] for r in rows)}
        print(json.dumps(run.data["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
