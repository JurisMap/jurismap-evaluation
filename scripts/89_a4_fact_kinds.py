"""Phase 8, A4 (task 4): what the "added facts" of the rubric are (the author, 2026-10-18).

GPT-5.4's rubric lists, per draft, the passages that state something about the case that the case
document does not state. Those lists mix invented case facts with the defence's own assertions,
procedural statements and facts the notice does state. GPT-5.4 classifies each listed passage as
invented, assertion, procedural or in_notice (src/jm_eval/drafting/judging.py FACT_KINDS_PROMPT), one
call per draft, against the case document.

  --dev       the six smoke-test drafts of the fictitious notice (development material only)
  (default)   every evaluated draft judged by GPT-5.4; then the blind audit sheet for the author: 50 passages,
              the same number from each kind the classifier gave, with the case document and the
              passage's paragraph in the draft, without the classifier's label (kept in a key file)

Outputs (never overwritten):
  data/v2/a4/fact_kinds.jsonl          one row per draft: the passages and their kinds
  results/a4_fact_audit.xlsx           the passages for the author to classify
  results/a4_fact_audit_key.csv        the audit's key (not for the author before the author marks the sheet)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import zipfile

import numpy as np
import pandas as pd

from jm_eval.chatbot import judging as CJ
from jm_eval.drafting import judging as DJ
from jm_eval.drafting import notices as N
from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "89_a4_fact_kinds"
KIND_MEANINGS = {
    "invented": "A specific fact about the party or its case (an event, a date, an amount, a document, a name, a number, a procedural act, "
                "a circumstance of the facts) that the case document does not state and the passage presents as a fact.",
    "assertion": "The party's own characterisation of its conduct, intent or situation (good faith, no intent, diligence, cooperation, no harm, "
                 "no profit, regular history), or an argument or legal qualification, without a specific checkable fact.",
    "procedural": "What any appeal implies and the case document need not state: the party was notified of the decision, the appeal is filed "
                  "in time, it is represented by lawyers.",
    "in_notice": "The case document states it, or it follows directly from what it states; also a restatement of a stated fact in other words "
                 "(a favourable slant alone makes it an assertion)."}
HOW_TO_MARK = [
    "Each row is a passage of a draft appeal that GPT-5.4 listed as stating something about the case that the case document does not state.",
    "Read the passage in its paragraph and against the case document (the notice and SUSEP's first-instance decision), then write one kind "
    "in hugo_kind: invented, assertion, procedural or in_notice.",
    "When a passage mixes kinds, choose invented if it contains any specific invented fact. Mark the listed passage only, not its paragraph.",
    "A passage followed by [VERIFICAR: ...] presents its fact as one to confirm, not as established.",
    "The sheet does not show how the classifier labelled the passages; 12 or 13 were drawn from each of its four kinds."]
SEED = 20261018
AUDIT_N = 50


def docx_text(path) -> str:
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    return "\n".join(t for t in (re.sub(r"<[^>]+>", "", p) for p in xml.split("</w:p>")) if t.strip())


def paragraph_of(draft: str, item: str) -> str:
    """The draft paragraph that contains the passage (compared without accents, case or punctuation)."""
    needle = N.normalise(item.strip().strip('"“”'))[:80]
    for par in re.split(r"\n\s*\n", draft):
        if needle and needle in N.normalise(par):
            return par.strip()[:1500]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--no-audit", action="store_true", help="classify only; no new audit sheet (v2.6, whose audit the author marked)")
    args = ap.parse_args()
    cfg = N.settings()
    spec = cfg["judging"]
    a4 = N.work_dir() if not args.dev else resolve(load_config("paths")["data"]["v2"]) / "a4"
    results = resolve(load_config("paths")["results"])
    log: list[dict] = []
    asker = CJ.Asker("judge_a", "a4_fact_kinds", log=log, max_tokens=spec["max_prompt_tokens"], rpm=spec["rpm"]["judge_a"])

    if args.dev:
        dev = spec["development"]
        notice = docx_text(resolve(load_config("paths")["repos"]["frontend_v1"]) / dev["notice"])
        for r in map(json.loads, (a4 / "judging_dev.jsonl").read_text(encoding="utf-8").splitlines()):
            if r["judge"] != "judge_a" or r["task"] != "rubric":
                continue
            items = DJ.added_facts(r["answer"])
            got = asker.ask(DJ.fact_kinds_prompt(notice, items), lambda p, n=len(items): DJ.validate_fact_kinds(p, n),
                            item=r["draft"], system=DJ.SYSTEM, extra={"a4_fact_kinds": DJ.PROMPTS_VERSION, "dev": True})
            print(f"== {r['draft']}: {pd.Series(got['kinds']).value_counts().to_dict() if got else 'UNUSABLE'}")
            for t, k in zip(items, got["kinds"] if got else []):
                print(f"   [{k}] {t[:110]}")
        print(json.dumps(CJ.calls_summary(log)))
        return 0

    frozen = {r["notice_id"]: r for r in N.load_frozen(cfg)}
    judged = {}
    for path in sorted(glob.glob(str(a4 / "judgments*.jsonl")), key=lambda p: (len(p), p)):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["judge"] == "judge_a" and r["task"] == "rubric" and r["answer"] is not None:
                judged[(r["notice_id"], r["mode"], r["run"])] = r["answer"]
    # Only drafts written now: judgments of superseded drafts (v2.6's N10 on the old notice) are left out.
    written = {(d["notice_id"], d["mode"], d["run"]) for d in map(json.loads, (a4 / "drafts.jsonl").read_text(encoding="utf-8").splitlines())
               if d["ok"]}
    judged = {k: v for k, v in judged.items() if k in written}
    with RunManifest(SCRIPT, a4, params={"prompts_version": DJ.PROMPTS_VERSION, "audit_n": AUDIT_N}, seeds={"audit": SEED}) as run:
        run.data["calls"] = log
        def classify(key):
            items = DJ.added_facts(judged[key])
            if not items:
                return {"notice_id": key[0], "mode": key[1], "run": key[2], "items": [], "kinds": []}
            got = asker.ask(DJ.fact_kinds_prompt(frozen[key[0]]["notice"], items), lambda p, n=len(items): DJ.validate_fact_kinds(p, n),
                            item=f"{key[0]}_{key[1]}_r{key[2]}", system=DJ.SYSTEM, extra={"a4_fact_kinds": DJ.PROMPTS_VERSION})
            return {"notice_id": key[0], "mode": key[1], "run": key[2], "items": items, "kinds": got["kinds"] if got else None}

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as pool:            # calls take about a minute (reasoning)
            rows = list(pool.map(classify, sorted(judged)))
        out = new_artefact_path(a4 / "fact_kinds.jsonl")
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        run.add_output(out, "A4: kinds of GPT-5.4's added facts, per draft")

        if args.no_audit:
            run.data["summary"] = {"drafts": len(rows), "calls": CJ.calls_summary(log)}
            print(json.dumps(run.data["summary"], indent=1))
            return 0
        flat = [{"notice_id": r["notice_id"], "mode": r["mode"], "run": r["run"], "item": t, "kind": k}
                for r in rows if r["kinds"] for t, k in zip(r["items"], r["kinds"])]
        flat = pd.DataFrame(flat)
        rng = np.random.default_rng(SEED)
        per_kind = {k: AUDIT_N // len(DJ.FACT_KINDS) + (1 if i < AUDIT_N % len(DJ.FACT_KINDS) else 0)
                    for i, k in enumerate(DJ.FACT_KINDS)}
        picked = []
        for k, n in per_kind.items():
            pool = flat.index[flat["kind"] == k].to_numpy()
            picked += list(rng.choice(pool, min(n, len(pool)), replace=False))
        audit = flat.loc[picked].sample(frac=1, random_state=SEED).reset_index(drop=True)
        audit.insert(0, "audit_id", [f"F{i + 1:02d}" for i in range(len(audit))])
        sheet = pd.DataFrame({
            "audit_id": audit["audit_id"],
            "passage": audit["item"],
            "draft_paragraph": [paragraph_of((a4 / "drafts" / f"{n}_{m}_r{k}.md").read_text(encoding="utf-8"), t)
                                for n, m, k, t in zip(audit["notice_id"], audit["mode"], audit["run"], audit["item"])],
            "case_document": [frozen[n]["notice"] for n in audit["notice_id"]],
            "hugo_kind": "", "hugo_comment": ""})
        sheet_path = new_artefact_path(N.result_path("a4_fact_audit.xlsx"))
        with pd.ExcelWriter(sheet_path) as w:                # the passages, the kinds' meaning, how to mark
            sheet.to_excel(w, sheet_name="passages", index=False)
            pd.DataFrame({"hugo_kind": list(DJ.FACT_KINDS), "meaning": [KIND_MEANINGS[k] for k in DJ.FACT_KINDS]}).to_excel(
                w, sheet_name="kinds", index=False)
            pd.DataFrame({"how to mark": HOW_TO_MARK}).to_excel(w, sheet_name="how to mark", index=False)
        run.add_output(sheet_path, "A4: blind audit of the added facts' kinds, for the author")
        key_path = new_artefact_path(N.result_path("a4_fact_audit_key.csv"))
        audit.to_csv(key_path, index=False)
        run.add_output(key_path, "A4: key of the blind audit (classifier's kind)")
        run.data["summary"] = {"drafts": len(rows), "passages": len(flat), "kinds": flat["kind"].value_counts().to_dict(),
                               "audit": len(audit), "calls": CJ.calls_summary(log)}
        print(json.dumps(run.data["summary"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
