"""Phase 8, A4 (task 1): The author's hand edits to the accepted notices, then the freeze of the test notices.

Every notice has passed the author's review (results/a4_notices_review_4.xlsx, then the re-check sheet
results/a4_notices_recheck_4.xlsx, all rows accepted). This script applies the exact text edits of
config/a4_drafting.yaml `hand_edits` (each `old` must occur exactly once in its notice), runs the
leakage check again on every notice, and freezes the set: later steps load the frozen file and check
its hash (`load_frozen`). No model is called.

Outputs (never overwritten):
  data/v2/a4/notices.jsonl            the frozen notices (hand edits recorded in each row)
  results/a4_notices.csv              one row per notice, without the texts
  results/a4_notices_freeze.json      the frozen file's path and sha256, and the reviews it passed
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from jm_eval.chatbot import nli as NLI
from jm_eval.drafting import notices as N
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve, sha256_file
from jm_eval.manifest import RunManifest

SCRIPT = "82_a4_freeze_notices"
REVIEWS = ["results/a4_notices_review_4.xlsx", "results/a4_notices_recheck_4.xlsx", "results/a4_notices_review_unreviewed.xlsx"]


def apply_edits(notices: list[dict], edits: list[dict]) -> list[dict]:
    """Text edits (`old` -> `new`, `old` occurring exactly once; skipped when already applied) and
    replacements of a notice's argument list (`arguments`) and accepted arguments (`accepted`)."""
    by_id = {r["notice_id"]: r for r in notices}
    for e in edits:
        r = by_id[e["notice"]]
        if "old" in e:
            if r["notice"].count(e["old"]) == 0 and e["new"] in r["notice"]:
                continue                                       # applied in an earlier freeze, carried over
            if r["notice"].count(e["old"]) != 1:
                raise ValueError(f"{e['notice']}: the text to replace occurs {r['notice'].count(e['old'])} times, not once")
            r = {**r, "notice": r["notice"].replace(e["old"], e["new"])}
        if "arguments" in e:
            r = {**r, "arguments": list(e["arguments"])}
        if "accepted" in e:
            if not all(1 <= i <= len(r["arguments"]) for i in e["accepted"]):
                raise ValueError(f"{e['notice']}: accepted {e['accepted']} outside the {len(r['arguments'])} arguments")
            r = {**r, "accepted": list(e["accepted"])}
        by_id[e["notice"]] = {**r, "hand_edits": [*[h for h in r.get("hand_edits", []) if h != e], e]}
    return [by_id[r["notice_id"]] for r in notices]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = N.settings()
    a4 = resolve(load_config("paths")["data"]["v2"]) / "a4"
    source = latest_artefact_path(a4 / "notices.jsonl")
    with RunManifest(SCRIPT, a4, models={"nli": cfg["leakage"]["nli_model"]}, params={"hand_edits": cfg["hand_edits"]}) as run:
        run.add_input(source)
        for review in REVIEWS:
            run.add_input(resolve(review))
        notices = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        notices = apply_edits(notices, cfg["hand_edits"])
        corpus = pd.read_parquet(resolve(cfg["corpus"])).set_index("acordao_hash")
        nli = NLI.load(cfg["leakage"]["nli_model"])
        out = []
        for r in notices:
            check = N.leakage(r["notice"], r["arguments"], corpus.loc[r["acordao_hash"]].to_dict(), cfg, nli)
            out.append({**r, "leakage": check, "still_flagged": check["flagged"], "accepted_by_hugo": True})
            if r.get("hand_edits"):
                print(r["notice_id"], "hand edited; flagged:", check["flagged"], "outcome amounts:", check["outcome_amounts"] or "-")
        path = new_artefact_path(a4 / "notices.jsonl")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
        run.add_output(path, "A4: the frozen test notices")
        table = pd.DataFrame([{"notice_id": r["notice_id"], "acordao_hash": r["acordao_hash"], "title": r["title"],
                               "base_key": r["base_key"], "outcome": r["outcome"], "replaces": r.get("replaces"),
                               "n_arguments": len(r["arguments"]),
                               "accepted": json.dumps(r["accepted"]) if r["accepted"] is not None else "",
                               "rewritten_after_review": bool(r.get("review_history")), "hand_edited": bool(r.get("hand_edits")),
                               "outcome_amounts": ", ".join(r["leakage"]["outcome_amounts"]),
                               "notice_chars": len(r["notice"])} for r in out])
        table_path = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_notices.csv")
        table.to_csv(table_path, index=False)
        run.add_output(table_path, "A4: the frozen test notices, one row per notice")
        record = {"file": str(path.relative_to(resolve("."))), "sha256": sha256_file(path), "notices": len(out),
                  "reviews": REVIEWS, "hand_edits": [e["notice"] for e in cfg["hand_edits"]], "run_id": run.run_id}
        record_path = new_artefact_path(resolve(cfg["freeze"]["record"]))
        record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        run.add_output(record_path, "A4: freeze record of the test notices")
        run.data["summary"] = record
        print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
