"""Phase 8, A4 (task 1): the rewrite pass after the author's review of the test notices.

Runs once every row of the review sheet (newest `results/a4_notices_review*.xlsx`) has a mark in
`hugo_notice_ok`; it stops and lists the rows without one otherwise.

1. Each notice marked as rejected is written again by Mistral Medium 3 from the author's comment, with the
   rejected version shown to it (prompt version 3 plus the review block), then checked for leakage
   as in scripts/80, with one leakage rewrite round.
2. Each argument list marked as rejected is extracted again from the author's comment, and for granted
   appeals the accepted arguments are read again from the new list.
3. The leakage check, with the stricter amount rule (version 2: any amount the dispositive sets
   rather than keeps), runs on every notice.
4. A re-check sheet holds the rewritten notices and any notice the check newly flags.

Outputs (never overwritten):
  data/v2/a4/notices.jsonl             all 15 notices, rewritten ones replaced (their history kept)
  results/a4_notices.csv               one row per notice, without the texts
  results/a4_notices_recheck.xlsx      what goes back to the author
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from jm_eval.chatbot import judging as J
from jm_eval.chatbot import nli as NLI
from jm_eval.drafting import notices as N
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "81_a4_notice_rewrite"


def a4_dir():
    return resolve(load_config("paths")["data"]["v2"]) / "a4"


def mark(value, spec: dict) -> bool | None:
    """True (accepted), False (rejected) or None (no mark) from a cell of the review sheet."""
    text = "" if value is None or (isinstance(value, float) and value != value) else str(value).strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    if text in spec["accept"]:
        return True
    if text in spec["reject"] or text in spec.get("partly", []):
        return False
    if text:
        raise ValueError(f"unreadable mark {value!r}: use one of {spec['accept']} or {spec['reject']}")
    return None


def clean_comment(text: str, spec: dict) -> str:
    """The comment without the reviewer's tag (config `review.comment_tags`)."""
    text = (text or "").strip()
    for tag in spec.get("comment_tags", []):
        if text.startswith(tag):
            text = text[len(tag):].strip()
    return text


def read_review(cfg: dict, sheet_path: str | None = None) -> tuple[pd.DataFrame, object]:
    path = resolve(sheet_path) if sheet_path else latest_artefact_path(resolve(cfg["review"]["sheet"]))
    sheet = pd.read_excel(path, dtype=str).fillna("")
    sheet["hugo_comment"] = sheet["hugo_comment"].map(lambda t: clean_comment(t, cfg["review"]))
    sheet["notice_ok"] = sheet["hugo_notice_ok"].map(lambda v: mark(v, cfg["review"]))
    sheet["arguments_ok"] = sheet["hugo_arguments_ok"].map(lambda v: mark(v, cfg["review"]))
    return sheet, path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="read the marks, say what would be rewritten and stop (no model call)")
    ap.add_argument("--sheet", help="the marked review sheet (default: the newest results/a4_notices_review*.xlsx)")
    ap.add_argument("--arguments-with-vote", action="store_true",
                    help="only extract again the argument lists of config `material.arguments_with_vote.notices`, "
                         "reading the vote too, from the marked sheet's comment and the first rejected list")
    args = ap.parse_args()
    cfg = N.settings()
    m = cfg["material"]
    sheet, sheet_path = read_review(cfg, args.sheet)
    missing = sheet.loc[sheet["notice_ok"].isna(), "notice_id"].tolist()
    if missing:
        print(f"{sheet_path.name}: no hugo_notice_ok mark for {', '.join(missing)}. The rewrite runs once every row is marked.")
        return 1
    notices_path = latest_artefact_path(a4_dir() / "notices.jsonl")
    notices = [json.loads(line) for line in notices_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reviewed = {r["notice_id"]: r for r in sheet.to_dict("records")}
    rewrite_notice = [n for n, r in reviewed.items() if r["notice_ok"] is False]
    rewrite_arguments = [n for n, r in reviewed.items() if r["arguments_ok"] is False]
    with_vote = m["arguments_with_vote"]["notices"] if args.arguments_with_vote else []
    if with_vote:                                    # a later pass on some argument lists only
        rewrite_notice, rewrite_arguments = [], list(with_vote)
    for n in rewrite_notice + rewrite_arguments:
        if not reviewed[n]["hugo_comment"].strip():
            raise ValueError(f"{n} is rejected without a comment in hugo_comment: the rewrite needs one")
    print(f"{sheet_path.name}: {len(reviewed)} reviewed; notices to rewrite: {rewrite_notice or 'none'}; "
          f"argument lists to extract again: {rewrite_arguments or 'none'}")
    if args.dry_run:
        return 0

    models = {"material": load_config("models")["judges"][m["model"]]["model"], "nli": cfg["leakage"]["nli_model"]}
    with RunManifest(SCRIPT, a4_dir(), models=models,
                     params={"a4": cfg, "prompts_version": N.PROMPTS_VERSION, "notice_prompt_version": N.NOTICE_PROMPT_VERSION,
                             "rewrite_notice": rewrite_notice, "rewrite_arguments": rewrite_arguments,
                             "arguments_with_vote": with_vote}) as run:
        run.add_input(sheet_path)
        run.add_input(notices_path)
        corpus = pd.read_parquet(resolve(cfg["corpus"])).set_index("acordao_hash")
        log: list[dict] = []
        run.data["calls"] = log
        ask = {step: J.Asker(m["model"], step, log=log, rpm=m["rpm"], max_tokens=m.get("rewrite_max_prompt_tokens"))
               for step in ("arguments", "accepted", "notice")}
        nli = NLI.load(cfg["leakage"]["nli_model"])
        out, recheck = [], []
        for r in notices:
            nid, row = r["notice_id"], corpus.loc[r["acordao_hash"]].to_dict()
            extra = {"a4": N.PROMPTS_VERSION, "acordao_hash": r["acordao_hash"], "review_of": sheet_path.name,
                     **({"with_vote": True} if nid in with_vote else {})}
            comment = reviewed.get(nid, {}).get("hugo_comment", "").strip()
            history = list(r.get("review_history", []))
            changed = False
            if nid in rewrite_arguments:
                # The comment refers to the list it was written on: the first rejected one.
                first = next((h["previous"] for h in history if h["what"] == "arguments"), r["arguments"])
                review = {"previous": first if nid in with_vote else r["arguments"], "comment": comment}
                got = ask["arguments"].ask(N.arguments_prompt(row, cfg, review, with_vote=nid in with_vote),
                                           lambda p: N.validate_arguments(p, m["arguments"]["min"], m["arguments"]["max"]),
                                           item=nid, extra=extra, system=N.SYSTEM)
                if got is None:
                    raise RuntimeError(f"{nid}: no usable argument list")
                history.append({"what": "arguments", "comment": comment, "previous": r["arguments"],
                                "with_vote": nid in with_vote})
                r = {**r, "arguments": got["arguments"]}
                if r["outcome"] in ("Provido", "Parcialmente provido"):
                    acc = ask["accepted"].ask(N.accepted_prompt(row, r["arguments"], cfg),
                                              lambda p: N.validate_accepted(p, len(r["arguments"])),
                                              item=nid, extra=extra, system=N.SYSTEM)
                    if acc is None:
                        raise RuntimeError(f"{nid}: no usable accepted arguments")
                    r = {**r, "accepted": acc["accepted"], "granted": acc["granted"]}
                changed = True
            if nid in rewrite_notice:
                review, avoid, versions = {"previous": r["notice"], "comment": comment}, [], []
                for round_ in range(cfg["leakage"]["rewrite_rounds"] + 1):
                    got = ask["notice"].ask(N.notice_prompt(row, cfg, avoid, review), N.validate_notice, item=nid,
                                            extra={**extra, "notice_prompt": N.NOTICE_PROMPT_VERSION, "round": round_},
                                            system=N.SYSTEM)
                    if got is None:
                        raise RuntimeError(f"{nid}: no usable notice (round {round_})")
                    check = N.leakage(got["notice"], r["arguments"], row, cfg, nli)
                    versions.append({"round": round_, "notice": got["notice"], "leakage": check})
                    if not check["flagged"]:
                        break
                    avoid = [r["arguments"][k - 1] for k in check["flagged_arguments"]]
                history.append({"what": "notice", "comment": comment, "previous": r["notice"], "versions": versions})
                r = {**r, "notice": versions[-1]["notice"]}
                changed = True
            check = N.leakage(r["notice"], r["arguments"], row, cfg, nli)       # rule version 2, on every notice
            newly = bool(check["outcome_amounts"]) and not r["leakage"].get("outcome_amounts")
            field = r.get("penalty_field_amount")
            r = {**r, "leakage": check, "still_flagged": check["flagged"], "review_history": history,
                 "penalty_field_amount_in_document": (field in N.amounts(r["notice"])) if field else None,
                 "hugo_notice_ok": reviewed[nid]["notice_ok"] if nid in reviewed else None,
                 "hugo_arguments_ok": reviewed[nid]["arguments_ok"] if nid in reviewed else None,
                 "review_marks": ({"notice": reviewed[nid]["hugo_notice_ok"], "arguments": reviewed[nid]["hugo_arguments_ok"],
                                   "comment": comment, "sheet": sheet_path.name} if nid in reviewed else None)}
            out.append(r)
            if changed or newly:
                recheck.append((r, "rewritten" if changed else "newly flagged (amount rule version 2)", comment))
            print(nid, "rewritten" if changed else "kept", "flagged:", check["flagged"],
                  "outcome amounts:", check["outcome_amounts"] or "-", flush=True)

        path = new_artefact_path(a4_dir() / "notices.jsonl")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
        run.add_output(path, "A4: test notices after the author's review")
        table = pd.DataFrame([{k: r.get(k) for k in ("notice_id", "acordao_hash", "title", "base_key", "outcome", "cell_size",
                                                     "replaces", "still_flagged", "penalty_field_amount",
                                                     "penalty_field_amount_in_document", "hugo_notice_ok", "hugo_arguments_ok")}
                              | {"n_arguments": len(r["arguments"]), "rewritten_after_review": bool(r["review_history"]),
                                 "accepted": json.dumps(r["accepted"]) if r["accepted"] is not None else "",
                                 "outcome_amounts": ", ".join(r["leakage"]["outcome_amounts"]),
                                 "forbidden_terms": ", ".join(r["leakage"]["forbidden_terms"]),
                                 "notice_chars": len(r["notice"])} for r in out])
        table_path = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_notices.csv")
        table.to_csv(table_path, index=False)
        run.add_output(table_path, "A4: the test notices after the author's review, one row per notice")
        sheet_out = pd.DataFrame([{
            "notice_id": r["notice_id"], "why": why, "your_comment": comment, "title": r["title"],
            "acordao_url": r["acordao_url"], "outcome": r["outcome"],
            "previous_notice": r["review_history"][-1]["previous"] if r["review_history"] and r["review_history"][-1]["what"] == "notice" else "",
            "notice": r["notice"],
            "appellant_arguments": "\n".join(f"{i}. {a}" for i, a in enumerate(r["arguments"], 1)),
            "accepted": ", ".join(map(str, r["accepted"])) if r["accepted"] else "", "granted": r["granted"] or "",
            "outcome_amounts": ", ".join(r["leakage"]["outcome_amounts"]),
            "hugo_notice_ok": "", "hugo_arguments_ok": "", "hugo_comment": ""} for r, why, comment in recheck])
        recheck_path = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_notices_recheck.xlsx")
        sheet_out.to_excel(recheck_path, index=False)
        run.add_output(recheck_path, "A4: rewritten and newly flagged notices for the author")
        run.data["summary"] = {"reviewed": len(reviewed), "rewritten_notices": rewrite_notice,
                               "rewritten_argument_lists": rewrite_arguments,
                               "recheck": [r["notice_id"] for r, _, _ in recheck], "material_calls": J.calls_summary(log)}
        print(json.dumps(run.data["summary"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
