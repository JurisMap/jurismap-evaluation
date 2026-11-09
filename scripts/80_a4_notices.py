"""Phase 8, A4 (task 1, and the material of task 5): the test notices and the appellants' arguments.

1. Selection: one decision drawn at random in each cell of the 5 most frequent legal-basis keys x
   3 outcomes on the merits (15), one per process. Excluded: every decision used in Phases 6 and 7
   (225), and decisions whose rapporteur was outvoted (config/a4_drafting.yaml).
2. Mistral Medium 3 lists the appellant's arguments from the report, reading the dispositive and the
   outcome too; for granted appeals (in full or in part) it says which of them the council accepted,
   from the dispositive, which prevails, and the vote.
3. Mistral reconstructs the case document from the report: the infraction notice (facts, dates and
   charged provisions) and SUSEP's first-instance decision as far as the penalty and how it was set.
4. Leakage check: no argument entailed by a sentence of the notice (local NLI) or contained in it
   (word 5-grams) above the thresholds, no outcome or appeal vocabulary, and no amount of the
   council's dispositive that the report does not state. A flagged notice is
   written once more with the flagged arguments named; one still flagged goes to the author with the flag.
5. Review sheet: 5 notices drawn at random, plus every notice still flagged, for the author.

Outputs (never overwritten):
  data/v2/a4/notices.jsonl            one row per notice: source decision, notice, arguments, accepted, leakage
  results/a4_notices.csv              one row per notice, without the texts
  results/a4_notices_review.xlsx      the notices for the author's review
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from jm_eval.chatbot import judging as J
from jm_eval.chatbot import nli as NLI
from jm_eval.drafting import notices as N
from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C

SCRIPT = "80_a4_notices"


def a4_dir():
    path = resolve(load_config("paths")["data"]["v2"]) / "a4"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_corpus(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(resolve(cfg["corpus"])).sort_values("acordao_hash").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="select the decisions, print the prompt sizes and stop (no model call)")
    args = ap.parse_args()
    cfg = N.settings()
    m = cfg["material"]
    models = {"material": load_config("models")["judges"][m["model"]]["model"], "nli": cfg["leakage"]["nli_model"]}
    with RunManifest(SCRIPT, a4_dir(), models=models, params={"a4": cfg, "prompts_version": N.PROMPTS_VERSION,
                                                                       "notice_prompt_version": N.NOTICE_PROMPT_VERSION},
                     seeds={"selection": cfg["selection"]["seed"], "review": cfg["leakage"]["review_seed"]}) as run:
        corpus, labels = load_corpus(cfg), C.load_labels()
        run.add_input(resolve(cfg["corpus"]))
        excluded = N.excluded_decisions(cfg)
        outvoted = corpus["acordao_acordao"].map(N.rapporteur_outvoted)
        chosen = N.select(corpus, labels, excluded, cfg)
        assert not set(chosen["acordao_hash"]) & excluded
        rows = corpus.set_index("acordao_hash").loc[chosen["acordao_hash"]].reset_index().to_dict("records")
        assert not any(N.rapporteur_outvoted(r["acordao_acordao"]) for r in rows)
        summary = {"excluded_used_before": len(excluded), "outvoted_in_corpus": int(outvoted.sum()),
                   "eligible": int(N.eligible(corpus, excluded, cfg, True).sum())}
        print(json.dumps(summary), "\n", chosen[["notice_id", "title", "base_key", "outcome", "cell_size"]].to_string(index=False))
        sizes = {r["acordao_hash"]: {"notice": J.estimated_tokens(N.notice_prompt(r, cfg), N.SYSTEM),
                                  "arguments": J.estimated_tokens(N.arguments_prompt(r, cfg), N.SYSTEM)}
                 for r in rows}
        print("largest prompts (estimated tokens):", {k: max(s[k] for s in sizes.values()) for k in ("notice", "arguments")})
        if args.dry_run:
            run.data["summary"] = {**summary, "dry_run": True}
            return 0

        log: list[dict] = []
        run.data["calls"] = log
        ask = {step: J.Asker(m["model"], step, log=log, rpm=m["rpm"]) for step in ("arguments", "accepted", "notice")}
        nli = NLI.load(cfg["leakage"]["nli_model"])
        out = []
        for sel, row in zip(chosen.to_dict("records"), rows):
            item, extra = sel["notice_id"], {"a4": N.PROMPTS_VERSION, "acordao_hash": sel["acordao_hash"]}
            got = ask["arguments"].ask(N.arguments_prompt(row, cfg),
                                       lambda p: N.validate_arguments(p, m["arguments"]["min"], m["arguments"]["max"]),
                                       item=item, extra=extra, system=N.SYSTEM)
            if got is None:
                raise RuntimeError(f"{item}: no usable argument list")
            arguments = got["arguments"]
            accepted = None
            if sel["outcome"] in ("Provido", "Parcialmente provido"):
                accepted = ask["accepted"].ask(N.accepted_prompt(row, arguments, cfg),
                                               lambda p: N.validate_accepted(p, len(arguments)),
                                               item=item, extra=extra, system=N.SYSTEM)
                if accepted is None:
                    raise RuntimeError(f"{item}: no usable accepted arguments")
            versions = []
            avoid: list[str] = []
            for round_ in range(cfg["leakage"]["rewrite_rounds"] + 1):
                got = ask["notice"].ask(N.notice_prompt(row, cfg, avoid), N.validate_notice, item=item,
                                        extra={**extra, "notice_prompt": N.NOTICE_PROMPT_VERSION, "round": round_}, system=N.SYSTEM)
                if got is None:
                    raise RuntimeError(f"{item}: no usable notice (round {round_})")
                check = N.leakage(got["notice"], arguments, row, cfg, nli)
                versions.append({"round": round_, "notice": got["notice"], "leakage": check})
                if not check["flagged"]:
                    break
                # Flagged for forbidden terms only: the same prompt again (its rules already forbid them).
                avoid = [arguments[k - 1] for k in check["flagged_arguments"]]
            final = versions[-1]
            field_amount = N.penalty_field_amount(row["acordao_penalidade"])
            out.append({**sel, "outvoted": False, "penalty_field_amount": field_amount,
                        "penalty_field_amount_in_document": (field_amount in N.amounts(final["notice"])) if field_amount else None, "acordao_url": row["acordao_url"], "notice": final["notice"],
                        "rounds": len(versions), "leakage": final["leakage"], "still_flagged": final["leakage"]["flagged"],
                        "versions": versions, "arguments": arguments,
                        "accepted": accepted["accepted"] if accepted else None,
                        "granted": accepted["granted"] if accepted else None})
            print(item, sel["outcome"], "arguments:", len(arguments), "rounds:", len(versions),
                  "flagged:", final["leakage"]["flagged"], flush=True)

        path = new_artefact_path(a4_dir() / "notices.jsonl")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
        run.add_output(path, "A4: test notices, the appellants' arguments and the accepted ones (Mistral Medium 3)")

        table = pd.DataFrame([{k: r[k] for k in ("notice_id", "acordao_hash", "title", "base_key", "outcome", "cell_size",
                                                 "replaces", "rounds", "still_flagged", "penalty_field_amount",
                                                 "penalty_field_amount_in_document")}
                              | {"n_arguments": len(r["arguments"]),
                                 "accepted": json.dumps(r["accepted"]) if r["accepted"] is not None else "",
                                 "max_entailment": max(a["entailment"] for a in r["leakage"]["per_argument"]),
                                 "max_containment": max(a["containment"] for a in r["leakage"]["per_argument"]),
                                 "forbidden_terms": ", ".join(r["leakage"]["forbidden_terms"]),
                                 "outcome_amounts": ", ".join(r["leakage"]["outcome_amounts"]),
                                 "notice_chars": len(r["notice"])} for r in out])
        table_path = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_notices.csv")
        table.to_csv(table_path, index=False)
        run.add_output(table_path, "A4: the test notices, one row per notice")

        rng = np.random.default_rng(cfg["leakage"]["review_seed"])
        sample = set(rng.choice([r["notice_id"] for r in out], cfg["leakage"]["review_sample"], replace=False).tolist())
        review = [r for r in out if r["notice_id"] in sample or r["still_flagged"]]
        sheet = pd.DataFrame([{
            "notice_id": r["notice_id"], "why": "sample" if r["notice_id"] in sample else "still flagged",
            "title": r["title"], "acordao_url": r["acordao_url"], "outcome": r["outcome"],
            "notice": r["notice"],
            "appellant_arguments": "\n".join(f"{i}. {a}" for i, a in enumerate(r["arguments"], 1)),
            "accepted": ", ".join(map(str, r["accepted"])) if r["accepted"] else "",
            "granted": r["granted"] or "",
            "leakage_flags": "; ".join(
                [f"argument {a['argument']}: entailment {a['entailment']}, containment {a['containment']}"
                 for a in r["leakage"]["per_argument"] if a["argument"] in r["leakage"]["flagged_arguments"]]
                + ([f"terms: {', '.join(r['leakage']['forbidden_terms'])}"] if r["leakage"]["forbidden_terms"] else [])
                + ([f"outcome amounts: {', '.join(r['leakage']['outcome_amounts'])}"] if r["leakage"]["outcome_amounts"] else [])),
            "penalty_field_amount_in_document": r["penalty_field_amount_in_document"],
            "hugo_notice_ok": "", "hugo_arguments_ok": "", "hugo_comment": ""} for r in review])
        sheet_path = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_notices_review.xlsx")
        sheet.to_excel(sheet_path, index=False)
        run.add_output(sheet_path, "A4: notices for the author's review")
        run.data["summary"] = {**summary, "notices": len(out), "rewritten": sum(r["rounds"] > 1 for r in out),
                               "still_flagged": sum(r["still_flagged"] for r in out), "review": sorted(r["notice_id"] for r in review),
                               "material_calls": J.calls_summary(log)}
        print(json.dumps(run.data["summary"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
