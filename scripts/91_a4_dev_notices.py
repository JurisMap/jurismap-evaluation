"""Phase 8, A4 improvement round: development notices (ground rule 8: never the test notices).

Six decisions, one in each of six cells of the test notices' grid (5 legal-basis keys x 3 outcomes)
drawn at random, with the test notices' eligibility rules, excluding the decisions of Phases 6 and 7,
outvoted rapporteurs, and the 15 test notices with their processes. Each gets what a test notice got
(scripts/80): the appellant's arguments and, for granted appeals, the accepted ones (Mistral, reading
the dispositive), and the case document (notice prompt version 3) with the leakage check and one
rewrite round. The author gives them a quick review.

Outputs:
  data/v2/a4_dev/notices.jsonl          the development notices (never overwritten)
  results/a4_dev_notices_review.xlsx    the six for the author's review
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

SCRIPT = "91_a4_dev_notices"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = N.settings()
    spec, m = cfg["improvement"]["dev_notices"], cfg["material"]
    out_dir = resolve(load_config("paths")["data"]["v2"]) / "a4_dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(resolve(cfg["corpus"])).sort_values("acordao_hash").reset_index(drop=True)
    labels = C.load_labels()
    assert corpus["acordao_hash"].tolist() == labels["acordao_hash"].tolist()
    test = N.load_frozen(cfg)
    test_groups = set(labels.set_index("acordao_hash").loc[[r["acordao_hash"] for r in test], "process_group"])
    ok = N.eligible(corpus, N.excluded_decisions(cfg), cfg, True).to_numpy() & ~labels["process_group"].isin(test_groups).to_numpy()
    cells = [(k, o) for k in N.strata(labels, cfg["selection"]["n_strata"]) for o in cfg["selection"]["outcomes"]]
    rng = np.random.default_rng(spec["seed"])
    chosen, used = [], set()
    for i in rng.permutation(len(cells)):           # cells in random order; an empty cell is skipped
        if len(chosen) == spec["n"]:
            break
        key, outcome = cells[i]
        pool = [j for j in np.flatnonzero(ok & (labels["base_key"] == key).to_numpy() & (corpus["acordao_resultado"] == outcome).to_numpy())
                if labels["process_group"].iat[j] not in used]
        if not pool:
            continue
        j = int(rng.choice(pool))
        used.add(labels["process_group"].iat[j])
        chosen.append({"notice_id": f"D{len(chosen) + 1:02d}", "acordao_hash": corpus["acordao_hash"].iat[j],
                       "title": corpus["acordao_titulo"].iat[j], "base_key": key, "outcome": outcome, "cell_size": len(pool)})
    models = {"material": load_config("models")["judges"][m["model"]]["model"], "nli": cfg["leakage"]["nli_model"]}
    with RunManifest(SCRIPT, out_dir, models=models, params={"dev_notices": spec, "notice_prompt_version": N.NOTICE_PROMPT_VERSION},
                     seeds={"dev_notices": spec["seed"]}) as run:
        log: list[dict] = []
        run.data["calls"] = log
        ask = {s: J.Asker(m["model"], s, log=log, rpm=m["rpm"]) for s in ("arguments", "accepted", "notice")}
        nli = NLI.load(cfg["leakage"]["nli_model"])
        rows = corpus.set_index("acordao_hash")
        out = []
        for sel in chosen:
            row = rows.loc[sel["acordao_hash"]].to_dict()
            extra = {"a4": N.PROMPTS_VERSION, "acordao_hash": sel["acordao_hash"], "dev": True}
            got = ask["arguments"].ask(N.arguments_prompt(row, cfg), lambda p: N.validate_arguments(p, m["arguments"]["min"], m["arguments"]["max"]),
                                       item=sel["notice_id"], extra=extra, system=N.SYSTEM)
            arguments = got["arguments"]
            accepted = None
            if sel["outcome"] in ("Provido", "Parcialmente provido"):
                accepted = ask["accepted"].ask(N.accepted_prompt(row, arguments, cfg), lambda p, n=len(arguments): N.validate_accepted(p, n),
                                               item=sel["notice_id"], extra=extra, system=N.SYSTEM)
            avoid, versions = [], []
            for round_ in range(cfg["leakage"]["rewrite_rounds"] + 1):
                got = ask["notice"].ask(N.notice_prompt(row, cfg, avoid), N.validate_notice, item=sel["notice_id"],
                                        extra={**extra, "notice_prompt": N.NOTICE_PROMPT_VERSION, "round": round_}, system=N.SYSTEM)
                check = N.leakage(got["notice"], arguments, row, cfg, nli)
                versions.append({"round": round_, "notice": got["notice"], "leakage": check})
                if not check["flagged"]:
                    break
                avoid = [arguments[k - 1] for k in check["flagged_arguments"]]
            out.append({**sel, "acordao_url": row["acordao_url"], "notice": versions[-1]["notice"], "leakage": versions[-1]["leakage"],
                        "versions": versions, "arguments": arguments,
                        "accepted": accepted["accepted"] if accepted else None, "granted": accepted["granted"] if accepted else None})
            print(sel["notice_id"], sel["title"], sel["outcome"], "flagged:", versions[-1]["leakage"]["flagged"], flush=True)
        path = new_artefact_path(out_dir / "notices.jsonl")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
        run.add_output(path, "A4 improvement round: development notices")
        sheet = pd.DataFrame([{"notice_id": r["notice_id"], "title": r["title"], "acordao_url": r["acordao_url"], "outcome": r["outcome"],
                               "notice": r["notice"], "appellant_arguments": "\n".join(f"{i}. {a}" for i, a in enumerate(r["arguments"], 1)),
                               "accepted": ", ".join(map(str, r["accepted"] or [])), "granted": r["granted"] or "",
                               "outcome_amounts": ", ".join(r["leakage"]["outcome_amounts"]),
                               "hugo_notice_ok": "", "hugo_arguments_ok": "", "hugo_comment": ""} for r in out])
        sheet_path = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_dev_notices_review.xlsx")
        sheet.to_excel(sheet_path, index=False)
        run.add_output(sheet_path, "A4 improvement round: development notices for the author's review")
        run.data["summary"] = {"notices": [r["notice_id"] for r in out], "calls": J.calls_summary(log)}
        print(json.dumps(run.data["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
