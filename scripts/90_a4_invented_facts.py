"""Phase 8, A4 (task 4): invented case facts per draft, from the classifier and the author's blind audit.

The author marked 50 passages of GPT-5.4's "added facts" (results/a4_fact_audit_2.xlsx), 12 or 13 drawn from
each kind the classifier gave (scripts/89). For each classifier kind k, the share of its passages the author
marked invented estimates P(invented | k); a draft's invented facts are estimated as the sum over k of
its passages of kind k times that share. The range uses the Wilson interval of the share for the kind
"invented" (the other kinds contribute little). The classifier's per-mode counts are taken as they are:
the audit is too small to check that its precision is the same in every mode.

Also counted: drafts that claim the record of past offences ("antecedentes") was not reported in the
notice or was added later, a winning ground at the council, which the author's audit found invented (F17, F49).

Outputs (never overwritten):
  results/a4_fact_audit_agreement.csv   classifier kind x the author's kind
  results/a4_invented_facts.csv         per mode: passages by kind, estimated invented facts per draft and range,
                                        share of drafts with the past-offences claim
"""

from __future__ import annotations

import argparse
import glob
import json
import re

import pandas as pd

from jm_eval import stats as S
from jm_eval.drafting import judging as DJ
from jm_eval.drafting import notices as N
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "90_a4_invented_facts"
MODES = ["full", "single_prompt", "no_legal_search", "no_precedents", "no_template"]
PAST_OFFENCES = [
    re.compile(r"antecedentes?[^.]{0,200}?(n[ãa]o\s+(?:foram|foi|constava\w*|constou|estava\w*)?\s*(?:informad|mencionad|indicad|apontad|descrit|consignad)"
               r"|posterior\w*|juntad\w*\s+(?:apenas\s+)?(?:posterior|ap[óo]s|depois|em\s+momento)|aus[êe]ncia\s+de\s+(?:men[çc][ãa]o|previs[ãa]o|indica[çc][ãa]o)"
               r"|sem\s+(?:pr[ée]via\s+)?(?:men[çc][ãa]o|intima[çc][ãa]o|ci[êe]ncia))", re.I | re.S),
    re.compile(r"(n[ãa]o\s+(?:foram|foi|constava\w*|constou)\s*(?:informad|mencionad|indicad|apontad|descrit|consignad)\w*|posterior\w*"
               r"|sem\s+pr[ée]via\s+intima[çc][ãa]o)[^.]{0,150}?antecedentes?", re.I | re.S)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    a4 = N.work_dir()                                      # JM_A4_VERSION selects v2.6 (default) or v2.7
    sheet_path = latest_artefact_path(N.result_path("a4_fact_audit.xlsx"))
    key_path = latest_artefact_path(N.result_path("a4_fact_audit_key.csv"))
    kinds_path = latest_artefact_path(a4 / "fact_kinds.jsonl")
    with RunManifest(SCRIPT, a4, params={}) as run:
        for p in (sheet_path, key_path, kinds_path):
            run.add_input(p)
        marks = pd.read_excel(sheet_path, sheet_name="passages")[["audit_id", "hugo_kind"]]
        provisional = marks["hugo_kind"].isna().all()
        if provisional:                                    # the version's audit not marked yet: v2.6's audited shares
            v26 = resolve(load_config("paths")["results"])
            marks = pd.read_excel(latest_artefact_path(v26 / "a4_fact_audit.xlsx"), sheet_name="passages")[["audit_id", "hugo_kind"]]
            key_path = latest_artefact_path(v26 / "a4_fact_audit_key.csv")
        audit = pd.read_csv(key_path).merge(marks, on="audit_id")
        audit["hugo_kind"] = audit["hugo_kind"].str.strip().str.lower()
        assert set(audit["hugo_kind"]) <= set(DJ.FACT_KINDS), set(audit["hugo_kind"])
        confusion = pd.crosstab(audit["kind"], audit["hugo_kind"]).reindex(index=DJ.FACT_KINDS, columns=DJ.FACT_KINDS, fill_value=0)
        conf_path = new_artefact_path(N.result_path("a4_fact_audit_agreement.csv"))
        confusion.to_csv(conf_path)
        run.add_output(conf_path, "A4: classifier kind x the author's kind on the 50 audited passages")
        share = {k: confusion.loc[k, "invented"] / confusion.loc[k].sum() for k in DJ.FACT_KINDS}
        lo, hi = S.wilson_interval(int(confusion.loc["invented", "invented"]), int(confusion.loc["invented"].sum()))

        rows = [json.loads(line) for line in kinds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        per = pd.DataFrame([{"mode": r["mode"], **{k: (r["kinds"] or []).count(k) for k in DJ.FACT_KINDS}} for r in rows])
        per["estimated_invented"] = sum(per[k] * share[k] for k in DJ.FACT_KINDS)
        per["low"] = per["invented"] * lo + sum(per[k] * share[k] for k in DJ.FACT_KINDS if k != "invented")
        per["high"] = per["invented"] * hi + sum(per[k] * share[k] for k in DJ.FACT_KINDS if k != "invented")

        frozen = {r["notice_id"]: r for r in N.load_frozen()}
        claims = []
        for d in map(json.loads, (a4 / "drafts.jsonl").read_text(encoding="utf-8").splitlines()):
            if d["ok"]:
                text = (a4 / "drafts" / f"{d['notice_id']}_{d['mode']}_r{d['run']}.md").read_text(encoding="utf-8")
                claims.append({"mode": d["mode"], "past_offences_claim": any(p.search(text) for p in PAST_OFFENCES)})
        real_ground = sum(any("antecedente" in a.lower() for a in r["arguments"]) for r in frozen.values())

        table = per.groupby("mode")[[*DJ.FACT_KINDS, "estimated_invented", "low", "high"]].mean().reindex(MODES)
        table["past_offences_claim_share"] = pd.DataFrame(claims).groupby("mode")["past_offences_claim"].mean().reindex(MODES)
        table = table.round(3)
        out = new_artefact_path(N.result_path("a4_invented_facts.csv"))
        table.to_csv(out)
        run.add_output(out, "A4: estimated invented case facts per draft, and the past-offences claim, per mode")
        run.data["summary"] = {"provisional_v26_audit": bool(provisional), "share_invented_by_kind": {k: round(float(v), 3) for k, v in share.items()},
                               "invented_precision_wilson": [round(lo, 3), round(hi, 3)],
                               "agreement": round(float((audit["kind"] == audit["hugo_kind"]).mean()), 3),
                               "real_appeals_raising_past_offences": real_ground}
        print(confusion.to_string()); print(table.to_string()); print(json.dumps(run.data["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
