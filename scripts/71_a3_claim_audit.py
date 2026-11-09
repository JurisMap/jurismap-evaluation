"""Phase 6, A3: blind audit of the judges' claim labels by the author (the author, 2026-10-11).

The judges are validated on the task they perform. A sample of the claims of the evaluated answers
is put in a sheet with the evidence passages the judges read and without any judge's label; The author
marks each claim supported, contradicted or not_found against that evidence. The sample is
stratified by how the judges split, so that agreements and disagreements are both covered.

  (default)        build the sheet and the key that links its rows to the claims
  --apply SHEET    each verifier against the author's labels: agreement per stratum and weighted to all
                   claims of the strata

Outputs (never overwritten):
  results/a3_claim_audit.xlsx       the sheet to mark (`hugo_label`, `comment`)
  data/v2/a3/claim_audit_key.csv    audit_id, claim_id, stratum (no label is written to the sheet)
  --apply: results/a3_claim_audit.csv, results/a3_claim_audit.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from jm_eval.chatbot import metrics as M
from jm_eval.chatbot import store
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path
from jm_eval.manifest import RunManifest
from jm_eval.stats import wilson_interval

SCRIPT = "71_a3_claim_audit"
MARKS = {"supported": "supported", "s": "supported", "contradicted": "contradicted", "c": "contradicted",
         "not_found": "not_found", "not found": "not_found", "n": "not_found"}


def strata(claims: pd.DataFrame, raters: list[str]) -> pd.Series:
    """How the verifiers split on each claim: "+" supported, "-" not, "." not judged; e.g. "judge_a+ judge_b-"."""
    def cell(label) -> str:
        return "." if label not in M.LABELS else "+" if label == "supported" else "-"
    return claims[raters].apply(lambda row: " ".join(f"{r}{cell(v)}" for r, v in row.items()), axis=1)


def sample(claims: pd.DataFrame, raters: list[str], n: int, seed: int) -> pd.DataFrame:
    """About `n` claims, the same number from each stratum as far as the strata allow."""
    table = claims.assign(stratum=strata(claims, raters))
    rng = np.random.default_rng(seed)
    sizes = table["stratum"].value_counts().sort_index()
    take = dict.fromkeys(sizes.index, 0)
    remaining = min(n, len(table))
    while remaining > 0:                                       # one more from every stratum that still has claims
        open_strata = [s for s in sizes.index if take[s] < sizes[s]]
        for s in open_strata[:remaining]:
            take[s] += 1
        remaining -= min(len(open_strata), remaining)
    picked = [table[table["stratum"] == s].iloc[np.sort(rng.choice(sizes[s], size=k, replace=False))] for s, k in take.items() if k]
    out = pd.concat(picked).iloc[rng.permutation(sum(take.values()))].reset_index(drop=True)
    out.insert(0, "audit_id", [f"A{i:02d}" for i in range(1, len(out) + 1)])
    return out


def build(args, cfg: dict, run: RunManifest) -> None:
    raters = cfg["faithfulness"]["verifiers"] + cfg["faithfulness"]["separate"]
    claims = store.load_table("claims.parquet")
    raters = [r for r in raters if r in claims.columns]
    questions = {q["question_id"]: q["question"] for q in store.load_questions()}
    picked = sample(claims, raters, cfg["audit"]["n"], cfg["audit"]["seed"])
    sheet = pd.DataFrame({
        "audit_id": picked["audit_id"], "question": picked["question_id"].map(questions), "claim": picked["claim"],
        "evidence": picked["evidence"].map(lambda ev: "\n\n".join(f"[{i}] {p}" for i, p in enumerate(ev, 1))),
        "hugo_label": None, "comment": None})
    sheet_path = new_artefact_path(store.results_dir() / "a3_claim_audit.xlsx")
    notes = pd.DataFrame({"how to mark": [
        "hugo_label: supported, contradicted or not_found (s, c, n are accepted).",
        "supported: the evidence states the claim, or it follows directly from it; every part of the claim is covered, in the decision the claim is about.",
        "contradicted: the evidence states something incompatible with the claim.",
        "not_found: the evidence neither states nor contradicts the claim, or covers only part of it.",
        "Judge against the evidence shown only: it is what the judges read."]})
    with pd.ExcelWriter(sheet_path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="audit", index=False)
        notes.to_excel(writer, sheet_name="instructions", index=False)
    run.add_output(sheet_path, "A3: claims for the author's blind audit of the judges")
    key_path = new_artefact_path(store.work_dir() / "claim_audit_key.csv")
    picked[["audit_id", "claim_id", "stratum"]].to_csv(key_path, index=False)
    run.add_output(key_path, "A3: key of the claim audit")
    summary = {"claims": len(claims), "sampled": len(picked), "strata": picked["stratum"].value_counts().to_dict(),
               "strata_in_all_claims": strata(claims, raters).value_counts().to_dict()}
    run.data["summary"] = summary
    print(json.dumps(summary, indent=1))


def analyse(marked: pd.DataFrame, key: pd.DataFrame, claims: pd.DataFrame, raters: list[str],
            sampled_by: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Each verifier against the audit labels: three labels, and supported against not supported.

    `sampled_by` are the verifiers whose labels defined the strata of the sample (default `raters`);
    the weights to all claims come from those strata.
    """
    table = key.merge(marked[["audit_id", "hugo_label"]], on="audit_id").merge(claims, on="claim_id")
    table["hugo"] = table["hugo_label"].map(lambda v: MARKS.get(str(v).strip().lower()))
    table = table[table["hugo"].notna()]
    weights = strata(claims, sampled_by or raters).value_counts(normalize=True)
    rows = []
    for rater in raters:
        judged = table[table[rater].isin(M.LABELS)]
        same = judged[rater] == judged["hugo"]
        same_binary = (judged[rater] == "supported") == (judged["hugo"] == "supported")
        per_stratum = same_binary.groupby(judged["stratum"]).mean()
        covered = weights[per_stratum.index]
        hugo_yes, rater_yes = judged["hugo"] == "supported", judged[rater] == "supported"
        low, high = wilson_interval(int(same_binary.sum()), len(judged))
        rows.append({
            "rater": rater, "claims": len(judged), "same_label": int(same.sum()),
            "same_label_share": float(same.mean()) if len(judged) else None,
            "same_supported_or_not": int(same_binary.sum()),
            "same_supported_or_not_share": float(same_binary.mean()) if len(judged) else None,
            "ci_low": low, "ci_high": high,
            "weighted_to_all_claims": float((per_stratum * covered).sum() / covered.sum()) if len(per_stratum) else None,
            "supported_by_hugo_rejected_by_rater": int((hugo_yes & ~rater_yes).sum()), "supported_by_hugo": int(hugo_yes.sum()),
            "not_supported_by_hugo_accepted_by_rater": int((~hugo_yes & rater_yes).sum()),
            "not_supported_by_hugo": int((~hugo_yes).sum())})
    summary = {"marked": int(len(table)), "hugo_labels": table["hugo"].value_counts().to_dict(),
               "by_stratum": table.groupby("stratum")["hugo"].value_counts().unstack(fill_value=0).to_dict("index")}
    return pd.DataFrame(rows), summary


def apply(args, cfg: dict, run: RunManifest) -> None:
    claims = store.load_table("claims.parquet")
    raters = [r for r in ["nli", *cfg["faithfulness"]["verifiers"], "majority", *cfg["faithfulness"]["separate"]] if r in claims.columns]
    sampled_by = [r for r in cfg["faithfulness"]["verifiers"] + cfg["faithfulness"]["separate"] if r in claims.columns]
    key = pd.read_csv(latest_artefact_path(store.work_dir() / "claim_audit_key.csv"))
    marked = pd.read_excel(args.apply, sheet_name="audit")
    run.add_input(args.apply)
    table, summary = analyse(marked, key, claims, raters, sampled_by)
    table_path = new_artefact_path(store.results_dir() / "a3_claim_audit.csv")
    table.to_csv(table_path, index=False, float_format="%.4f")
    run.add_output(table_path, "A3: the verifiers against the author's blind labels")
    summary_path = new_artefact_path(store.results_dir() / "a3_claim_audit.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    run.add_output(summary_path, "A3: The author's labels in the claim audit")
    run.data["summary"] = summary
    print(table.round(3).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", help="the sheet the author marked")
    args = ap.parse_args()
    cfg = load_config("a3_chatbot")
    with RunManifest(SCRIPT, store.results_dir(), params={"audit": cfg["audit"], "apply": bool(args.apply)},
                     seeds={"audit": cfg["audit"]["seed"]}) as run:
        (apply if args.apply else build)(args, cfg, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
