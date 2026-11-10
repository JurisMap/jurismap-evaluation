"""Phase 8 (A4), recorded in Phase 11: v2.6 against v2.7 on the same 15 test notices, paired by notice.

The comparison was first reported in DECISIONS.md (2026-10-23, 2026-10-24) and PHASE_8_SUMMARY.md from an
analysis that wrote no file. This script computes it from the files the two versions' reports wrote, so that
every number of that table has a source (Phase 11, results/RESULTS.md). It makes no model call.

Inputs (the newest version of each):
  results/a4_drafting_scores.csv, results/a4_v27_drafting_scores.csv     per draft: rubric, recall, metrics
  results/a4_fact_audit_agreement.csv, results/a4_v27_fact_audit_agreement.csv   each version's blind audit
  data/v2/a4/fact_kinds.jsonl, data/v2/a4_v27/fact_kinds.jsonl           the classifier's kind per passage

Unit: the notice (mean over its written drafts, every mode and run); Wilcoxon signed-rank over the notices.
Lost drafts and the draft that cites its own source decision are left out (jm_eval.drafting.compare).

Output (never overwritten):
  results/a4_versions.csv    per measure: v2.6 and v2.7 means over notices, pooled draft means, the Wilcoxon test
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from jm_eval.drafting import compare as V
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "97_a4_version_comparison"
VERSIONS = {"v2.6": ("a4", "a4"), "v2.7": ("a4_v27", "a4_v27")}     # version -> (data directory, results prefix)
METRICS = {
    "estimated_invented": "Invented case facts per draft (estimated; each version corrected by its own audit)",
    "judge_a_fidelity_to_facts": "Fidelity to facts, GPT-5.4 (1-5)",
    "judge_gemini_fidelity_to_facts": "Fidelity to facts, Gemini 3.8 Flash (1-5; same family as the system)",
    "judge_a_legal_accuracy": "Legal accuracy, GPT-5.4 (1-5)",
    "judge_a_use_of_precedents": "Use of precedents, GPT-5.4 (1-5)",
    "judge_a_structure": "Structure, GPT-5.4 (1-5)",
    "judge_a_persuasiveness": "Persuasiveness, GPT-5.4 (1-5)",
    "judge_a_rubric_mean": "Rubric mean, GPT-5.4 (1-5)",
    "recall": "Argument recall (majority of GPT-5.4, Mistral Medium 3 and the NLI model)",
    "accepted_raised": "Accepted argument raised (granted notices only)",
    "template_coverage": "Template coverage (share of the template's sections)",
    "cost_usd": "Cost per draft (USD)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    paths = load_config("paths")
    results, data_v2 = resolve(paths["results"]), resolve(paths["data"]["v2"])
    with RunManifest(SCRIPT, results, params={"versions": list(VERSIONS), "metrics": METRICS,
                                              "unit": "notice", "test": "scipy.stats.wilcoxon (two-sided)"}) as run:
        tables = {}
        for version, (data_dir, prefix) in VERSIONS.items():
            scores_path = latest_artefact_path(results / f"{prefix}_drafting_scores.csv")
            audit_path = latest_artefact_path(results / f"{prefix}_fact_audit_agreement.csv")
            kinds_path = latest_artefact_path(data_v2 / data_dir / "fact_kinds.jsonl")
            for p in (scores_path, audit_path, kinds_path):
                run.add_input(p, version=version)
            scores = pd.read_csv(scores_path)
            shares = V.invented_shares(pd.read_csv(audit_path, index_col=0))
            kinds = [json.loads(line) for line in kinds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            invented = V.invented_per_draft(kinds, shares)
            merged = scores.merge(invented, on=["notice_id", "mode", "run"], how="left", validate="one_to_one")
            kept = V.written(merged)
            assert kept["estimated_invented"].notna().all(), f"{version}: a written draft has no classified facts"
            tables[version] = kept
            run.data["summary"][version] = {"drafts": len(scores), "lost": int(scores["lost"].astype(bool).sum()),
                                            "leaked": int(scores["leaked"].astype(bool).sum()), "compared": len(kept),
                                            "share_invented_by_kind": {k: round(v, 3) for k, v in shares.items()}}
        table = V.paired_by_notice(tables["v2.6"], tables["v2.7"], METRICS)
        table.insert(0, "version_b", "v2.7")
        table.insert(0, "version_a", "v2.6")
        out = new_artefact_path(results / "a4_versions.csv")
        table.to_csv(out, index=False, float_format="%.6g")
        run.add_output(out, "A4: v2.6 against v2.7, written drafts paired by notice (Wilcoxon)")
        print(table[["metric", "notices", "mean_a", "mean_b", "wilcoxon_p"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
