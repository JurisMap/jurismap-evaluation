"""Phase 6.1: which chat strategy leaves the fewest unsupported statements (development sets only).

Reads, for every candidate of config/a3_candidates.yaml, the tables that scripts/62 to 65 wrote
with `--candidate NAME`, and applies the rule fixed in that file before any candidate was scored:

- measured: the unsupported rate (share of the claims of the open answers that are not supported
  after the retrieval-miss control, mean of GPT-5.4 and Mistral Medium 3), the contradiction
  rate, the answers with a contradicted claim, coverage of the reference points, the templated and
  unanswerable questions, length, latency and cost;
- eligible: coverage, templated answers and abstentions not below the baseline by more than the
  rule allows;
- selected: the lowest unsupported rate among the eligible, ties to the cheapest;
- adopted only if the paired bootstrap interval of its difference from the baseline, over the
  open questions, lies below zero.

Outputs (never overwritten):
  results/a3_candidates.csv               one row per candidate
  results/a3_candidates_selection.json    the rule's outcome
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from jm_eval.chatbot import metrics as M
from jm_eval.chatbot import store
from jm_eval.io import load_config, new_artefact_path
from jm_eval.manifest import RunManifest

SCRIPT = "70_a3_candidate_selection"


def per_question(claims: pd.DataFrame, judges: list[str]) -> pd.DataFrame:
    """Per open question: claims and the expected number of unsupported and contradicted ones, mean of the judges."""
    control = claims.loc[claims["full_text_label"].notna(), ["claim_id", "full_text_label"]]
    rate = M.miss_rate(control)["miss_rate"]
    parts = []
    for judge in judges:
        judged = M.with_control(claims, judge, control, rate)
        judged = judged.assign(unsupported=1.0 - judged["p_supported"])
        parts.append(judged.groupby("question_id").agg(claims=("claim_id", "size"), unsupported=("unsupported", "sum"),
                                                       contradicted=("p_contradicted", "sum"),
                                                       raw_unsupported=(judge, lambda s: float((s != "supported").sum()))))
    return sum(parts) / len(parts)


def measure(name: str, judges: list[str]) -> tuple[dict, pd.DataFrame]:
    """The figures of one candidate and its per-question table."""
    store.use_candidate(name)
    answers = store.load_table("answers.parquet", True)
    det = store.load_table("deterministic.parquet", True)
    claims = store.load_table("claims.parquet", True)
    cov = store.load_table("completeness.parquet", True)
    store.use_candidate(None)

    questions = per_question(claims, judges)
    control = claims[claims["full_text_label"].notna()]
    contradicted_answers = np.mean([claims[claims[j] == "contradicted"]["answer_id"].nunique() for j in judges])
    cov = cov[cov["judge"].isin(judges)].dropna(subset=["coverage"])
    coverage = cov.assign(value=cov["coverage"].map(M.COVERAGE_VALUE)).groupby("judge")["value"].mean().mean()
    templated, unanswerable = det[det["type"] == "templated"], det[det["type"] == "unanswerable"]
    open_answers = answers[answers["type"] == "open"]
    quote_checks = [json.loads(q) for q in answers.get("quote_check", pd.Series(dtype=str)).dropna() if q and q != "null"]
    row = {
        "candidate": name, "open_answers": int(open_answers["answer_id"].nunique()), "claims": int(len(claims)),
        "claims_per_answer": len(claims) / max(1, open_answers["answer_id"].nunique()),
        "unsupported_rate": float(questions["unsupported"].sum() / questions["claims"].sum()),
        "unsupported_rate_raw": float(questions["raw_unsupported"].sum() / questions["claims"].sum()),
        "contradiction_rate": float(questions["contradicted"].sum() / questions["claims"].sum()),
        "unsupported_claims_per_answer": float(questions["unsupported"].sum() / max(1, len(questions))),
        "answers_with_a_contradicted_claim": float(contradicted_answers),
        "miss_control_claims": int(len(control)),
        "miss_rate": M.miss_rate(control[["claim_id", "full_text_label"]])["miss_rate"],
        "coverage": float(coverage),
        "templated_correct": int(templated["correct"].sum()), "templated": int(len(templated)),
        "abstained": int((unanswerable["correct"] == True).sum()), "unanswerable": int(len(unanswerable)),  # noqa: E712
        "answers_with_an_unknown_process_number": int((~det["citations_none_unknown"]).sum()),
        "open_answer_chars_median": float(open_answers["answer"].str.len().median()),
        "latency_s_median_open": float(open_answers["latency_ms"].median() / 1000),
        "cost_usd_per_open_answer": float(open_answers["cost_usd"].mean()),
        "prompt_tokens_per_open_answer": float(open_answers["prompt_tokens"].mean()),
        "quotes": sum(q["quotes"] for q in quote_checks) if quote_checks else None,
        "quotes_found": sum(q["quotes_found"] for q in quote_checks) if quote_checks else None,
        "sentences_removed_by_the_quote_check": sum(q["sentences_removed"] for q in quote_checks) if quote_checks else None,
    }
    return row, questions


def paired_difference(candidate: pd.DataFrame, baseline: pd.DataFrame, resamples: int, seed: int) -> dict:
    """Difference of the unsupported rates (candidate minus baseline), questions resampled together."""
    both = candidate.join(baseline, lsuffix="_c", rsuffix="_b", how="inner")
    values = both[["unsupported_c", "claims_c", "unsupported_b", "claims_b"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diffs = draws[:, 0] / draws[:, 1] - draws[:, 2] / draws[:, 3]
    low, high = np.nanquantile(diffs, [0.025, 0.975])
    total = values.sum(axis=0)
    return {"questions": int(len(values)), "difference": float(total[0] / total[1] - total[2] / total[3]),
            "ci_low": float(low), "ci_high": float(high)}


def apply_rule(table: pd.DataFrame, differences: dict, rule: dict) -> dict:
    """The rule of config/a3_candidates.yaml on the candidates' figures."""
    base = table.set_index("candidate").loc["baseline"]
    limits = rule["constraints"]
    eligibility = {}
    for row in table.to_dict("records"):
        reasons = []
        if row["coverage"] < base["coverage"] - limits["coverage_drop_max"]:
            reasons.append(f"coverage {row['coverage']:.3f} is more than {limits['coverage_drop_max']} below the baseline's {base['coverage']:.3f}")
        if row["templated_correct"] < base["templated_correct"] - limits["templated_correct_drop_max"]:
            reasons.append(f"{row['templated_correct']} templated answers correct against the baseline's {base['templated_correct']}")
        if row["abstained"] < base["abstained"] - limits["abstention_drop_max"]:
            reasons.append(f"{row['abstained']} abstentions against the baseline's {base['abstained']}")
        eligibility[row["candidate"]] = reasons
    eligible = table[table["candidate"].map(lambda c: not eligibility[c])]
    best = float(eligible["unsupported_rate"].min())
    tied = eligible[eligible["unsupported_rate"] <= best + rule["tie_margin"]]
    selected = str(tied.sort_values(["cost_usd_per_open_answer", "unsupported_rate"]).iloc[0]["candidate"])
    difference = differences.get(selected)
    adopted = selected != "baseline" and difference is not None and difference["ci_high"] < 0
    return {"eligible": [c for c, reasons in eligibility.items() if not reasons],
            "not_eligible": {c: reasons for c, reasons in eligibility.items() if reasons},
            "lowest_unsupported_rate": best, "tied": tied["candidate"].tolist(), "selected": selected,
            "difference_from_baseline": difference, "replaces_the_baseline": bool(adopted)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", help="only these candidates (default: every candidate of the config)")
    args = ap.parse_args()
    cfg = load_config("a3_candidates")
    judges, rule = cfg["measure"]["judges"], cfg["rule"]
    names = args.candidates or list(cfg["candidates"])
    seed = load_config("seeds")["bootstrap"]
    results = store.results_dir()
    with RunManifest(SCRIPT, results, params={"candidates": cfg, "names": names}, seeds={"bootstrap": seed}) as run:
        rows, tables = [], {}
        for name in names:
            row, tables[name] = measure(name, judges)
            rows.append({**row, "label": cfg["candidates"][name]["label"]})
        table = pd.DataFrame(rows)
        differences = {name: paired_difference(tables[name], tables["baseline"], rule["bootstrap_resamples"], seed)
                       for name in names if name != "baseline"}
        for column in ("difference", "ci_low", "ci_high"):
            table[f"unsupported_rate_{column}_from_baseline"] = table["candidate"].map(lambda c: differences.get(c, {}).get(column))
        outcome = apply_rule(table, differences, rule)
        table["eligible"] = table["candidate"].isin(outcome["eligible"])
        table["selected"] = table["candidate"] == outcome["selected"]

        table_path = new_artefact_path(results / "a3_candidates.csv")
        table.to_csv(table_path, index=False, float_format="%.4f")
        run.add_output(table_path, "A3 (Phase 6.1): chat strategies on the development sets")
        outcome_path = new_artefact_path(results / "a3_candidates_selection.json")
        outcome_path.write_text(json.dumps({"rule": rule, "judges": judges, "development_sets": cfg["development_sets"],
                                            **outcome}, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(outcome_path, "A3 (Phase 6.1): outcome of the selection rule")
        run.data["summary"] = outcome
        shown = ["candidate", "claims", "unsupported_rate", "unsupported_rate_raw", "contradiction_rate",
                 "answers_with_a_contradicted_claim", "coverage", "templated_correct", "abstained",
                 "open_answer_chars_median", "latency_s_median_open", "cost_usd_per_open_answer", "eligible"]
        print(table[shown].round(3).to_string(index=False))
        print(json.dumps(outcome, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
