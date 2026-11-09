"""Phase 8, A4 (tasks 5 to 8): results of the drafting evaluation and ablation. No model is called.

Inputs: the judges' answers (data/v2/a4/judgments*.jsonl, merged: the newest usable answer per draft,
judge and task), the automatic metrics (results/a4_drafting*.csv), the frozen notices and the drafts'
server records. Choices (DECISIONS 2026-10-18, config/a4_drafting.yaml `analysis`):
- the reported judge is GPT-5.4 (judge_a); Mistral (judge_b) is beside it, with Krippendorff's ordinal
  alpha between them per criterion; Gemini 3.8 Flash is reported separately;
- an argument is raised when at least two of GPT-5.4, Mistral and the NLI model say so; until Mistral's
  answers exist, recall is GPT-5.4's alone and marked provisional;
- the draft that cites its own source decision is a failure mode, out of recall and the rubric (a
  sensitivity check keeps it); a lost draft is the worst outcome (rubric 1, recall 0) in the main
  analysis, and the sensitivity analysis drops the notices with an incomplete set;
- the unit of the tests is the notice (the mean over its three runs): Friedman across the five modes,
  then Wilcoxon signed-rank tests of full against each ablation with Holm correction.

Outputs (never overwritten):
  results/a4_drafting_scores.csv     one row per draft: rubric by judge, added facts, recall
  results/a4_ablation_stats.csv      per metric and analysis: means per mode, Friedman, Wilcoxon (Holm)
  results/a4_judge_agreement.csv     alpha between GPT-5.4 and Mistral per criterion
  results/failure_modes.csv          the Phase 8 rates filled in
  results/figures/a4_ablation.pdf    rubric (GPT-5.4) and recall per mode
"""

from __future__ import annotations

import argparse
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from jm_eval import stats as S  # noqa: E402
from jm_eval.drafting import judging as DJ  # noqa: E402
from jm_eval.drafting import notices as N  # noqa: E402
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve  # noqa: E402
from jm_eval.manifest import RunManifest  # noqa: E402

SCRIPT = "88_a4_report"
MODES = ["full", "single_prompt", "no_legal_search", "no_precedents", "no_template"]


def merged_judgments(a4) -> dict:
    """(notice, mode, run, judge, task) -> the newest usable answer, for drafts written now (judgments of
    superseded drafts, e.g. v2.6's N10 on the old notice, are left out)."""
    written = {(d["notice_id"], d["mode"], d["run"]) for d in map(json.loads, (a4 / "drafts.jsonl").read_text(encoding="utf-8").splitlines())
               if d["ok"]}
    out = {}
    for path in sorted(glob.glob(str(a4 / "judgments*.jsonl")), key=lambda p: (len(p), p)):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["answer"] is not None and (r["notice_id"], r["mode"], r["run"]) in written:
                out[(r["notice_id"], r["mode"], r["run"], r["judge"], r["task"])] = r["answer"]
    return out


def draft_rows(cfg, a4, results) -> pd.DataFrame:
    frozen = {r["notice_id"]: r for r in N.load_frozen(cfg)}
    auto = pd.read_csv(latest_artefact_path(N.result_path("a4_drafting.csv")))
    guard = {}
    for d in map(json.loads, (a4 / "drafts.jsonl").read_text(encoding="utf-8").splitlines()):
        guard.setdefault((d["notice_id"], d["mode"], d["run"]), d.get("fact_guard"))
    judged = merged_judgments(a4)
    # Drafts citing their own source decision (the citation metric), the failure mode the analysis sets
    # apart; for v2.6 this is config `analysis.leaked` (N11 no template run 2).
    leaked = {(r["notice_id"], r["mode"], int(r["run"])) for r in auto.to_dict("records") if r.get("cited_source") is True}
    rows = []
    for a in auto.to_dict("records"):
        key = (a["notice_id"], a["mode"], int(a["run"]))
        row = {"notice_id": key[0], "mode": key[1], "run": key[2], "outcome": a["outcome"], "lost": not a["ok"],
               "leaked": key in leaked}
        notice = frozen[key[0]]
        for judge in ("judge_a", "judge_b", "judge_gemini"):
            ans = judged.get((*key, judge, "rubric"))
            for c in DJ.CRITERIA:
                row[f"{judge}_{c}"] = ans["scores"][c] if ans else np.nan
            row[f"{judge}_rubric_mean"] = np.mean(list(ans["scores"].values())) if ans else np.nan
            if ans:
                facts = DJ.added_facts(ans)
                row[f"{judge}_added_facts"] = len(facts)
                row[f"{judge}_added_missed_by_guard"] = len(DJ.missed_by_guard(facts, guard.get(key)))
        votes = {j: judged.get((*key, j, "arguments")) for j in ("judge_a", "judge_b")}
        nli = judged.get((*key, "nli", "arguments"))
        n_args = len(notice["arguments"])
        labels_a = votes["judge_a"]["labels"] if votes["judge_a"] else None
        labels_b = votes["judge_b"]["labels"] if votes["judge_b"] else None
        nli_raised = [v["raised"] for v in nli["votes"]] if nli else None
        if labels_a is not None and labels_b is not None and nli_raised is not None:
            raised = [(la == "raised") + (lb == "raised") + nr >= 2 for la, lb, nr in zip(labels_a, labels_b, nli_raised)]
            row["recall_basis"] = "majority"
        elif labels_a is not None:
            raised = [la == "raised" for la in labels_a]
            row["recall_basis"] = "judge_a (provisional)"
        else:
            raised = None
        if raised is not None:
            row["recall"] = sum(raised) / n_args
            row["partly_a"] = sum(la == "partly" for la in labels_a) / n_args if labels_a else np.nan
            acc = notice.get("accepted")
            if acc:
                row["accepted_raised"] = sum(raised[i - 1] for i in acc) / len(acc)
        rows.append(row)
    return pd.DataFrame(rows)


def notice_means(df: pd.DataFrame, metric: str, *, lost_worst: bool, keep_leaked: bool, worst: float) -> pd.DataFrame:
    """Per notice and mode: the mean of a metric over the runs (lost drafts as `worst`, or left out)."""
    d = df.copy()
    if not keep_leaked:
        d.loc[d["leaked"], metric] = np.nan
    if lost_worst:
        d.loc[d["lost"], metric] = worst
    else:
        d = d[~d["lost"]]
    return d.groupby(["notice_id", "mode"])[metric].mean().unstack("mode").reindex(columns=MODES)


def test_modes(table: pd.DataFrame) -> dict:
    complete = table.dropna()
    out = {"notices": len(complete)}
    if len(complete) >= 3:
        out["friedman_chi2"], out["friedman_p"] = st.friedmanchisquare(*[complete[m] for m in MODES])
        ps = []
        for m in MODES[1:]:
            diff = complete["full"] - complete[m]
            p = st.wilcoxon(complete["full"], complete[m]).pvalue if (diff != 0).any() else 1.0
            out[f"full_minus_{m}"] = float(diff.mean())
            ps.append(p)
        for m, p, ph in zip(MODES[1:], ps, S.holm(ps)):
            out[f"wilcoxon_p_{m}"], out[f"holm_p_{m}"] = float(p), float(ph)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = N.settings()
    a4 = N.work_dir()                                      # JM_A4_VERSION selects v2.6 (default) or v2.7
    results = resolve(load_config("paths")["results"])
    with RunManifest(SCRIPT, a4, params={"analysis": cfg["analysis"], "judging": cfg["judging"]}) as run:
        df = draft_rows(cfg, a4, results)
        auto = pd.read_csv(latest_artefact_path(N.result_path("a4_drafting.csv")))
        df = df.merge(auto.drop(columns=["outcome", "ok", "lost"], errors="ignore"), on=["notice_id", "mode", "run"], how="left")
        scores_path = new_artefact_path(N.result_path("a4_drafting_scores.csv"))
        df.to_csv(scores_path, index=False)
        run.add_output(scores_path, "A4: one row per draft, judges' scores, recall and automatic metrics")

        metrics = {f"judge_a_{c}": 1.0 for c in (*DJ.CRITERIA, "rubric_mean")} | {"recall": 0.0, "accepted_raised": 0.0,
                   "template_coverage": 0.0, "cited_supplied": 0.0, "judge_a_added_missed_by_guard": np.nan}
        stats_rows = []
        for metric, worst in metrics.items():
            if metric not in df or df[metric].notna().sum() == 0:
                continue
            for name, lost_worst, keep_leaked in (("main", cfg["analysis"]["lost_as_worst"] and not np.isnan(worst), False),
                                                  ("sensitivity_complete_only", False, False),
                                                  ("sensitivity_with_leaked", cfg["analysis"]["lost_as_worst"] and not np.isnan(worst), True)):
                table = notice_means(df, metric, lost_worst=lost_worst, keep_leaked=keep_leaked, worst=worst)
                stats_rows.append({"metric": metric, "analysis": name, **{f"mean_{m}": table[m].mean() for m in MODES},
                                   **test_modes(table)})
        stats_path = new_artefact_path(N.result_path("a4_ablation_stats.csv"))
        pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
        run.add_output(stats_path, "A4: means per mode, Friedman and Wilcoxon (Holm), main and sensitivity analyses")

        agree = []
        for c in (*DJ.CRITERIA, "rubric_mean"):
            a, b = df[f"judge_a_{c}"], df[f"judge_b_{c}"]
            if b.notna().any():
                both = df[a.notna() & b.notna() & ~df["leaked"]]
                agree.append({"criterion": c, "drafts": len(both),
                              "alpha_ordinal": S.krippendorff_alpha_ordinal([[x, y] for x, y in zip(both[f"judge_a_{c}"], both[f"judge_b_{c}"])])
                              if c != "rubric_mean" else None,
                              "mean_gpt": both[f"judge_a_{c}"].mean(), "mean_mistral": both[f"judge_b_{c}"].mean(),
                              "mean_gemini": df[f"judge_gemini_{c}"].mean()})
        if agree:
            agree_path = new_artefact_path(N.result_path("a4_judge_agreement.csv"))
            pd.DataFrame(agree).to_csv(agree_path, index=False)
            run.add_output(agree_path, "A4: agreement between GPT-5.4 and Mistral per rubric criterion")

        # Failure modes: the Phase 8 rates
        ok = df[~df["lost"]]
        n = len(df)
        rates = {
            "Hallucinated precedent": (f"{int(((ok['process_not_in_corpus'] + ok['ruling_unresolved'] + ok['ruling_process_mismatch']) > 0).sum())} of {len(ok)} "
                                       "written drafts cite a precedent not in the corpus or under another decision's or no decision's ruling number "
                                       f"(after the manual check of results/a4_citations_unresolved*.csv); {int(ok['cited_source'].sum())} cites its own source decision"),
            "Hallucinated statute or article": (f"{int(ok['statute_beyond'].sum())} cited articles beyond their act's last article, of "
                                                f"{int((ok['statute_found'] + ok['statute_in_range'] + ok['statute_beyond']).sum())} graded "
                                                f"(+{int(ok['statute_in_range'].sum())} in range not read, {int(ok['statute_unknown_act'].sum())} of unknown acts: manual review)"),
            "Legal search unavailable or empty": "n/a: not measurable from the logs (the LegalSearcher calls record no grounding metadata or tool use)",
            "No template uploaded": (f"template section coverage {df.loc[df['mode']=='no_template','template_coverage'].mean():.2f} without the template "
                                     f"against {df.loc[df['mode']=='full','template_coverage'].mean():.2f} in full mode"),
            "Facts misread from the notice": (f"{ok['judge_a_added_missed_by_guard'].mean():.1f} facts per draft that the case document does not state "
                                              f"and the fact guard neither replaced nor flagged (GPT-5.4); the guard flagged {ok['fact_guard_flagged'].mean():.1f} "
                                              f"and replaced {ok['fact_guard_replaced'].mean():.2f} per draft") if "judge_a_added_missed_by_guard" in ok else "pending",
            "Agent loop or timeout": (f"{int((df['latency_s'] > 300).sum())} of {n} draft requests over 300 s (median "
                                      f"{df['latency_s'].median():.0f} s; full mode {df.loc[df['mode']=='full','latency_s'].median():.0f} s)"),
            "Output blocked (recitation or safety)": (f"{int((ok['blocked_retries'] > 0).sum())} written drafts recovered after a blocked response; "
                                                      f"{int(df['lost'].sum())} of {n} lost (all recitation blocks, notices "
                                                      f"{', '.join(sorted(df.loc[df['lost'],'notice_id'].unique()))})"),
        }
        fm = pd.read_csv(latest_artefact_path(results / "failure_modes.csv"))
        for failure, rate in rates.items():
            fm.loc[fm["failure"] == failure, "measured_rate"] = rate
        fm_path = new_artefact_path(results / "failure_modes.csv")
        fm.to_csv(fm_path, index=False)
        run.add_output(fm_path, "Failure modes with the Phase 8 rates")

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        main = pd.DataFrame(stats_rows).query("analysis == 'main'").set_index("metric")
        crit = [f"judge_a_{c}" for c in DJ.CRITERIA if f"judge_a_{c}" in main.index]
        x = np.arange(len(crit))
        for i, m in enumerate(MODES):
            axes[0].bar(x + (i - 2) * 0.16, [main.loc[c, f"mean_{m}"] for c in crit], 0.16, label=m.replace("_", " "))
        axes[0].set_xticks(x, [c.replace("judge_a_", "").replace("_", "\n") for c in crit], fontsize=8)
        axes[0].set_ylim(1, 5)
        axes[0].set_ylabel("GPT-5.4 score (1-5)")
        axes[0].legend(fontsize=7, ncol=2)
        if "recall" in main.index:
            axes[1].bar(range(len(MODES)), [main.loc["recall", f"mean_{m}"] for m in MODES], color="0.5")
            axes[1].set_xticks(range(len(MODES)), [m.replace("_", "\n") for m in MODES], fontsize=8)
            axes[1].set_ylim(0, 1)
            axes[1].set_ylabel("argument recall")
        fig.tight_layout()
        fig_path = new_artefact_path(N.result_path("figures/a4_ablation.pdf"))
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path)
        run.add_output(fig_path, "A4: rubric and recall per mode")
        run.data["summary"] = {"drafts": n, "recall_basis": df["recall_basis"].value_counts(dropna=False).to_dict()}
        print(pd.DataFrame(stats_rows).query("analysis == 'main'")[["metric", *[f"mean_{m}" for m in MODES], "notices", "friedman_p"]]
              .round(3).to_string(index=False))
        print(json.dumps(run.data["summary"], default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
