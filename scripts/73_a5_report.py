"""Phase 7, A5 (tasks 2 and 3): results of the context-size test, from the working tables.

For each setting of config/a5_context_size.yaml (k = 5, 10, 20, 30 and the hierarchical summary at
30) it computes the Phase 6 figures with Phase 6's own functions (scripts/68): templated accuracy,
citations, faithfulness raw and after the retrieval-miss control, latency and cost, and completeness
unless `report.completeness` is false in the config (it is: DECISIONS.md, 2026-10-18). The
differences from k = 10 (and of the hierarchical summary from k = 30) are paired by question: per
answer, the share of supported claims after the control (and the coverage), with a percentile
interval over resampled questions. No model is called.

Outputs (never overwritten):
  results/a5_context_size.csv           every figure: setting, k, section, group, metric, rater, n, value, interval
  results/a5_context_size_paired.csv    differences from the reference setting, paired by question
  results/figures/a5_context_size.pdf   faithfulness, latency and cost against k (and coverage)
  results/costs_phase7.csv              calls, tokens and cost per step and model paid in this phase
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jm_eval.chatbot import metrics as M  # noqa: E402
from jm_eval.chatbot import store  # noqa: E402
from jm_eval.io import load_config, new_artefact_path  # noqa: E402
from jm_eval.manifest import RunManifest  # noqa: E402

SCRIPT = "73_a5_report"
REFERENCE = {"k5": "k10", "k20": "k10", "k30": "k10", "h30": "k30"}     # what each setting is compared with
# Figure colours as in the A1 and A2 figures; the two judges carry a hue, marker shapes repeat it.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
JUDGE_STYLE = {"judge_a": ("GPT-5.4", "#2a78d6", "o"), "judge_b": ("Mistral Medium 3", "#eb6834", "s")}
SYSTEM_COLOUR = "#52514e"


def phase6_report():
    spec = importlib.util.spec_from_file_location("a3_report", Path(__file__).with_name("68_a3_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setting_tables(name: str) -> dict[str, pd.DataFrame]:
    store.use_setting(name)
    try:
        return {"questions": {q["question_id"]: q for q in store.load_questions()},
                **{t: store.load_table(f"{t}.parquet") for t in ("answers", "deterministic", "claims", "completeness")}}
    finally:
        store.use_setting(None)


def extra_system_rows(R, answers: pd.DataFrame) -> list[dict]:
    """What Phase 6 did not need: answers cut by the output limit, and the quote check."""
    rows = [R.row("system", "all", "answers_not_finished", "", len(answers), int((~answers["finish_reason"].astype(str).str.endswith("STOP")).sum()),
                  note="finish reason other than STOP (e.g. the output limit)")]
    checks = [json.loads(q) for q in answers["quote_check"] if isinstance(q, str) and q != "null"]
    if checks:
        removed = sum(c.get("sentences_removed", 0) or 0 for c in checks)
        rows.append(R.row("system", "all", "quote_check_removed_sentences", "", len(checks), removed,
                          note=f"in {sum(1 for c in checks if c.get('sentences_removed'))} answers"))
    for group, sub in [("open", answers[answers["type"] == "open"])]:
        rows.append(R.row("system", group, "answer_chars_median", "", len(sub), sub["answer"].str.len().median()))
    return rows


def controlled_rows(R, claims: pd.DataFrame, verifiers: list[str], resamples: int, seed: int) -> list[dict]:
    """After the control, with intervals over resampled questions: the supported share, and the
    unsupported claims per answer (the plan's answer-level rate is near 1 in every setting)."""
    control = claims.loc[claims["full_text_label"].notna(), ["claim_id", "full_text_label"]]
    rate = M.miss_rate(control)["miss_rate"]
    rows = []
    for rater in verifiers:
        judged = M.with_control(claims, rater, control, rate)
        by_answer = judged.groupby("answer_id").agg(question_id=("question_id", "first"), claims=("p_supported", "size"),
                                                    supported=("p_supported", "sum"))
        ci = R.cluster_bootstrap(by_answer["supported"], by_answer["claims"], by_answer["question_id"], resamples, seed)
        rows.append(R.row("faithfulness", "all", "faithfulness_corrected_ci", rater, int(by_answer["claims"].sum()),
                          by_answer["supported"].sum() / by_answer["claims"].sum(), *ci,
                          note="after the retrieval-miss control; interval over resampled questions"))
        unsupported = by_answer["claims"] - by_answer["supported"]
        ci = R.cluster_bootstrap(unsupported, pd.Series(1.0, index=by_answer.index), by_answer["question_id"], resamples, seed)
        rows.append(R.row("faithfulness", "all", "unsupported_claims_per_answer", rater, len(by_answer), unsupported.mean(), *ci,
                          note="claims not supported after the retrieval-miss control, per open answer"))
    return rows


def per_answer_values(claims: pd.DataFrame, per_answer_cov: pd.DataFrame, verifiers: list[str], judges: list[str]) -> pd.DataFrame:
    """Per open answer: share of supported claims after the control (each verifier) and coverage (each judge)."""
    control = claims.loc[claims["full_text_label"].notna(), ["claim_id", "full_text_label"]]
    rate = M.miss_rate(control)["miss_rate"]
    out = []
    for rater in verifiers:
        judged = M.with_control(claims, rater, control, rate)
        part = judged.groupby("question_id")["p_supported"].mean().rename("value").reset_index()
        out.append(part.assign(metric="faithfulness_corrected", rater=rater))
    for judge in judges:
        part = per_answer_cov[per_answer_cov["judge"] == judge].groupby("question_id")["coverage"].mean().rename("value").reset_index()
        out.append(part.assign(metric="coverage", rater=judge))
    return pd.concat(out, ignore_index=True)


def paired_rows(values: dict[str, pd.DataFrame], resamples: int, seed: int) -> list[dict]:
    rows = []
    for name, reference in REFERENCE.items():
        if name not in values or reference not in values:
            continue
        merged = values[name].merge(values[reference], on=["question_id", "metric", "rater"], suffixes=("", "_ref"))
        for (metric, rater), sub in merged.groupby(["metric", "rater"], sort=True):
            diff = (sub["value"] - sub["value_ref"]).to_numpy()
            rng = np.random.default_rng(seed)
            draws = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append({"setting": name, "reference": reference, "metric": metric, "rater": rater, "questions": len(sub),
                         "mean": round(float(sub["value"].mean()), 4), "mean_reference": round(float(sub["value_ref"].mean()), 4),
                         "difference": round(float(diff.mean()), 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4),
                         "note": "mean over the open questions asked in both settings; one answer each; interval over resampled questions"})
    return rows


def cost_rows(cfg5: dict) -> pd.DataFrame:
    """Calls, tokens and cost paid in this phase: the chatbot's answers and every judge or material call not cached."""
    rows = []
    for name, setting in cfg5["settings"].items():
        if setting.get("reuse_phase6"):
            continue
        store.use_setting(name)
        versions = sorted(store.work_dir().glob("answers*.parquet"))
        store.use_setting(None)
        if not versions:
            continue
        a = pd.concat([pd.read_parquet(v) for v in versions], ignore_index=True).drop_duplicates("request_id")
        served = a["credentials_used"].map(json.loads)
        total_calls = sum(sum(c.values()) for c in served)
        rows.append({"setting": name, "step": "chatbot answers", "model": a["model"].iloc[0], "calls": len(a),
                     "input_tokens": int(a["prompt_tokens"].sum()), "output_tokens": int(a["output_tokens"].sum()),
                     "cost_usd": round(float(a["cost_usd"].sum()), 4), "max_input_tokens": int(a["prompt_tokens"].max()),
                     "served_by_primary_key": round(sum(c.get("primary", 0) for c in served) / total_calls, 4) if total_calls else None,
                     "basis": "answers tables (every paid run); includes the case summaries of the hierarchical mode"})
    seen, calls = set(), []
    manifests = sorted(store.a5_dir().rglob("*.manifest.json")) + sorted(store.results_dir().glob("72_a5_settings*.manifest.json"))
    for path in manifests:
        setting = path.parent.name if path.parent.parent == store.a5_dir() else "all"
        for c in json.loads(path.read_text(encoding="utf-8")).get("calls", []):
            if c.get("cached") is False and c.get("cache_entry") not in seen:
                seen.add(c.get("cache_entry"))
                calls.append({**c, "setting": setting})
    table = pd.DataFrame(calls)
    if len(table):
        for (setting, step, model), sub in table.groupby(["setting", "step", "model"], sort=True):
            rows.append({"setting": setting, "step": step, "model": model, "calls": len(sub),
                         "input_tokens": int(sub["input_tokens"].fillna(0).sum()), "output_tokens": int(sub["output_tokens"].fillna(0).sum()),
                         "cost_usd": round(float(sub["cost_usd"].fillna(0).sum()), 4),
                         "max_input_tokens": int(sub["input_tokens"].fillna(0).max()),
                         "served_by_primary_key": round(float((sub["credentials"] == "primary").mean()), 4),
                         "basis": "run manifests: calls not served from the cache; output tokens include reasoning"})
    out = pd.DataFrame(rows)
    out.insert(0, "phase", "7")
    judge_rows = out[out["step"] != "chatbot answers"]
    total = {"phase": "7", "setting": "all", "step": "TOTAL", "cost_usd": round(float(out["cost_usd"].sum()), 4),
             "max_input_tokens": int(judge_rows["max_input_tokens"].max()) if len(judge_rows) else None,
             "basis": f"plan line for Phase 7: {cfg5['budget']['plan_line_usd']} USD; k = 10 reuses Phase 6's answers (counted there); "
                      "max_input_tokens here is the largest judge prompt"}
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


# ---------------------------------------------------------------------------- figure

def _axes_style(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure(table: pd.DataFrame, cfg5: dict, path: Path, completeness: bool) -> None:
    """Five panels against k: supported claims and unsupported claims per answer after the control,
    coverage, latency, cost.

    The full-text settings are joined by a line; the hierarchical summary at k = 30 is an open marker.
    """
    full = [n for n, s in cfg5["settings"].items() if s["context_mode"] == "full_text"]
    summary = [n for n, s in cfg5["settings"].items() if s["context_mode"] != "full_text"]
    ks = {n: s["top_k"] for n, s in cfg5["settings"].items()}

    def value(setting, section, group, metric, rater):
        hit = table[(table["setting"] == setting) & (table["section"] == section) & (table["group"] == group)
                    & (table["metric"] == metric) & (table["rater"] == rater)]
        return (float(hit["value"].iloc[0]), hit["ci_low"].iloc[0], hit["ci_high"].iloc[0]) if len(hit) else (np.nan, np.nan, np.nan)

    panels = [("Supported claims after the control", "faithfulness", "all", "faithfulness_corrected_ci", list(JUDGE_STYLE)),
              ("Unsupported claims per open answer", "faithfulness", "all", "unsupported_claims_per_answer", list(JUDGE_STYLE)),
              *([("Coverage of the reference points", "completeness", "all", "coverage", list(JUDGE_STYLE))] if completeness else []),
              ("Median latency of an open answer (s)", "system", "open", "latency_s_median", [""]),
              ("Mean cost of an open answer (USD)", "system", "open", "cost_usd_per_answer", [""])]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for ax, (title, section, group, metric, raters) in zip(axes, panels):
        for rater in raters:
            label, colour, marker = JUDGE_STYLE.get(rater, ("JurisMap v2", SYSTEM_COLOUR, "D"))
            pts = [(ks[n], *value(n, section, group, metric, rater)) for n in full]
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(x, y, color=colour, linewidth=2, marker=marker, markersize=7, markeredgecolor="white",
                    markeredgewidth=1.0, label=label, zorder=3)
            lows = [p[2] for p in pts]
            if all(v is not None and v == v for v in lows):
                ax.fill_between(x, lows, [p[3] for p in pts], color=colour, alpha=0.14, linewidth=0)
            for n in summary:
                v = value(n, section, group, metric, rater)[0]
                ax.plot([ks[n]], [v], linestyle="none", marker=marker, markersize=8, markerfacecolor="white",
                        markeredgecolor=colour, markeredgewidth=1.6, zorder=4)
        _axes_style(ax)
        ax.set_xticks(sorted(set(ks.values())))
        ax.set_xlabel("k (decisions in the context)", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        if metric in {"faithfulness_corrected_ci", "coverage"}:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if metric in {"unsupported_claims_per_answer", "latency_s_median", "cost_usd_per_answer"}:
            ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor="white",
                                           markeredgecolor=MUTED, markeredgewidth=1.6))
    labels.append("Hierarchical summary, k = 30 (open marker)")
    handles.append(matplotlib.lines.Line2D([], [], linestyle="-", marker="D", color=SYSTEM_COLOUR, linewidth=2, markersize=6))
    labels.append("JurisMap v2 (latency, cost)")
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, frameon=False,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.07))
    fig.text(0.01, -0.03, "Shaded: 95% interval over resampled questions (open questions: 11 at k = 5, 12 otherwise). Hierarchical "
                          "summary: each case summarised to about 300 tokens before the chat. Each panel has its own scale.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--costs-only", action="store_true", help="write only results/costs_phase7.csv (interim, e.g. after the pilot)")
    args = ap.parse_args()
    cfg5 = load_config("a5_context_size")
    results = store.results_dir()
    if args.costs_only:
        with RunManifest(f"{SCRIPT}_costs", results, params={"costs_only": True}) as run:
            costs = cost_rows(cfg5)
            costs_path = new_artefact_path(results / "costs_phase7.csv")
            costs.to_csv(costs_path, index=False)
            run.add_output(costs_path, "Phase 7: calls, tokens and cost per step and model (interim)")
        print(costs.drop(columns=["basis", "phase"]).to_string(index=False))
        return 0
    R = phase6_report()
    resamples, seed = load_config("a1_methods")["bootstrap_resamples"], load_config("seeds")["bootstrap"]
    verifiers, judges = cfg5["judges"]["verifiers"], cfg5["judges"]["completeness"]
    completeness = cfg5.get("report", {}).get("completeness", True)
    with RunManifest(SCRIPT, results, params={"a5": cfg5, "bootstrap_resamples": resamples}, seeds={"bootstrap": seed}) as run:
        rows, values = [], {}
        for name, setting in cfg5["settings"].items():
            t = setting_tables(name)
            claims = t["claims"]
            part = R.deterministic_rows(t["deterministic"])
            faith_rows, _ = R.faithfulness_rows(claims, verifiers, [], resamples, seed)
            cov_rows, per_answer_cov = R.completeness_rows(t["completeness"], judges, [], resamples, seed)
            part += faith_rows + controlled_rows(R, claims, verifiers, resamples, seed) + (cov_rows if completeness else [])
            part += R.system_rows(t["answers"]) + extra_system_rows(R, t["answers"])
            n_open = int((t["answers"]["type"] == "open").sum())
            part.append(R.row("faithfulness", "all", "claims_per_answer", "", n_open, len(claims) / n_open if n_open else None))
            rows += [{"setting": name, "k": setting["top_k"], "context_mode": setting["context_mode"], **r} for r in part]
            values[name] = per_answer_values(claims, per_answer_cov, verifiers, judges if completeness else [])
        table = pd.DataFrame(rows)
        main_path = new_artefact_path(results / "a5_context_size.csv")
        table.to_csv(main_path, index=False)
        run.add_output(main_path, "A5: results of the context-size test, one block per setting")

        paired = pd.DataFrame(paired_rows(values, resamples, seed))
        paired_path = new_artefact_path(results / "a5_context_size_paired.csv")
        paired.to_csv(paired_path, index=False)
        run.add_output(paired_path, "A5: differences from k = 10 (and of the summary from k = 30), paired by question")

        figures = results / "figures"
        figures.mkdir(exist_ok=True)
        figure_path = new_artefact_path(figures / "a5_context_size.pdf")
        figure(table, cfg5, figure_path, completeness)
        run.add_output(figure_path, "A5: faithfulness, latency and cost against k")

        costs = cost_rows(cfg5)
        costs_path = new_artefact_path(results / "costs_phase7.csv")
        costs.to_csv(costs_path, index=False)
        run.add_output(costs_path, "Phase 7: calls, tokens and cost per step and model")
        run.data["summary"] = {"total_usd": float(costs.iloc[-1]["cost_usd"])}

    headline = table[table["group"].isin(["all", "open"]) & table["metric"].isin(
        ["accuracy", "faithfulness", "faithfulness_corrected", "contradiction_rate", "answers_with_a_contradicted_claim",
         "coverage", "claims_per_answer", "latency_s_median", "cost_usd_per_answer", "prompt_tokens_median",
         "answers_not_finished", "process_numbers_unknown", "miss_rate"])]
    print(headline.drop(columns=["section", "note", "context_mode"]).to_string(index=False))
    print(paired.drop(columns=["note"]).to_string(index=False))
    print(costs.drop(columns=["basis", "phase"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
