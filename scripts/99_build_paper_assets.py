"""Phase 11: the paper's tables and figures, rebuilt from the files of results/.

Every table and figure the paper may use is written under one consistent name (`tab_a1_retrieval.csv`,
`fig_a2_trustworthiness.pdf`, ...) into the paper folder (config/paper_assets.yaml `output`), from the source
files pinned in the same config. Nothing is recomputed that an analysis script computes: the tables select
and format, the figures draw. The script makes no model call and reads no data/ file except the B2 workbook.

Tables are CSV with the values formatted as the paper prints them (intervals "a to b", p-values "< 0.001").
Figures are PDF and PNG with no title (captions belong to the manuscript); their notes are in ASSETS.csv.

Outputs (replaced on every run: these are the paper's working copies; the record of each run is its manifest):
  <paper>/ISWA/tables/tab_*.csv        the tables
  <paper>/ISWA/figures/fig_*.{pdf,png} the figures (and the two agent graphs of Phase 2 as SVG)
  <paper>/ISWA/tables/ASSETS.csv       one row per asset: file, caption note, source files with their hash,
                                        the script and run manifest that wrote each source
  results/99_build_paper_assets.manifest.json

Usage:
  uv run python scripts/99_build_paper_assets.py [--out-dir DIR] [--only tab_a1_retrieval fig_a5_context_size]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

from jm_eval import paper as P  # noqa: E402
from jm_eval.io import load_config, resolve, sha256_file  # noqa: E402
from jm_eval.manifest import RunManifest  # noqa: E402

SCRIPT = "99_build_paper_assets"

# palette of the phase figures (scripts/46, 73): version 2 blue, version 1 orange
V2, V1, NEUTRAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#6b6a66", "#0b0b0b", "#52514e", "#e4e3df"
JUDGE = {"judge_a": V2, "judge_b": V1}
MODES = ["full", "single_prompt", "no_legal_search", "no_precedents", "no_template"]
METHOD_ORDER = ["M4", "M5", "M4c", "M4b", "M3", "M1", "M2", "M6", "M7"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42})

TABLES: dict[str, tuple] = {}
FIGURES: dict[str, tuple] = {}


def table(name: str, note: str):
    def wrap(fn):
        TABLES[name] = (fn, note)
        return fn
    return wrap


def figure(name: str, note: str):
    def wrap(fn):
        FIGURES[name] = (fn, note)
        return fn
    return wrap


# =========================================================================================== tables

# ---------------------------------------------------------------------------- versions, B2, B3

@table("tab_versions", "Version 1 (expert-evaluated) and version 2 (technically evaluated): the settings that differ "
                       "(paper scope 'version difference' of the reproducibility table; ground rule 9).")
def tab_versions(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("reproducibility", a)
    t = t[t["paper_scope"] == "version difference"]
    return t[["section", "setting", "v1", "v2"]].rename(columns={"section": "Component", "setting": "Setting",
                                                                 "v1": "Version 1", "v2": "Version 2"})


@table("tab_b2_constructs", "B2: construct scores of the five experts (1-5 scale), with the bootstrap 95% interval over "
                            "respondents and the exact one-sided Wilcoxon p against the neutral point 3 (smallest attainable "
                            "p with N = 5: 0.031).")
def tab_b2_constructs(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.excel("b2_workbook", "Constructs", a)
    return pd.DataFrame({
        "Construct": t["Construct"], "Items": t["Items"],
        **{e: t[e].map(lambda x: P.f(x, 2)) for e in ["E1", "E2", "E3", "E4", "E5"]},
        "Mean": t["Mean"].map(lambda x: P.f(x, 2)), "Median": t["Median"].map(lambda x: P.f(x, 2)),
        "Bootstrap 95% CI": [P.interval(lo, hi, 2) for lo, hi in zip(t["Boot_CI95_low"], t["Boot_CI95_high"])],
        "Wilcoxon p (> 3)": t["Wilcoxon_exact_p_one_sided"].map(lambda x: P.f(x, 3)),
        "r_wg(j)": t["rwg_j"].map(lambda x: P.f(x, 2))})


@table("tab_b3_themes", "B3: themes of the experts' open answers (theme x expert; 1 = present), coded by the author "
                        "with the approved codebook.")
def tab_b3_themes(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("b3_matrix", a)
    return t.rename(columns={"theme": "Theme", "label": "Description", "n_experts": "Experts (of 5)"})


# ---------------------------------------------------------------------------- A1

@table("tab_a1_labels", "A1 silver labels: The author's audit of 60 decisions and a review of 100 decisions by a second "
                        "model (GPT-5.4; 20 labels with planted errors, all caught). Wilson 95% intervals.")
def tab_a1_labels(S: P.Sources, a: str) -> pd.DataFrame:
    audit, review = S.json("a1_label_audit", a), S.json("a1_label_review", a)
    rows = []
    for key, label in [("legal_basis", "Legal basis (R-base)"), ("penalty_type", "Penalty type"),
                       ("infringed_provision", "Infringed provision (R-prov)"), ("infraction_category", "Infraction category (R-cat)")]:
        au, rv = audit[key], review.get(key)
        rows.append({"Label": label,
                     "Author's audit": P.share(au["correct"], au["judged"]),
                     "Audit accuracy": P.pct(au["accuracy"]),
                     "Audit Wilson 95%": P.interval(*au["wilson_95"]),
                     "Second-model review": P.share(rv["correct"], rv["n"]) if rv else "",
                     "Review share": P.pct(rv["share"]) if rv else "",
                     "Review Wilson 95%": P.interval(*rv["wilson_95"]) if rv else ""})
    cat = review["category"]
    for pair, label in [("reviewer vs gemini", "Category: GPT-5.4 vs Gemini labels"),
                        ("gemini vs mistral (corpus)", "Category: Gemini vs Mistral labels (whole corpus)"),
                        ("gemini vs hugo", "Category: Gemini labels vs the author")]:
        c = cat[pair]
        rows.append({"Label": label, "Second-model review": f"{P.pct(c['agreement'])} agreement, kappa {P.f(c['kappa'], 2)}",
                     "Review share": f"n = {c['n']:,}"})
    return pd.DataFrame(rows).fillna("")


def _a1_methods(S: P.Sources) -> dict:
    return S.cfg["labels"]["methods"]


@table("tab_a1_retrieval", "A1, comparison set (1,503 decisions in both versions), relevance R-base (identical set of "
                           "specific legal provisions; 824 queries). nDCG@10 with its bootstrap 95% interval over queries.")
def tab_a1_retrieval(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a1_main", a)
    t = t[(t["scope"] == "comparison_set") & (t["relevance"] == "R-base")]
    rows = []
    for m in METHOD_ORDER:
        r = t[t["method"] == m].set_index("metric")
        rows.append({"Method": _a1_methods(S)[m], "Queries": int(r.loc["nDCG@10", "queries"]),
                     **{k: P.f(r.loc[k, "mean"]) for k in ["P@5", "P@10", "R@5", "R@10", "MAP", "MRR", "nDCG@10"]},
                     "nDCG@10 95% CI": P.interval(r.loc["nDCG@10", "ci_low"], r.loc["nDCG@10", "ci_high"]),
                     "mAR": P.f(r.loc["mAR", "mean"])})
    return pd.DataFrame(rows)


@table("tab_a1_relevance", "A1: nDCG@10 under every relevance definition, on the comparison set (1,503 decisions) and the "
                           "v2 corpus (2,390). R-base is primary; R-cat uses the Gemini category labels, R-cat-mistral the "
                           "blind second labelling by Mistral Medium 3.")
def tab_a1_relevance(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a1_main", a)
    t = t[t["metric"] == "nDCG@10"]
    rels = ["R-base", "R-base-relaxed", "R-prov", "R-cat", "R-cat-mistral", "R-cat-multi"]
    rows = []
    for scope, scope_label in [("comparison_set", "Comparison set"), ("v2", "v2 corpus")]:
        for m in METHOD_ORDER:
            r = t[(t["scope"] == scope) & (t["method"] == m)].set_index("relevance")
            rows.append({"Scope": scope_label, "Method": _a1_methods(S)[m], **{rel: P.f(r.loc[rel, "mean"]) for rel in rels}})
    return pd.DataFrame(rows)


def p_text(p) -> str:
    s = P.pval(p)
    return f"p {s}" if s.startswith("<") else f"p = {s}"


def _contrast(r) -> str:
    return f"{P.signed(r['diff'])} ({P.interval(r['ci_low'], r['ci_high'])}), {p_text(r['p_holm'])}"


@table("tab_a1_contrasts", "A1 planned contrasts: difference of means with its bootstrap 95% interval; p is the paired "
                           "randomisation test, Holm-adjusted over the four contrasts.")
def tab_a1_contrasts(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a1_contrasts", a)
    cols = [("R-base", "nDCG@10"), ("R-base", "R@5"), ("R-base", "mAR"), ("R-cat", "nDCG@10")]
    rows = []
    for contrast in ["M4 - M3", "M4 - M5", "M4 - M4c", "M4c - M4b"]:
        for scope, scope_label in [("comparison_set", "Comparison set"), ("v2", "v2 corpus")]:
            sub = t[(t["contrast"] == contrast) & (t["scope"] == scope)]
            row = {"Contrast": sub["label"].iloc[0], "Scope": scope_label}
            for rel, metric in cols:
                hit = sub[(sub["relevance"] == rel) & (sub["metric"] == metric)]
                row[f"{rel} {metric}"] = _contrast(hit.iloc[0]) if len(hit) else ""
            rows.append(row)
    return pd.DataFrame(rows)


@table("tab_a1_complexity", "A1, R-base nDCG@10, queries with one specific provision and with several: difference of means "
                            "with its bootstrap 95% interval and Holm-adjusted p.")
def tab_a1_complexity(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a1_complexity_contrasts", a)
    t = t[(t["relevance"] == "R-base") & (t["metric"] == "nDCG@10")]
    rows = []
    for contrast in ["M4 - M3", "M4 - M5"]:
        for scope, scope_label in [("comparison_set", "Comparison set"), ("v2", "v2 corpus")]:
            sub = t[(t["contrast"] == contrast) & (t["scope"] == scope)].set_index("infractions")
            rows.append({"Contrast": sub["label"].iloc[0], "Scope": scope_label,
                         "One provision": _contrast(sub.loc["one"]), "Queries (one)": int(sub.loc["one", "queries"]),
                         "Several provisions": _contrast(sub.loc["several"]), "Queries (several)": int(sub.loc["several", "queries"])})
    return pd.DataFrame(rows)


@table("tab_a1_heldout", "A1 check of the choice of the retrieval vector (Phase 4.1): nDCG@10 on the held-out half of the "
                         "queries, which took no part in the choice (split by process). Fused 0.5 is version 2's vector.")
def tab_a1_heldout(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a1_candidates", a)
    t = t[(t["split"] == "held_out") & (t["metric"] == "nDCG@10")]
    names = {"compressed": "Compression alone (v2 until v2.2)", "full": "Full report alone", "full_chunked": "Full report in chunks",
             "fused_25": "Fused, 0.25 full", "fused_50": "Fused, 0.5 full (v2)", "fused_75": "Fused, 0.75 full",
             "M3": _a1_methods(S)["M3"], "M2": "BM25", "M1": "TF-IDF"}
    rels = ["R-base", "R-base-relaxed", "R-prov", "R-cat", "R-cat-mistral", "R-cat-multi"]
    rows = []
    for scope, scope_label in [("comparison_set", "Comparison set"), ("v2", "v2 corpus")]:
        for m, label in names.items():
            r = t[(t["scope"] == scope) & (t["method"] == m)].set_index("relevance")
            rows.append({"Scope": scope_label, "Vector": label,
                         "Queries (R-base / R-cat)": f"{int(r.loc['R-base', 'queries'])} / {int(r.loc['R-cat', 'queries'])}",
                         **{rel: P.f(r.loc[rel, "mean"]) if rel in r.index else "" for rel in rels}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- A2

@table("tab_a2_map", "A2: the served map of version 2 against its own vectors (v2 corpus, 2,390 decisions), t-SNE on the "
                     "same vectors, and each version's map on the comparison set (version 1: mean of five seeds of its "
                     "setting). Same category / legal basis: share of a decision's 10 nearest neighbours with the same label.")
def tab_a2_map(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a2_metrics", a)
    idx = t.set_index(["set", "projection", "metric"])["mean"]
    cols = {"v2 served map (2,390)": ("v2_full", "frozen_v2"), "t-SNE, v2 vectors (2,390)": ("v2_full", "tsne_p30"),
            "Embedding space, v2 (2,390)": ("v2_full", "original"),
            "v1 map (1,503)": ("v1_comparison", "umap_n15_d0.1"), "v2 served map (1,503)": ("v2_comparison", "frozen_v2"),
            "Embedding space, v1 (1,503)": ("v1_comparison", "original")}
    metrics = [*(f"trustworthiness@{k}" for k in (5, 10, 20, 50)), *(f"continuity@{k}" for k in (5, 10, 20, 50)),
               "knn_jaccard@10", "distance_spearman", "same_label@10/R-cat", "same_label@10/R-base"]
    names = {"knn_jaccard@10": "kNN Jaccard@10", "distance_spearman": "Distance Spearman",
             "same_label@10/R-cat": "Same category among 10 nearest", "same_label@10/R-base": "Same legal basis among 10 nearest"}
    rows = []
    for metric in metrics:
        row = {"Metric": names.get(metric, metric.replace("@", " at k = ").capitalize())}
        for label, (st, proj) in cols.items():
            v = idx.get((st, proj, metric))
            row[label] = P.f(v)
        rows.append(row)
    for label_metric, name in [("same_label_chance@10/R-cat", "Same category by chance"), ("same_label_chance@10/R-base", "Same legal basis by chance")]:
        rows.append({"Metric": name, "Embedding space, v2 (2,390)": P.f(idx.get(("v2_full", "original", label_metric))),
                     "Embedding space, v1 (1,503)": P.f(idx.get(("v1_comparison", "original", label_metric)))})
    return pd.DataFrame(rows).fillna("")


# ---------------------------------------------------------------------------- A3

@table("tab_a3_strategies", "A3 strategy selection (Phase 6.1) on the development sets: GPT-5.4 and Mistral Medium 3 as "
                            "judges; the rule (coverage at most 0.10 below the baseline, no loss of templated answers or "
                            "abstentions; then the lowest unsupported rate) was fixed before any strategy was scored.")
def tab_a3_strategies(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a3_strategies", a)
    return pd.DataFrame({
        "Strategy": t["candidate"], "Description": t["label"],
        "Unsupported rate": t["unsupported_rate"].map(P.f),
        "Difference from baseline (95% CI)": [("" if P.is_missing(d) else f"{P.signed(d)} ({P.interval(lo, hi)})")
                                               for d, lo, hi in zip(t["unsupported_rate_difference_from_baseline"],
                                                                    t["unsupported_rate_ci_low_from_baseline"],
                                                                    t["unsupported_rate_ci_high_from_baseline"])],
        "Unsupported claims per answer": t["unsupported_claims_per_answer"].map(lambda x: P.f(x, 1)),
        "Contradiction rate": t["contradiction_rate"].map(P.f),
        "Coverage": t["coverage"].map(P.f),
        "Templated correct": [P.share(c, n) for c, n in zip(t["templated_correct"], t["templated"])],
        "Abstentions": [P.share(c, n) for c, n in zip(t["abstained"], t["unanswerable"])],
        "Median latency, open answer (s)": t["latency_s_median_open"].map(lambda x: P.f(x, 1)),
        "Cost per open answer": t["cost_usd_per_open_answer"].map(P.usd),
        "Eligible": t["eligible"].map({True: "yes", False: "no"}), "Selected": t["selected"].map({True: "yes", False: ""})})


def _a3(t: pd.DataFrame, section: str, group: str, metric: str, rater) -> pd.Series | None:
    m = (t["section"] == section) & (t["group"] == group) & (t["metric"] == metric)
    m &= t["rater"].isna() if rater is None else (t["rater"] == rater)
    hit = t[m]
    return hit.iloc[0] if len(hit) else None


@table("tab_a3_chatbot", "A3 on the evaluated set (version 2 as frozen, 6 case sets of 10 decisions; 66 answers, 36 "
                         "open): checks scored by code, and claim-level faithfulness of the open answers by each judge. "
                         "Supported after the control: claims left 'not found' are re-judged against the full decision on a "
                         "sample. Gemini 3.8 Flash (same family as the system) is reported separately, on the first runs only.")
def tab_a3_chatbot(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a3_chatbot", a)
    rows = []

    def code(label, section, group, metric, fmt):
        r = _a3(t, section, group, metric, "code")
        rows.append({"Measure": label, "Code": fmt(r)})

    code("Templated questions correct", "templated", "all", "accuracy", lambda r: f"{r['note'].split(' correct')[0]} (Wilson {P.interval(r['ci_low'], r['ci_high'], 2)})")
    code("Unanswerable questions declined", "unanswerable", "all", "abstention_rate", lambda r: r["note"].split(";")[0].replace(" abstained", ""))
    code("Process numbers cited in the answers", "citations", "all", "process_numbers_cited", lambda r: f"{int(r['value'])}")
    code("... of decisions not supplied and not in the question", "citations", "all", "process_numbers_unknown", lambda r: f"{int(r['value'])}")
    judges = ["judge_a", "judge_b", "judge_gemini"]
    names = S.cfg["labels"]["judges"]
    for label, metric, fmt in [
            ("Claims judged", "faithfulness", lambda r: f"{int(r['n'])}"),
            ("Supported", "faithfulness", lambda r: f"{P.f(r['value'])} ({P.interval(r['ci_low'], r['ci_high'])})"),
            ("Supported after the control", "faithfulness_corrected", lambda r: P.f(r["value"])),
            ("Contradicted (judge's label)", "contradiction_rate", lambda r: P.f(r["value"])),
            ("Answers with a contradicted claim", "answers_with_a_contradicted_claim", lambda r: P.share(round(r["value"] * r["n"]), r["n"]))]:
        row = {"Measure": label}
        for j in judges:
            r = _a3(t, "faithfulness", "all", metric, j)
            row[names[j]] = fmt(r) if r is not None else ""
        rows.append(row)
    r = _a3(t, "miss_control", "all", "miss_rate", "judge_a (full text)")
    rows.append({"Measure": "Retrieval-miss control: re-judged claims supported by the full decision",
                 names["judge_a"]: f"{P.f(r['value'])} ({r['note'].split(' re-judged')[0]})"})
    row = {"Measure": "Coverage of the reference points (95% CI)"}
    for j in judges:
        r = _a3(t, "completeness", "all", "coverage", j)
        row[names[j]] = f"{P.f(r['value'])} ({P.interval(r['ci_low'], r['ci_high'])})"
    rows.append(row)
    for label, group, metric, fmt in [("Median latency, open answer (s)", "open", "latency_s_median", lambda v: P.f(v, 1)),
                                      ("Cost per open answer", "open", "cost_usd_per_answer", P.usd),
                                      ("Median prompt, tokens", "all", "prompt_tokens_median", lambda v: f"{v:,.0f}")]:
        rows.append({"Measure": label, "Code": fmt(_a3(t, "system", group, metric, None)["value"])})
    return pd.DataFrame(rows).fillna("")


@table("tab_a3_judges", "A3 judges validated on their own task: The author's blind labels of 50 claims (supported against "
                        "not supported; 'weighted' weights each pattern of the judges' labels by its share of all 1,463 "
                        "claims), and planted control claims of known truth (55 true, 51 false).")
def tab_a3_judges(S: P.Sources, a: str) -> pd.DataFrame:
    audit = S.csv("a3_audit", a).set_index("rater")
    controls = S.csv("a3_controls", a)
    controls = controls[controls["controls"] == "all"].set_index("rater")
    names = S.cfg["labels"]["judges"]
    rows = []
    for rater in ["judge_a", "judge_b", "judge_gemini", "nli"]:
        au = audit.loc[rater] if rater in audit.index else None
        c = controls.loc[rater] if rater in controls.index else None
        rows.append({
            "Verifier": names[rater],
            "Audited claims": int(au["claims"]) if au is not None else "",
            "Agrees with the author": f"{P.share(au['same_supported_or_not'], au['claims'])} ({P.pct(au['same_supported_or_not_share'], 0)}; {P.interval(au['ci_low'], au['ci_high'], 2)})" if au is not None else "",
            "Weighted to all claims": P.f(au["weighted_to_all_claims"], 2) if au is not None else "",
            "Rejects claims the author supports": P.share(au["supported_by_hugo_rejected_by_rater"], au["supported_by_hugo"]) if au is not None else "",
            "Accepts claims the author does not": P.share(au["not_supported_by_hugo_accepted_by_rater"], au["not_supported_by_hugo"]) if au is not None else "",
            "True controls accepted": f"{P.share(c['true_accepted'], c['true_controls'])} ({c['true_accepted_ci']})" if c is not None else "",
            "False controls rejected": f"{P.share(c['false_rejected'], c['false_controls'])} ({c['false_rejected_ci']})" if c is not None else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- A5

@table("tab_a5_context", "A5: the chat with k decisions in its context (version 2; the Phase 6 questions, one answer each). "
                         "k = 10 is production's setting; 'summary' gives each of 30 decisions as a short summary. "
                         "Unsupported claims: not labelled supported by GPT-5.4 after the retrieval-miss control.")
def tab_a5_context(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a5_context", a)
    settings = [("k5", "k = 5"), ("k10", "k = 10"), ("k20", "k = 20"), ("k30", "k = 30"), ("h30", "Summary, k = 30")]

    def get(setting, section, group, metric, rater):
        m = (t["setting"] == setting) & (t["section"] == section) & (t["group"] == group) & (t["metric"] == metric)
        m &= t["rater"].isna() if rater is None else (t["rater"] == rater)
        hit = t[m]
        return hit.iloc[0] if len(hit) else None

    spec = [
        ("Templated correct", ("templated", "all", "accuracy", "code"), lambda r: r["note"].replace(" correct", "")),
        ("Open answers", ("faithfulness", "all", "unsupported_claims_per_answer", "judge_a"), lambda r: f"{int(r['n'])}"),
        ("Claims per open answer", ("faithfulness", "all", "claims_per_answer", None), lambda r: P.f(r["value"], 0)),
        ("Supported after the control, GPT-5.4 (95% CI)", ("faithfulness", "all", "faithfulness_corrected_ci", "judge_a"),
         lambda r: f"{P.f(r['value'])} ({P.interval(r['ci_low'], r['ci_high'], 2)})"),
        ("Supported after the control, Mistral Medium 3", ("faithfulness", "all", "faithfulness_corrected", "judge_b"), lambda r: P.f(r["value"])),
        ("Unsupported claims per open answer, GPT-5.4", ("faithfulness", "all", "unsupported_claims_per_answer", "judge_a"), lambda r: P.f(r["value"], 1)),
        ("Unsupported claims per open answer, Mistral Medium 3", ("faithfulness", "all", "unsupported_claims_per_answer", "judge_b"), lambda r: P.f(r["value"], 1)),
        ("Contradicted, GPT-5.4 (judge's label)", ("faithfulness", "all", "contradiction_rate", "judge_a"), lambda r: P.f(r["value"])),
        ("Median prompt, tokens", ("system", "open", "prompt_tokens_median", None), lambda r: f"{r['value']:,.0f}"),
        ("Median latency, open answer (s)", ("system", "open", "latency_s_median", None), lambda r: P.f(r["value"], 0)),
        ("Cost per open answer", ("system", "open", "cost_usd_per_answer", None), lambda r: P.usd(r["value"])),
        ("Process numbers cited / unknown", ("citations", "all", "process_numbers_cited", "code"), None),
    ]
    rows = []
    for label, key, fmt in spec:
        row = {"Measure": label}
        for s, s_label in settings:
            if fmt is None:
                cited, unknown = get(s, *key), get(s, "citations", "all", "process_numbers_unknown", "code")
                row[s_label] = f"{int(cited['value'])} / {int(unknown['value'])}"
            else:
                r = get(s, *key)
                row[s_label] = fmt(r) if r is not None else ""
        rows.append(row)
    paired = S.csv("a5_paired", a)
    for rater, name in [("judge_a", "GPT-5.4"), ("judge_b", "Mistral Medium 3")]:
        row = {"Measure": f"Paired difference from k = 10, supported after the control, {name} (95% CI)"}
        for s, s_label in settings:
            hit = paired[(paired["setting"] == s) & (paired["rater"] == rater)]
            if len(hit):
                h = hit.iloc[0]
                row[s_label] = f"{P.signed(h['difference'])} ({P.interval(h['ci_low'], h['ci_high'], 2)}) vs {h['reference']}"
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


# ---------------------------------------------------------------------------- A4

@table("tab_a4_versions", "A4: version 2 before (v2.6) and after (v2.7) the fact controls, on the same 15 test notices "
                          "and precedents. Written drafts (lost drafts and the one draft citing its own source left out); "
                          "means over notices; Wilcoxon signed-rank test over the 15 notices (9 for the accepted argument).")
def tab_a4_versions(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a4_versions", a)
    digits = {"estimated_invented": 1, "cost_usd": 3, "recall": 2, "accepted_raised": 2, "template_coverage": 2}
    return pd.DataFrame({
        "Measure": t["label"], "Notices": t["notices"],
        "v2.6": [P.usd(v) if m == "cost_usd" else P.f(v, digits.get(m, 2)) for m, v in zip(t["metric"], t["mean_a"])],
        "v2.7": [P.usd(v) if m == "cost_usd" else P.f(v, digits.get(m, 2)) for m, v in zip(t["metric"], t["mean_b"])],
        "Notices higher / lower in v2.7": [f"{h} / {lo}" for h, lo in zip(t["notices_higher_in_b"], t["notices_lower_in_b"])],
        "Wilcoxon p": t["wilcoxon_p"].map(P.pval)})


@table("tab_a4_invented", "A4: estimated invented case facts per draft, by mode: GPT-5.4's classified passages times the "
                          "share of each class the author found invented in a blind audit of 50 passages per version "
                          "(range from the Wilson bounds of that share); and the share of drafts that claim the record of "
                          "past offences was missing from the notice or added later (none of the 15 real appeals did).")
def tab_a4_invented(S: P.Sources, a: str) -> pd.DataFrame:
    v6, v7 = S.csv("a4_v26_invented", a).set_index("mode"), S.csv("a4_v27_invented", a).set_index("mode")
    modes = S.cfg["labels"]["modes"]
    return pd.DataFrame([{
        "Mode": modes[m],
        "v2.6 invented per draft": f"{P.f(v6.loc[m, 'estimated_invented'], 2)} ({P.interval(v6.loc[m, 'low'], v6.loc[m, 'high'], 2)})",
        "v2.7 invented per draft": f"{P.f(v7.loc[m, 'estimated_invented'], 2)} ({P.interval(v7.loc[m, 'low'], v7.loc[m, 'high'], 2)})",
        "v2.6 past-offences claim": P.pct(v6.loc[m, "past_offences_claim_share"], 0),
        "v2.7 past-offences claim": P.pct(v7.loc[m, "past_offences_claim_share"], 0)} for m in MODES])


ABLATION_METRICS = {"judge_a_legal_accuracy": "Legal accuracy", "judge_a_fidelity_to_facts": "Fidelity to facts",
                    "judge_a_use_of_precedents": "Use of precedents", "judge_a_structure": "Structure",
                    "judge_a_persuasiveness": "Persuasiveness", "judge_a_rubric_mean": "Rubric mean",
                    "recall": "Argument recall", "accepted_raised": "Accepted argument raised",
                    "template_coverage": "Template coverage"}


@table("tab_a4_ablation", "A4 ablation (rubric by GPT-5.4, 1-5): means over the 15 notices per mode, lost drafts scored as "
                          "the worst outcome; Friedman test over notices; * different from the full pipeline after Holm "
                          "correction (Wilcoxon). With 15 notices a difference that is not significant is inconclusive.")
def tab_a4_ablation(S: P.Sources, a: str) -> pd.DataFrame:
    rows = []
    modes = S.cfg["labels"]["modes"]
    for version, src in [("v2.6", "a4_v26_ablation"), ("v2.7", "a4_v27_ablation")]:
        t = S.csv(src, a)
        t = t[t["analysis"] == "main"].set_index("metric")
        for metric, label in ABLATION_METRICS.items():
            r = t.loc[metric]
            digits = 2
            row = {"Version": version, "Measure": label, "Notices": int(r["notices"])}
            for m in MODES:
                star = "" if m == "full" else ("*" if r[f"holm_p_{m}"] < 0.05 else "")
                row[modes[m]] = P.f(r[f"mean_{m}"], digits) + star
            row["Friedman p"] = P.pval(r["friedman_p"])
            rows.append(row)
    return pd.DataFrame(rows)


@table("tab_a4_judges", "A4 rubric judges: mean score per criterion and ordinal Krippendorff's alpha between GPT-5.4 and "
                        "Mistral Medium 3. The judges agree on the order of drafts more than on levels; the paper reports "
                        "GPT-5.4's levels.")
def tab_a4_judges(S: P.Sources, a: str) -> pd.DataFrame:
    rows = []
    for version, src in [("v2.6", "a4_v26_agreement"), ("v2.7", "a4_v27_agreement")]:
        t = S.csv(src, a)
        for _, r in t.iterrows():
            rows.append({"Version": version, "Criterion": r["criterion"].replace("_", " ").capitalize(), "Drafts": int(r["drafts"]),
                         "GPT-5.4": P.f(r["mean_gpt"], 2), "Mistral Medium 3": P.f(r["mean_mistral"], 2),
                         "Gemini 3.8 Flash": P.f(r["mean_gemini"], 2), "Alpha (GPT-5.4, Mistral)": P.f(r["alpha_ordinal"], 2)})
    return pd.DataFrame(rows)


@table("tab_a4_checks", "A4 checks by code, per version (225 drafts each: 15 notices x 5 modes x 3 runs): drafts lost (all "
                        "to recitation blocks of the model), citations of precedents and of statute articles.")
def tab_a4_checks(S: P.Sources, a: str) -> pd.DataFrame:
    rows = []
    for version, src in [("v2.6", "a4_v26_scores"), ("v2.7", "a4_v27_scores")]:
        t = S.csv(src, a)
        w = t[~t["lost"].astype(bool)]
        graded = w[["statute_found", "statute_in_range", "statute_beyond"]].sum().sum()   # an unknown act is not graded
        rows.append({"Version": version, "Drafts": len(t), "Written": len(w), "Lost": int(t["lost"].astype(bool).sum()),
                     "Precedent citations": int(w["n_citations"].sum()),
                     "... to a supplied precedent": int(w["cited_supplied"].sum()),
                     "... to a corpus decision not supplied": int(w["cited_corpus_not_supplied"].sum()),
                     "... process not in the corpus": int(w["process_not_in_corpus"].sum()),
                     "... ruling number unresolved": int(w["ruling_unresolved"].sum()),
                     "... ruling number of another decision": int(w["ruling_process_mismatch"].sum()),
                     "Drafts citing their own source decision": int(w["cited_source"].astype(bool).sum()),
                     "Statute articles graded": int(graded),
                     "... beyond the act's last article": int(w["statute_beyond"].sum()),
                     "... in range, not read (manual review)": int(w["statute_in_range"].sum()),
                     "Statute articles of an act not in the lookup": int(w["statute_unknown_act"].sum())})
    return pd.DataFrame(rows)


@table("tab_a4_dev_arms", "A4 improvement round (development notices, one full-mode draft each, before v2.7 was frozen): "
                          "the fact controls alone and together. Rule fixed beforehand: fewest invented facts per draft "
                          "among eligible arms (recall drop at most 0.10; legal accuracy and use of precedents drops at "
                          "most 0.5; at most 8 [VERIFICAR] markers per draft), the fewest controls within 0.5 of the best.")
def tab_a4_dev_arms(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a4_dev_arms", a)
    return pd.DataFrame({"Arm": t["arm"], "Invented per draft (estimated)": t["invented_estimate"].map(lambda x: P.f(x, 1)),
                         "Argument recall": t["recall"].map(lambda x: P.f(x, 2)),
                         "Legal accuracy": t["gpt_legal_accuracy"].map(lambda x: P.f(x, 1)),
                         "Fidelity to facts": t["gpt_fidelity_to_facts"].map(lambda x: P.f(x, 1)),
                         "Use of precedents": t["gpt_use_of_precedents"].map(lambda x: P.f(x, 1)),
                         "[VERIFICAR] markers per draft": t["verify_markers"].map(lambda x: P.f(x, 1)),
                         "Cost per draft": t["cost_usd"].map(P.usd), "Latency (s)": t["latency_s"].map(lambda x: P.f(x, 0)),
                         "Eligible": t["eligible"].map({True: "yes", False: "no"}), "Selected": t["selected"].map({True: "yes", False: ""})})


# ---------------------------------------------------------------------------- A6, failure modes

@table("tab_a6_latency", "A6: latency and cost per call of version 2 as deployed (v2.7, legal search off), with v2.7 in "
                         "full mode and v2.6 for comparison; 30 calls each, interleaved, one at a time, on a local server "
                         "calling the cloud APIs (not production Cloud Run, not concurrent load). p50 with a bootstrap 95% CI.")
def tab_a6_latency(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a6_latency", a)
    names = {"map_view": "Map view", "case_upload": "Case upload", "chat": "Chat",
             "v2.7_deployed": "Defense draft, as deployed (v2.7, no legal search)",
             "v2.7_full": "Defense draft, v2.7 full pipeline", "v2.6_full": "Defense draft, v2.6 full pipeline"}
    return pd.DataFrame({
        "Endpoint": [names.get(u, u) for u in t["unit"]],
        "Calls ok": [P.share(o, c) for o, c in zip(t["ok"], t["calls"])],
        "Latency p50 (s)": [f"{P.f(x, 1)} ({P.f(lo, 1)} to {P.f(hi, 1)})" for x, lo, hi in
                            zip(t["latency_p50_s"], t["latency_p50_ci_low_s"], t["latency_p50_ci_high_s"])],
        "Latency p95 (s)": t["latency_p95_s"].map(lambda x: P.f(x, 1)),
        "Model calls": t["model_calls_mean"].map(lambda x: P.f(x, 1)),
        "Prompt tokens": t["prompt_tokens_mean"].map(lambda x: f"{x:,.0f}"),
        "Output tokens": t["output_tokens_mean"].map(lambda x: f"{x:,.0f}"),
        "USD per call": t["usd_mean"].map(P.usd)})


@table("tab_a6_paired", "A6: defense arms paired by round (30 rounds; wall-clock): median difference, rounds in which the "
                        "first arm was slower, and the Wilcoxon signed-rank p (Holm).")
def tab_a6_paired(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("a6_paired", a)
    names = {("v2.7_full", "v2.6_full"): "v2.7 − v2.6, full pipeline (the fact controls)",
             ("v2.7_deployed", "v2.7_full"): "v2.7 as deployed − v2.7 full (no legal search)"}
    return pd.DataFrame({"Contrast": [names[(x, y)] for x, y in zip(t["arm"], t["against"])],
                         "Median difference (s)": t["median_difference_s"].map(lambda x: P.signed(x, 1)),
                         "Rounds slower": [P.share(round(s * r), r) for s, r in zip(t["share_of_rounds_slower"], t["rounds"])],
                         "Median ratio": t["median_ratio"].map(lambda x: P.f(x, 2)),
                         "Wilcoxon p (Holm)": t["wilcoxon_p_holm"].map(P.pval)})


@table("tab_failure_modes", "Failure modes of version 2: how each is detected and mitigated, the residual risk, and the "
                            "rate measured in the evaluation.")
def tab_failure_modes(S: P.Sources, a: str) -> pd.DataFrame:
    t = S.csv("failure_modes", a)
    return t[["failure", "detection", "mitigation", "residual_risk", "measured_rate", "measured_in"]].rename(columns={
        "failure": "Failure mode", "detection": "Detection", "mitigation": "Mitigation", "residual_risk": "Residual risk",
        "measured_rate": "Measured rate", "measured_in": "Measured in"})


# =========================================================================================== figures

def _style(ax, grid: str = "x") -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=7.5)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


@figure("fig_a1_retrieval", "A1: nDCG@10 per method on the comparison set (1,503 decisions), with the bootstrap 95% "
                            "interval over queries. Left: legal basis (R-base, primary; 824 queries). Right: infraction "
                            "category (R-cat). Diamond: version 2; square: version 1. Each panel has its own scale.")
def fig_a1_retrieval(S: P.Sources, a: str):
    t = S.csv("a1_main", a)
    t = t[(t["scope"] == "comparison_set") & (t["metric"] == "nDCG@10")]
    labels = _a1_methods(S)
    methods = METHOD_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(methods) + 0.9), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, rel, title in zip(axes, ["R-base", "R-cat"], ["Legal basis (R-base)", "Infraction category (R-cat)"]):
        sub = t[t["relevance"] == rel].set_index("method")
        for pos, m in zip(y, methods):
            colour, marker = {"M4": (V2, "D"), "M3": (V1, "s")}.get(m, (NEUTRAL, "o"))
            r = sub.loc[m]
            ax.plot([r["ci_low"], r["ci_high"]], [pos, pos], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(r["mean"], pos, marker=marker, markersize=6.5, color=colour, markeredgecolor="white", markeredgewidth=1.0)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.set_xlabel("nDCG@10", color=MUTED)
        _style(ax)
    axes[0].set_yticks(y, [labels[m] for m in methods], color=INK)
    fig.tight_layout(w_pad=2.0)
    return fig


@figure("fig_a2_trustworthiness", "A2: trustworthiness (top) and continuity (bottom) of two-dimensional maps against k. "
                                  "Left: each version's map on the comparison set (1,503 decisions), each against its own "
                                  "vectors. Right: version 2's served map on its corpus (2,390), with t-SNE, PaCMAP and PCA "
                                  "on the same vectors. Bands: lowest to highest of five seeds (the served map is one fit).")
def fig_a2_trustworthiness(S: P.Sources, a: str):
    t = S.csv("a2_metrics", a)
    idx = t.set_index(["set", "projection", "metric"])
    ks = [5, 10, 20, 50]
    panels = [("Comparison set (1,503)", [("v1_comparison", "umap_n15_d0.1", "Version 1 map", V1, "s", "-"),
                                          ("v2_comparison", "frozen_v2", "Version 2 served map", V2, "D", "-")]),
              ("Version 2 corpus (2,390)", [("v2_full", "frozen_v2", "Version 2 served map", V2, "D", "-"),
                                            ("v2_full", "tsne_p30", "t-SNE", NEUTRAL, "o", "--"),
                                            ("v2_full", "pacmap", "PaCMAP", NEUTRAL, "^", ":"),
                                            ("v2_full", "pca", "PCA", NEUTRAL, "v", "-.")])]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey="row")
    for row, metric in enumerate(["trustworthiness", "continuity"]):
        for col, (title, series) in enumerate(panels):
            ax = axes[row, col]
            for st, proj, label, colour, marker, line in series:
                v = idx.loc[[(st, proj, f"{metric}@{k}") for k in ks]]
                if (v["fits"] > 1).any():
                    ax.fill_between(ks, v["min"], v["max"], color=colour, alpha=0.15, linewidth=0)
                ax.plot(ks, v["mean"], color=colour, linestyle=line, linewidth=1.6, marker=marker, markersize=5.5,
                        markeredgecolor="white", markeredgewidth=0.8, label=label)
            ax.set_xscale("log")
            ax.set_xticks(ks, [str(k) for k in ks])
            ax.minorticks_off()
            _style(ax, "y")
            if row == 0:
                ax.set_title(title, loc="left", color=INK)
            if row == 1:
                ax.set_xlabel("k (nearest neighbours)", color=MUTED)
        axes[row, 0].set_ylabel(metric.capitalize(), color=INK)
    handles, names = {}, []
    for ax in axes[0]:
        for h, n in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(n, h)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=len(handles), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


@figure("fig_a3_strategies", "A3 strategy selection on the development sets: unsupported rate (mean of GPT-5.4 and Mistral "
                             "Medium 3) against coverage of the reference points. Dashed: the coverage limit (the baseline's "
                             "less 0.10); the eligible strategy with the lowest unsupported rate was selected.")
def fig_a3_strategies(S: P.Sources, a: str):
    t = S.csv("a3_strategies", a)
    limit = float(t.loc[t["candidate"] == "baseline", "coverage"].iloc[0]) - S.json("a3_selection", a)["rule"]["constraints"]["coverage_drop_max"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.axvline(limit, color=MUTED, linestyle="--", linewidth=1)
    ax.text(limit, t["unsupported_rate"].max() * 1.12, " coverage limit", color=MUTED, fontsize=7, va="bottom")
    for _, r in t.iterrows():
        colour = V2 if r["selected"] else (NEUTRAL if r["eligible"] else "#b5b4b0")
        marker = "D" if r["selected"] else ("o" if r["eligible"] else "x")
        ax.plot(r["coverage"], r["unsupported_rate"], marker=marker, markersize=7, color=colour, linestyle="none",
                markeredgecolor="white" if marker != "x" else colour, markeredgewidth=1.0)
        ax.annotate(r["candidate"], (r["coverage"], r["unsupported_rate"]), xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.set_xlabel("Coverage of the reference points", color=MUTED)
    ax.set_ylabel("Unsupported rate (claims)", color=MUTED)
    ax.set_ylim(0, t["unsupported_rate"].max() * 1.2)
    _style(ax, "y")
    fig.tight_layout()
    return fig


@figure("fig_a5_context_size", "A5: the chat with k decisions in its context. Unsupported claims per open answer after the "
                                "control (GPT-5.4 and Mistral Medium 3, shaded 95% interval over questions), median latency and "
                                "mean cost of an open answer. Open marker: each of 30 decisions given as a short summary.")
def fig_a5_context_size(S: P.Sources, a: str):
    t = S.csv("a5_context", a)
    full, summary = [("k5", 5), ("k10", 10), ("k20", 20), ("k30", 30)], [("h30", 30)]

    def val(setting, section, group, metric, rater):
        m = (t["setting"] == setting) & (t["section"] == section) & (t["group"] == group) & (t["metric"] == metric)
        m &= t["rater"].isna() if rater is None else (t["rater"] == rater)
        r = t[m].iloc[0]
        return r["value"], r["ci_low"], r["ci_high"]

    panels = [("Unsupported claims per open answer", "faithfulness", "all", "unsupported_claims_per_answer", ["judge_a", "judge_b"]),
              ("Median latency, open answer (s)", "system", "open", "latency_s_median", [None]),
              ("Mean cost, open answer (USD)", "system", "open", "cost_usd_per_answer", [None])]
    names = S.cfg["labels"]["judges"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (title, section, group, metric, raters) in zip(axes, panels):
        for rater in raters:
            colour = JUDGE.get(rater, NEUTRAL)
            pts = [(k, *val(s, section, group, metric, rater)) for s, k in full]
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(x, y, color=colour, linewidth=1.8, marker="o" if rater != "judge_b" else "s", markersize=5.5,
                    markeredgecolor="white", markeredgewidth=0.8, label=names.get(rater, "") if rater else None)
            lo, hi = [p[2] for p in pts], [p[3] for p in pts]
            if not any(P.is_missing(v) for v in lo):
                ax.fill_between(x, lo, hi, color=colour, alpha=0.14, linewidth=0)
            for s, k in summary:
                ax.plot([k], [val(s, section, group, metric, rater)[0]], linestyle="none", marker="o" if rater != "judge_b" else "s",
                        markersize=6.5, markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.4)
        ax.set_xticks([5, 10, 20, 30])
        ax.set_xlabel("k (decisions in the context)", color=MUTED)
        ax.set_title(title, loc="left", color=INK, fontsize=8)
        ax.set_ylim(bottom=0)
        _style(ax, "y")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


@figure("fig_a4_invented_facts", "A4: estimated invented case facts per draft, by drafting mode, before (v2.6) and after "
                                 "(v2.7) the fact controls, 15 notices x 3 runs per mode. Bars: range from the Wilson bounds "
                                 "of the share each version's blind audit found invented.")
def fig_a4_invented_facts(S: P.Sources, a: str):
    v6, v7 = S.csv("a4_v26_invented", a).set_index("mode"), S.csv("a4_v27_invented", a).set_index("mode")
    modes = S.cfg["labels"]["modes"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(MODES))[::-1]
    for table_, colour, marker, label, dy in [(v6, V1, "s", "v2.6", 0.13), (v7, V2, "D", "v2.7 (fact controls)", -0.13)]:
        for pos, m in zip(y, MODES):
            r = table_.loc[m]
            ax.plot([r["low"], r["high"]], [pos + dy, pos + dy], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(r["estimated_invented"], pos + dy, marker=marker, markersize=6, color=colour, markeredgecolor="white",
                    markeredgewidth=0.8, label=label if m == MODES[0] else None)
    ax.set_yticks(y, [modes[m] for m in MODES], color=INK)
    ax.set_xlabel("Invented case facts per draft (estimated)", color=MUTED)
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    _style(ax)
    fig.tight_layout()
    return fig


@figure("fig_a4_ablation", "A4 ablation of version 2 (v2.7): means over the 15 notices per drafting mode (GPT-5.4 rubric, "
                           "1-5; argument recall and template coverage as shares), lost drafts scored as the worst outcome. "
                           "Filled: different from the full pipeline after Holm correction; hollow: not.")
def fig_a4_ablation(S: P.Sources, a: str):
    t = S.csv("a4_v27_ablation", a)
    t = t[t["analysis"] == "main"].set_index("metric")
    metrics = ["judge_a_use_of_precedents", "judge_a_persuasiveness", "judge_a_legal_accuracy", "judge_a_fidelity_to_facts",
               "recall", "template_coverage"]
    modes = S.cfg["labels"]["modes"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.4, 2.5), sharey=True)
    y = np.arange(len(MODES))[::-1]
    for ax, metric in zip(axes, metrics):
        r = t.loc[metric]
        for pos, m in zip(y, MODES):
            sig = m == "full" or r[f"holm_p_{m}"] < 0.05
            colour = V2 if m == "full" else NEUTRAL
            ax.plot(r[f"mean_{m}"], pos, marker="D" if m == "full" else "o", markersize=6, linestyle="none",
                    color=colour, markerfacecolor=colour if sig else "white", markeredgecolor=colour, markeredgewidth=1.2)
        ax.axvline(r["mean_full"], color=V2, linewidth=0.8, alpha=0.4)
        ax.set_title(f"{ABLATION_METRICS[metric]}\nFriedman {p_text(r['friedman_p'])}", loc="left", color=INK, fontsize=7.5)
        ax.margins(x=0.18)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        _style(ax)
        if metric in {"recall", "template_coverage"}:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1 if metric == "recall" else 0))
    axes[0].set_yticks(y, [modes[m] for m in MODES], color=INK)
    fig.tight_layout(w_pad=1.6)
    return fig


COPIED = {"fig_c1_agent_graph_v1": ("agent_graph_v1", "Version 1's agent graph (LangGraph fixed sequence), from the tagged code (Phase 2)."),
          "fig_c1_agent_graph_v2": ("agent_graph_v27", "Version 2's agent graph (Google ADK) at v2.7, with the fact controls, from the tagged code (Phases 2 and 11).")}


# =========================================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", help="write tables/ and figures/ under this folder instead of the paper folder")
    ap.add_argument("--only", nargs="+", help="build only these assets")
    ap.add_argument("--allow-stale", action="store_true", help="build even if a pinned source is not the newest version")
    args = ap.parse_args()

    cfg = load_config("paper_assets")
    S = P.Sources(cfg, allow_stale=args.allow_stale)
    problems = S.check()
    if problems and not args.allow_stale:
        raise SystemExit("Pinned sources need attention (config/paper_assets.yaml):\n  " + "\n  ".join(problems))
    if args.out_dir:
        tables_dir, figures_dir = Path(args.out_dir) / "tables", Path(args.out_dir) / "figures"
    else:
        base = resolve(load_config("paths")["paper_folder"])
        if not base.is_dir():
            raise SystemExit(f"The paper folder ({base}) does not exist here: give --out-dir")
        tables_dir, figures_dir = base / cfg["output"]["tables"], base / cfg["output"]["figures"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else None
    index = P.manifest_index()

    with RunManifest(SCRIPT, resolve(load_config("paths")["results"]), record_outputs=False,
                     params={"out_tables": str(tables_dir), "out_figures": str(figures_dir), "only": args.only,
                             "allow_stale": args.allow_stale, "stale": problems}) as run:
        records = []
        for name, (fn, note) in TABLES.items():
            if wanted and name not in wanted:
                continue
            out = tables_dir / f"{name}.csv"
            try:
                table = fn(S, name)
            except P.MissingOptionalSource as e:
                records.append((name, "table (not built)", [], f"{note} Not built: {e.note}."))
                continue
            table.to_csv(out, index=False)
            records.append((name, "table", [out], note))
        for name, (fn, note) in FIGURES.items():
            if wanted and name not in wanted:
                continue
            try:
                fig = fn(S, name)
            except P.MissingOptionalSource as e:
                records.append((name, "figure (not built)", [], f"{note} Not built: {e.note}."))
                continue
            outs = []
            for fmt in cfg["output"]["figure_formats"]:
                out = figures_dir / f"{name}.{fmt}"
                fig.savefig(out, dpi=cfg["output"]["png_dpi"] if fmt == "png" else None, bbox_inches="tight",
                            metadata={"CreationDate": None} if fmt == "pdf" else None)
                outs.append(out)
            plt.close(fig)
            records.append((name, "figure", outs, note))
        for name, (src, note) in COPIED.items():
            if wanted and name not in wanted:
                continue
            out = figures_dir / f"{name}.svg"
            shutil.copyfile(S.path(src, name), out)
            records.append((name, "figure (copied)", [out], note))

        rows = []
        for name, kind, outs, note in records:
            sources = sorted(S.used.get(name, set()))
            for src in sources:
                run.add_input(S.paths[src], asset=name)
            prod = {src: P.producer(S.paths[src], index) for src in sources}
            rows.append({"asset": name, "kind": kind, "files": "; ".join(o.name for o in outs), "note": note,
                         "sources": "; ".join(P.rel(S.paths[s]) for s in sources),
                         "source_sha256": "; ".join(sha256_file(S.paths[s])[:12] for s in sources),
                         "scripts": "; ".join(sorted({p[1] for p in prod.values() if p})) or "",
                         "run_manifests": "; ".join(f"{P.rel(S.paths[s])} <- {p[0]}" if p else
                                                    f"{P.rel(S.paths[s])} <- ({cfg.get('source_notes', {}).get(s, 'no run manifest')})"
                                                    for s, p in prod.items())})
            for o in outs:
                run.add_output(o, note)
        assets = pd.DataFrame(rows)
        if not wanted:
            assets_path = tables_dir / "ASSETS.csv"
            assets.to_csv(assets_path, index=False)
            run.add_output(assets_path, "index of the paper's assets and their sources")
        run.data["summary"] = {"tables": sum(r[1] == "table" for r in records),
                               "figures": sum(r[1].startswith("figure") and "not built" not in r[1] for r in records),
                               "not_built": [r[0] for r in records if "not built" in r[1]]}
        print(f"{run.data['summary']['tables']} tables -> {tables_dir}\n{run.data['summary']['figures']} figures -> {figures_dir}")
        if run.data["summary"]["not_built"]:
            print("not built (source absent):", ", ".join(run.data["summary"]["not_built"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
