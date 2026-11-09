"""Phase 4c, A1: the retrieval benchmark on the CRSNSP corpus.

Protocol (config/a1_methods.yaml): leave-one-out. Each decision is a query against all the other
decisions of its scope; decisions of the same process are never ranked for each other. A query is
evaluated when at least one other decision is relevant to it, so every method is scored on the same
queries. Scopes: the comparison set (1,503 decisions, primary) and version 2's corpus (2,390).

Relevance, reported separately: R-base (identical set of specific provisions of the legal basis,
primary), R-base-relaxed (Jaccard >= 0.5), R-prov (provisions read from the report and vote), R-cat
(same infraction category, assigned by Gemini 3.7 Flash) and R-cat-mistral (the same taxonomy
assigned blind by Mistral Medium 3, where that labelling exists).

Per method and metric: mean over queries, percentile bootstrap 95% interval (10,000 resamples of
queries, the same draws for every method), and a paired randomisation test against BM25 with Holm
correction over the methods. The same test is applied to the planned contrasts: version 2 against
version 1 (M4 - M3), against the full report alone (M4 - M5) and against the compression alone
(M4 - M4c), and the two compression models (M4c - M4b).

Decisions with several infractions: results/a1_retrieval_by_complexity.csv repeats the summary
for the queries with one infraction and for those with several (one or several specific provisions
for R-base; one or several categories for R-cat), and R-cat-multi counts two decisions as relevant
when they share any category, primary or secondary.

Outcome-consistency proxies: among each query's 10 nearest neighbours, the share with the same
outcome and the share with the same penalty type, next to the share expected by chance.

Outputs (never overwritten):
  results/a1_retrieval_main.csv, a1_retrieval_contrasts.csv, a1_outcome_consistency.csv
  results/a1_retrieval_by_complexity.csv, a1_retrieval_by_complexity_contrasts.csv
  results/figures/a1_retrieval_main.pdf, a1_retrieval_relevance.pdf
  data/v2/a1_runs/a1_per_query.parquet     every query's metrics (scope, relevance, method)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jm_eval import stats  # noqa: E402
from jm_eval.io import load_config, new_artefact_path, resolve  # noqa: E402
from jm_eval.manifest import RunManifest  # noqa: E402
from jm_eval.retrieval import corpus as C  # noqa: E402
from jm_eval.retrieval import labels as L  # noqa: E402
from jm_eval.retrieval import lexical, metrics  # noqa: E402

SCRIPT = "46_a1_retrieval"
# From the v2.3 freeze, version 2's vector (M4) is the full report and its compression summed, and the
# compression alone is M4c. Before it, M4 was the compression and the contrasts were M4 - M3,
# M4 - M5 and M4 - M4b (results files up to _4).
CONTRASTS = [("M4", "M3", "version 2 - version 1"), ("M4", "M5", "version 2 - full report alone"),
             ("M4", "M4c", "version 2 - compression alone"),
             ("M4c", "M4b", "Gemini 3.7 Flash - Gemini 2.5 Flash-Lite compression")]
PROXIES = {"outcome": "outcome", "penalty_type": "penalty_key"}
# Figure colours: the two versions carry a hue, every other method is neutral (validated: blue and
# orange are distinguishable under colour-vision deficiency; marker shapes repeat the distinction).
STYLE = {"M4": ("#2a78d6", "D"), "M3": ("#eb6834", "s")}
NEUTRAL, INK, MUTED, GRID = ("#6b6a66", "o"), "#0b0b0b", "#52514e", "#e4e3df"


class Similarities:
    """Similarity matrix of every method over a subset of the corpus (rows in acordao_hash order)."""

    def __init__(self, cfg: dict):
        self.cfg, self._counts, self._vectors = cfg, {}, {}

    def counts(self, text_key: str):
        if text_key not in self._counts:
            self._counts[text_key] = lexical.count_matrix(C.load_texts(text_key)[1])
        return self._counts[text_key]

    def matrix(self, method: str, rows: np.ndarray) -> np.ndarray:
        spec = self.cfg["methods"][method]
        if spec["kind"] == "tfidf":
            return lexical.tfidf_similarity_from_counts(self.counts(spec["text"])[rows])
        if spec["kind"] == "bm25":
            return lexical.bm25_scores_from_counts(self.counts(spec["text"])[rows], spec["k1"], spec["b"])
        if method not in self._vectors:
            self._vectors[method] = C.method_vectors(method)
        return metrics.cosine_similarity(self._vectors[method][rows])


def summarise(per_query: dict[str, pd.DataFrame], baseline: str, cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Means, bootstrap intervals and tests for one scope and relevance definition."""
    methods, names = list(per_query), metrics.METRICS
    stacked = np.hstack([per_query[m][names].to_numpy() for m in methods])
    low, high = stats.bootstrap_mean_ci(stacked, resamples=cfg["bootstrap_resamples"], seed=seed, chunk=200)
    rows = []
    for i, method in enumerate(methods):
        for j, metric in enumerate(names):
            values = per_query[method][metric].to_numpy()
            row = {"method": method, "metric": metric, "queries": len(values), "mean": values.mean(),
                   "ci_low": low[i * len(names) + j], "ci_high": high[i * len(names) + j]}
            if method != baseline:
                row["diff_vs_baseline"], row["p_vs_baseline"] = stats.paired_randomisation_test(
                    values, per_query[baseline][metric].to_numpy(), permutations=cfg["randomisation_permutations"], seed=seed)
            rows.append(row)
    main = pd.DataFrame(rows)
    main["p_holm_vs_baseline"] = np.nan
    for metric in names:
        tested = main[(main["metric"] == metric) & main["p_vs_baseline"].notna()].index
        main.loc[tested, "p_holm_vs_baseline"] = stats.holm(main.loc[tested, "p_vs_baseline"].to_numpy())

    rows = []
    for a, b, label in CONTRASTS:
        if a not in per_query or b not in per_query:
            continue
        for metric in names:
            x, y = per_query[a][metric].to_numpy(), per_query[b][metric].to_numpy()
            diff, p = stats.paired_randomisation_test(x, y, permutations=cfg["randomisation_permutations"], seed=seed)
            d_low, d_high = stats.bootstrap_mean_ci(x - y, resamples=cfg["bootstrap_resamples"], seed=seed, chunk=200)
            rows.append({"contrast": f"{a} - {b}", "label": label, "metric": metric, "queries": len(x),
                         "mean_a": x.mean(), "mean_b": y.mean(), "diff": diff, "ci_low": d_low, "ci_high": d_high, "p": p})
    contrasts = pd.DataFrame(rows)
    if len(contrasts):
        contrasts["p_holm"] = np.nan
        for metric in names:
            idx = contrasts[contrasts["metric"] == metric].index
            contrasts.loc[idx, "p_holm"] = stats.holm(contrasts.loc[idx, "p"].to_numpy())
    return main, contrasts


def by_complexity(per_query: dict[str, pd.DataFrame], sets: pd.Series, baseline: str, cfg: dict, seed: int,
                  scope: str, relevance: str) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """The same summary for the queries with one infraction and for those with several.

    `sets` holds, per decision of the scope, the provisions or categories that describe it: one
    element is one infraction, more are several.
    """
    size = np.array([len(s) if s is not None and hasattr(s, "__len__") else 0 for s in sets])
    queries = next(iter(per_query.values())).index
    out = []
    for group, mask in (("one", size[queries] == 1), ("several", size[queries] >= 2)):
        if mask.sum() < 2:
            continue
        table, contrasts = summarise({m: f[mask] for m, f in per_query.items()}, baseline, cfg, seed)
        out.append((table.assign(scope=scope, relevance=relevance, infractions=group),
                    contrasts.assign(scope=scope, relevance=relevance, infractions=group)))
    return out


def _axes_style(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def interval_figure(table: pd.DataFrame, panels: list[tuple[str, pd.Series]], methods: list[str], labels: dict,
                    title: str, path: Path) -> None:
    """Dot-and-interval plot: one row per method, one panel per entry of `panels` (title, row filter).

    Panels show different metrics or query sets, so each has its own scale (stated under the figure).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels) + 4.2, 0.36 * len(methods) + 1.5), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (panel_title, mask) in zip(np.atleast_1d(axes), panels):
        sub = table[mask].set_index("method")
        for pos, method in zip(y, methods):
            if method not in sub.index:
                continue
            colour, marker = STYLE.get(method, NEUTRAL)
            r = sub.loc[method]
            ax.plot([r["ci_low"], r["ci_high"]], [pos, pos], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(r["mean"], pos, marker=marker, markersize=7, color=colour, markeredgecolor="white", markeredgewidth=1.2)
        ax.set_title(panel_title, fontsize=9, color=INK, loc="left")
        ax.set_ylim(-0.6, len(methods) - 0.4)
        _axes_style(ax)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(y)
    first.set_yticklabels([f"{m}  {labels[m]}" for m in methods], fontsize=8, color=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Mean over queries with the bootstrap 95% interval. Diamond: version 2. Square: version 1. "
             "Each panel has its own scale.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), w_pad=2.0)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", help="default: every method of config/a1_methods.yaml")
    ap.add_argument("--out-dir", help="write everything here instead of results/ and data/v2/a1_runs/ (trial runs)")
    args = ap.parse_args()

    cfg, labels_cfg, paths = C.methods_config(), L.settings(), load_config("paths")
    seed = load_config("seeds")["bootstrap"]
    methods = args.methods or [m for m, spec in cfg["methods"].items() if spec.get("benchmark", True)]
    baseline, threshold, k = cfg["baseline"], labels_cfg["relevance"]["jaccard_threshold"], cfg["neighbours"]
    results = Path(args.out_dir) if args.out_dir else resolve(paths["results"])
    runs_dir = Path(args.out_dir) if args.out_dir else resolve(paths["data"]["v2"]) / "a1_runs"
    figures = results / "figures"
    for d in (runs_dir, figures):
        d.mkdir(parents=True, exist_ok=True)
    short = {m: s["label"] for m, s in cfg["methods"].items()}
    params = {"methods": methods, "baseline": baseline, "jaccard_threshold": threshold, "neighbours": k,
              "bootstrap_resamples": cfg["bootstrap_resamples"], "randomisation_permutations": cfg["randomisation_permutations"],
              "relevance": cfg["relevance"], "labels_version": labels_cfg["version"]}
    with RunManifest(SCRIPT, results, models={m: cfg["methods"][m] for m in methods}, params=params,
                     seeds={"bootstrap_and_randomisation": seed}, record_outputs=args.out_dir is None) as run:
        labels = C.load_labels()
        hashes = C.load_corpus([])["acordao_hash"].tolist()
        assert labels["acordao_hash"].tolist() == hashes, "labels and corpus are not aligned"
        run.add_input(resolve(labels_cfg["corpus"]))
        # a definition whose label does not exist yet (a category labelling not run) is left out
        available = {name: spec for name, spec in cfg["relevance"].items() if labels[spec["source"]].notna().any()}
        sims = Similarities(cfg)

        main_rows, contrast_rows, proxy_rows, per_query_rows, query_counts = [], [], [], [], {}
        complexity_rows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        scopes = {"comparison_set": np.flatnonzero(labels["in_comparison_set"].to_numpy()), "v2": np.arange(len(labels))}
        for scope, rows in scopes.items():
            sub = labels.iloc[rows].reset_index(drop=True)
            allowed = metrics.candidate_mask(sub["process_group"].to_numpy())
            relevance = {name: L.build_relevance(sub, spec, threshold) for name, spec in available.items()}
            per_query: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in relevance}
            proxies: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in PROXIES}
            for method in methods:
                similarity = sims.matrix(method, rows)
                for name, rel in relevance.items():
                    per_query[name][method] = metrics.per_query_metrics(similarity, rel, allowed)
                for proxy, column in PROXIES.items():
                    proxies[proxy][method] = metrics.neighbour_agreement(similarity, allowed, sub[column].to_numpy(), k)
                print(scope, method, {n: round(float(per_query[n][method]["nDCG@10"].mean()), 4) for n in relevance}, flush=True)
                del similarity

            for name in relevance:
                complexity_rows.extend(by_complexity(per_query[name], sub[cfg["relevance"][name]["sets"]], baseline, cfg, seed,
                                                     scope, name) if cfg["relevance"][name].get("sets") else [])
                table, contrasts = summarise(per_query[name], baseline, cfg, seed)
                query_counts[f"{scope}/{name}"] = int(table["queries"].iloc[0])
                main_rows.append(table.assign(scope=scope, relevance=name))
                contrast_rows.append(contrasts.assign(scope=scope, relevance=name))
                for method, frame in per_query[name].items():
                    per_query_rows.append(frame.assign(scope=scope, relevance=name, method=method,
                                                       acordao_hash=sub["acordao_hash"].to_numpy()[frame.index]))
            for proxy, by_method in proxies.items():
                observed = np.column_stack([f["observed"].to_numpy() for f in by_method.values()])
                low, high = stats.bootstrap_mean_ci(observed, resamples=cfg["bootstrap_resamples"], seed=seed, chunk=200)
                for i, (method, frame) in enumerate(by_method.items()):
                    proxy_rows.append({"scope": scope, "proxy": proxy, "method": method, "label": short[method],
                                       "queries": len(frame), "neighbours": k, "share_same": frame["observed"].mean(),
                                       "ci_low": low[i], "ci_high": high[i], "share_expected_by_chance": frame["expected"].mean(),
                                       "lift": frame["observed"].mean() / frame["expected"].mean()})

        columns = ["scope", "relevance", "method", "label", "metric", "queries", "mean", "ci_low", "ci_high",
                   "diff_vs_baseline", "p_vs_baseline", "p_holm_vs_baseline"]
        main = pd.concat(main_rows, ignore_index=True)
        main["label"] = main["method"].map(short)
        main = main[columns]
        contrasts = pd.concat(contrast_rows, ignore_index=True)
        contrasts = contrasts[["scope", "relevance", *[c for c in contrasts.columns if c not in ("scope", "relevance")]]]
        proxy_table = pd.DataFrame(proxy_rows)
        per_query_table = pd.concat(per_query_rows, ignore_index=True)

        lead = ["scope", "relevance", "infractions"]
        complexity = pd.concat([m for m, _ in complexity_rows], ignore_index=True)
        complexity["label"] = complexity["method"].map(short)
        complexity = complexity[lead + [c for c in columns if c not in lead]]
        complexity_contrasts = pd.concat([c for _, c in complexity_rows], ignore_index=True)
        complexity_contrasts = complexity_contrasts[lead + [c for c in complexity_contrasts.columns if c not in lead]]
        outputs = [(results / "a1_retrieval_by_complexity.csv", complexity,
                    "A1: retrieval metrics for queries with one infraction and with several"),
                   (results / "a1_retrieval_by_complexity_contrasts.csv", complexity_contrasts,
                    "A1: planned contrasts for queries with one infraction and with several"),
                   (results / "a1_retrieval_main.csv", main, "A1: retrieval metrics per method, scope and relevance definition"),
                   (results / "a1_retrieval_contrasts.csv", contrasts, "A1: planned contrasts (v2 - v1, compression, compression model)"),
                   (results / "a1_outcome_consistency.csv", proxy_table, "A1: same outcome and same penalty type among the top-10 neighbours")]
        for path, frame, note in outputs:
            path = new_artefact_path(path)
            frame.to_csv(path, index=False, float_format="%.6g")
            run.add_output(path, note)
        per_query_path = new_artefact_path(runs_dir / "a1_per_query.parquet")
        per_query_table.to_parquet(per_query_path, index=False)
        run.add_output(per_query_path, "A1: metrics of every query")

        primary = (main["scope"] == "comparison_set") & (main["relevance"] == "R-base")
        fig_path = new_artefact_path(figures / "a1_retrieval_main.pdf")
        interval_figure(main, [(metric, primary & (main["metric"] == metric)) for metric in ("nDCG@10", "MAP", "R@5")],
                        methods, short, f"Retrieval on the comparison set, R-base ({query_counts['comparison_set/R-base']:,} queries)",
                        fig_path)
        run.add_output(fig_path, "A1 figure: nDCG@10, MAP and R@5 per method (comparison set, R-base)")
        ndcg = (main["scope"] == "comparison_set") & (main["metric"] == "nDCG@10")
        fig_path = new_artefact_path(figures / "a1_retrieval_relevance.pdf")
        interval_figure(main, [(f"{name}\n{query_counts[f'comparison_set/{name}']:,} queries", ndcg & (main["relevance"] == name))
                               for name in cfg["relevance"] if f"comparison_set/{name}" in query_counts],
                        methods, short, "nDCG@10 on the comparison set under each relevance definition", fig_path)
        run.add_output(fig_path, "A1 figure: nDCG@10 per method under each relevance definition (comparison set)")

        run.data["summary"] = {"queries": query_counts, "relevance_definitions": list(available)}
        show = main[primary].pivot(index="method", columns="metric", values="mean").loc[methods, metrics.METRICS]
        print(show.round(4).to_string())
        print(query_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
