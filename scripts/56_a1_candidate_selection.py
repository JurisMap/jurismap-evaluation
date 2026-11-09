"""Phase 4.1, A1: score the candidates for version 2's retrieval vector and apply the selection rule.

Candidates (config/a1_candidates.yaml) are weighted sums of the compressed report's vector, the
full report's vector and the full report embedded in chunks, all with version 2's embedding model.
The protocol is that of scripts/46 (leave-one-out, decisions of the same process never ranked).

Queries are split by process group into a development half and a held-out half; every decision
stays a candidate for every ranking. The selection reads development queries only:
  score = mean over the label families (legal basis; category) of the mean nDCG@10 of the family's
          relevance definitions, on the corpus version 2 serves;
  the candidate with the highest score replaces the current vectors only if the bootstrap 95%
  interval of its paired score difference to the current one is above zero.
The held-out half then shows whether the choice holds on queries that took no part in it.

Also reported: every metric of A1 per candidate, split and scope; the selected candidate against
the current one, the full report and version 1's method (paired tests); queries with one infraction
and with several; and how many of a decision's 10 nearest neighbours each candidate keeps when the
compression is run again (the 50 reports of scripts/45).

Outputs (never overwritten):
  results/a1_candidates.csv, a1_candidates_contrasts.csv, a1_candidates_selection.json
  results/figures/a1_candidates.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jm_eval import stats  # noqa: E402
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve  # noqa: E402
from jm_eval.manifest import RunManifest  # noqa: E402
from jm_eval.retrieval import candidates as K  # noqa: E402
from jm_eval.retrieval import corpus as C  # noqa: E402
from jm_eval.retrieval import labels as L  # noqa: E402
from jm_eval.retrieval import lexical, metrics  # noqa: E402

SCRIPT = "56_a1_candidate_selection"
SPLITS = ("development", "held_out", "all")
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#6b6a66"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def per_decision(frame: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    """Per metric: one value per decision of the scope, NaN where the decision is not a query."""
    out = {}
    for metric in metrics.METRICS:
        values = np.full(n, np.nan)
        values[frame.index.to_numpy()] = frame[metric].to_numpy()
        out[metric] = values
    return out


def summarise(values: dict[str, dict[str, np.ndarray]], mask: np.ndarray, cfg: dict, seed: int) -> pd.DataFrame:
    """Mean of every metric per method over the queries of `mask`, with the bootstrap interval."""
    methods = list(values)
    queries = mask & ~np.isnan(values[methods[0]]["nDCG@10"])
    stacked = np.column_stack([values[m][metric][queries] for m in methods for metric in metrics.METRICS])
    low, high = stats.bootstrap_mean_ci(stacked, resamples=cfg["selection"]["bootstrap_resamples"], seed=seed, chunk=200)
    rows = []
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics.METRICS):
            column = i * len(metrics.METRICS) + j
            rows.append({"method": method, "metric": metric, "queries": int(queries.sum()), "mean": stacked[:, column].mean(),
                         "ci_low": low[column], "ci_high": high[column]})
    return pd.DataFrame(rows)


def contrast(values: dict[str, dict[str, np.ndarray]], a: str, b: str, mask: np.ndarray, cfg: dict, seed: int) -> list[dict]:
    queries = mask & ~np.isnan(values[a]["nDCG@10"])
    rows = []
    for metric in metrics.METRICS:
        x, y = values[a][metric][queries], values[b][metric][queries]
        diff, p = stats.paired_randomisation_test(x, y, permutations=cfg["selection"]["bootstrap_resamples"], seed=seed)
        low, high = stats.bootstrap_mean_ci(x - y, resamples=cfg["selection"]["bootstrap_resamples"], seed=seed, chunk=200)
        rows.append({"contrast": f"{a} - {b}", "metric": metric, "queries": int(queries.sum()), "mean_a": x.mean(), "mean_b": y.mean(),
                     "diff": diff, "ci_low": low, "ci_high": high, "p": p})
    return rows


def neighbours_kept(frozen: np.ndarray, rerun: np.ndarray, rows: list[int], k: int) -> float:
    """Mean number of a decision's k nearest neighbours kept when its vector is replaced by `rerun`."""
    kept = []
    for new, row in zip(rerun, rows):
        before, after = frozen @ frozen[row], frozen @ new
        before[row] = after[row] = -np.inf
        kept.append(len(set(np.argsort(-before, kind="stable")[:k]) & set(np.argsort(-after, kind="stable")[:k])))
    return float(np.mean(kept))


def figure(table: pd.DataFrame, panels: list[tuple[str, pd.Series]], methods: list[str], labels: dict, marks: dict, title: str, path: Path) -> None:
    """Dot-and-interval plot, one row per method and one panel per entry of `panels`; each panel has its own scale."""
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels) + 4.6, 0.36 * len(methods) + 1.6), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (panel_title, mask) in zip(np.atleast_1d(axes), panels):
        sub = table[mask].set_index("method")
        for pos, method in zip(y, methods):
            colour, marker = marks.get(method, (NEUTRAL, "o"))
            r = sub.loc[method]
            ax.plot([r["ci_low"], r["ci_high"]], [pos, pos], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(r["mean"], pos, marker=marker, markersize=7, color=colour, markeredgecolor="white", markeredgewidth=1.2)
        ax.set_title(panel_title, fontsize=9, color=INK, loc="left")
        ax.set_ylim(-0.6, len(methods) - 0.4)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(y)
    first.set_yticklabels([labels[m] for m in methods], fontsize=8, color=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Mean nDCG@10 over the held-out queries with the bootstrap 95% interval. Diamond: version 2's current vectors. "
             "Star: the selected candidate. Square: version 1. Each panel has its own scale.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), w_pad=2.0)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", help="write everything here instead of results/ (trial runs)")
    args = ap.parse_args()

    cfg, methods_cfg, labels_cfg, paths, seeds = K.settings(), C.methods_config(), L.settings(), load_config("paths"), load_config("seeds")
    seed, threshold, sel = seeds["bootstrap"], labels_cfg["relevance"]["jaccard_threshold"], cfg["selection"]
    results = Path(args.out_dir) if args.out_dir else resolve(paths["results"])
    (results / "figures").mkdir(parents=True, exist_ok=True)
    candidates, references, current = list(cfg["candidates"]), list(cfg["references"]), cfg["current"]
    part_methods = {name: spec["method"] for name, spec in cfg["parts"].items()}
    with RunManifest(SCRIPT, results, models={m: methods_cfg["methods"][m] for m in [*part_methods.values(), *references]},
                     params={"candidates": cfg, "relevance": methods_cfg["relevance"], "jaccard_threshold": threshold},
                     seeds={"a1_selection_split": seeds["a1_selection_split"], "bootstrap": seed}, record_outputs=args.out_dir is None) as run:
        labels = C.load_labels()
        run.add_input(resolve(labels_cfg["corpus"]))
        parts = {name: C.method_vectors(method) for name, method in part_methods.items()}
        vectors = {name: K.combine(parts, weights) for name, weights in cfg["candidates"].items()}
        development = K.development_mask(labels["process_group"].to_numpy(), cfg["split"]["development_share"], seeds["a1_selection_split"])
        available = {name: spec for name, spec in methods_cfg["relevance"].items() if labels[spec["source"]].notna().any()}
        counts = lexical.count_matrix(C.load_texts("report")[1]) if set(references) & {"M1", "M2"} else None

        def similarity(method: str, rows: np.ndarray) -> np.ndarray:
            if method in vectors:
                return vectors[method][rows] @ vectors[method][rows].T
            spec = methods_cfg["methods"][method]
            if spec["kind"] == "tfidf":
                return lexical.tfidf_similarity_from_counts(counts[rows])
            if spec["kind"] == "bm25":
                return lexical.bm25_scores_from_counts(counts[rows], spec["k1"], spec["b"])
            return metrics.cosine_similarity(C.method_vectors(method)[rows])

        tables, values_of, masks_of, sets_of = [], {}, {}, {}
        scopes = {"v2": np.arange(len(labels)), "comparison_set": np.flatnonzero(labels["in_comparison_set"].to_numpy())}
        for scope, rows in scopes.items():
            sub = labels.iloc[rows].reset_index(drop=True)
            allowed = metrics.candidate_mask(sub["process_group"].to_numpy())
            relevance = {name: L.build_relevance(sub, spec, threshold) for name, spec in available.items()}
            values: dict[str, dict[str, dict[str, np.ndarray]]] = {name: {} for name in relevance}
            for method in [*candidates, *references]:
                sim = similarity(method, rows)
                for name, rel in relevance.items():
                    values[name][method] = per_decision(metrics.per_query_metrics(sim, rel, allowed), len(rows))
                print(scope, method, {n: round(float(np.nanmean(values[n][method]["nDCG@10"][development[rows]])), 4) for n in relevance}, flush=True)
                del sim
            masks = {"development": development[rows], "held_out": ~development[rows], "all": np.ones(len(rows), dtype=bool)}
            for split, mask in masks.items():
                for name in relevance:
                    tables.append(summarise(values[name], mask, cfg, seed).assign(scope=scope, split=split, relevance=name))
            values_of[scope], masks_of[scope] = values, masks
            sets_of[scope] = {name: np.array([len(s) if s is not None and hasattr(s, "__len__") else 0 for s in sub[spec["sets"]]])
                              for name, spec in available.items() if spec.get("sets")}

        table = pd.concat(tables, ignore_index=True)
        table["kind"] = np.where(table["method"].isin(candidates), "candidate", "reference")
        table = table[["scope", "split", "relevance", "method", "kind", "metric", "queries", "mean", "ci_low", "ci_high"]]

        # ---- selection: development queries of the selection scope only
        scope, families = cfg["scope"], sel["families"]
        definitions = [d for family in families.values() for d in family]

        def score_inputs(split: str) -> dict[str, np.ndarray]:
            mask = masks_of[scope][split]
            return {d: np.column_stack([values_of[scope][d][c][sel["metric"]][mask] for c in candidates]) for d in definitions}

        dev = score_inputs("development")
        scores = dict(zip(candidates, K.selection_score(dev, families).tolist()))
        best = max(scores, key=scores.get)
        if best == current:
            interval, replace = None, False
        else:
            interval = K.score_difference_interval(dev, families, candidates.index(best), candidates.index(current),
                                                   resamples=sel["bootstrap_resamples"], seed=seed)
            replace = interval[1] > 0
        selected = best if replace else current
        held = score_inputs("held_out")
        held_scores = dict(zip(candidates, K.selection_score(held, families).tolist()))
        held_interval = None if best == current else K.score_difference_interval(
            held, families, candidates.index(best), candidates.index(current), resamples=sel["bootstrap_resamples"], seed=seed)

        # ---- the selected candidate against the current vectors, the full report and version 1
        contrast_rows = []
        others = [c for c in dict.fromkeys([current, "full", *[r for r in references if r == "M3"]]) if c != best]
        for scope_name in scopes:
            for split in ("held_out", "all"):
                for name in available:
                    for other in others:
                        contrast_rows += [{"scope": scope_name, "split": split, "relevance": name, "infractions": "any", **row}
                                          for row in contrast(values_of[scope_name][name], best, other, masks_of[scope_name][split], cfg, seed)]
            for name, size in sets_of[scope_name].items():            # one infraction or several, all queries
                for group, group_mask in (("one", size == 1), ("several", size >= 2)):
                    for other in others:
                        contrast_rows += [{"scope": scope_name, "split": "all", "relevance": name, "infractions": group, **row}
                                          for row in contrast(values_of[scope_name][name], best, other, group_mask, cfg, seed)]
        contrasts = pd.DataFrame(contrast_rows)
        contrasts = contrasts[contrasts["metric"].isin(["nDCG@10", "MAP", "R@5", "mAR", "P@10"])].reset_index(drop=True)

        # ---- compression run again (the 50 reports of scripts/45): neighbours kept per candidate
        stability, k = {}, methods_cfg["neighbours"]
        rerun_dirs = sorted(resolve(paths["data"]["v2"]).glob("compression_stability_*"))
        if rerun_dirs and "compressed" in parts:
            hashes = (rerun_dirs[-1] / "hashes_v2.txt").read_text(encoding="utf-8").split()
            position = {h: i for i, h in enumerate(labels["acordao_hash"])}
            rows50 = [position[h] for h in hashes]
            rerun = np.load(rerun_dirs[-1] / "embeddings_v2.npy")
            for name, weights in cfg["candidates"].items():
                if "compressed" not in weights:
                    stability[name] = float(k)                          # no compression in the vector
                    continue
                again = K.combine({**{p: parts[p][rows50] for p in weights if p != "compressed"}, "compressed": rerun}, weights)
                stability[name] = neighbours_kept(vectors[name], again, rows50, k)

        summary = {
            "rule": "highest development score; it replaces the current vectors only if the bootstrap 95% interval of the "
                    "paired score difference is above zero",
            "scope": scope, "metric": sel["metric"], "families": families,
            "queries": {split: {d: int((~np.isnan(values_of[scope][d][current][sel["metric"]]) & masks_of[scope][split]).sum())
                                for d in definitions} for split in SPLITS},
            "development_scores": scores, "best_on_development": best, "current": current,
            "development_difference_to_current": None if interval is None else dict(zip(("difference", "ci_low", "ci_high"), interval)),
            "replace_current": replace, "selected": selected,
            "held_out_scores": held_scores,
            "held_out_difference_to_current": None if held_interval is None else dict(zip(("difference", "ci_low", "ci_high"), held_interval)),
            "best_on_held_out": max(held_scores, key=held_scores.get),
            f"neighbours_kept_of_{k}_after_a_second_compression": stability,
        }

        for path, frame, note in ((results / "a1_candidates.csv", table, "A1 (4.1): retrieval metrics per candidate, split, scope and relevance"),
                                  (results / "a1_candidates_contrasts.csv", contrasts, "A1 (4.1): the best candidate against the current vectors, the full report and version 1")):
            path = new_artefact_path(path)
            frame.to_csv(path, index=False, float_format="%.6g")
            run.add_output(path, note)
        path = new_artefact_path(results / "a1_candidates_selection.json")
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run.add_output(path, "A1 (4.1): selection scores, the rule's outcome and the held-out check")

        shown = [*candidates, *references]
        names = {c: c.replace("_", " ") for c in candidates} | {r: f"{r}  {methods_cfg['methods'][r]['label'].split(',')[0]}" for r in references}
        names[current] += " (current)"
        if best != current:
            names[best] += " (best on development)"
        marks = {current: (BLUE, "D"), "M3": (ORANGE, "s")} | ({best: (INK, "*")} if best != current else {})
        held_ndcg = (table["scope"] == scope) & (table["split"] == "held_out") & (table["metric"] == sel["metric"])
        fig_path = new_artefact_path(results / "figures" / "a1_candidates.pdf")
        figure(table, [(f"{d}\n{summary['queries']['held_out'][d]:,} queries", held_ndcg & (table["relevance"] == d)) for d in definitions],
               shown, names, marks, "Candidates for version 2's retrieval vector: held-out queries, 2,390 decisions", fig_path)
        run.add_output(fig_path, "A1 (4.1) figure: nDCG@10 of the candidates on the held-out queries")

        run.data["summary"] = summary
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
