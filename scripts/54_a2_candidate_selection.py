"""Phase 5.1, A2: score the UMAP candidates for version 2's map and apply the selection rule.

Reads the fits of scripts/53_a2_candidate_fits.py. Per candidate (config/a2_candidates.yaml):
  fitted map, mean and SD over the seeds   trustworthiness and continuity at k = 10, kNN Jaccard at 10,
                                           Spearman correlation of pairwise distances; same-label
                                           shares and the gap between neighbouring points (reported only)
  placed decisions, mean and SD over folds the same three neighbourhood metrics for held-out decisions
                                           projected onto a map fitted without them
  between seeds                            Procrustes disparity, overlap of the 10 nearest map neighbours
  repeat                                   whether a second fit with production's seed is identical

Selection: every metric of `selection.metrics` is standardised over the grid (z-score) and the
candidate with the highest mean wins. The references (t-SNE, PaCMAP) are scored with the grid's
means and SDs and cannot be selected.

With --refine ID: the same metrics for that candidate fitted for longer, scored on the grid's scale.

Outputs (never overwritten):
  results/a2_candidates.csv, a2_candidates_selection.json, results/figures/a2_candidates.pdf
  results/a2_candidates_refine.csv            (--refine)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve  # noqa: E402
from jm_eval.manifest import RunManifest  # noqa: E402
from jm_eval.mapping import metrics as M  # noqa: E402
from jm_eval.mapping import projections as P  # noqa: E402
from jm_eval.retrieval import corpus as C  # noqa: E402
from jm_eval.retrieval import metrics as retrieval_metrics  # noqa: E402

SCRIPT = "54_a2_candidate_selection"
CONFIG = "a2_candidates"
INK, MUTED = "#0b0b0b", "#52514e"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # one hue, light to dark


def score_table(table: pd.DataFrame, metrics: list[str], scale: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `z_<metric>`, `score` (their mean) and `rank` (candidates only, 1 = selected).

    The z-scores use the mean and SD of the candidates of `table`, or `scale` (index: metric;
    columns: mean, sd) when given. Returns the table and the scale used.
    """
    table = table.copy()
    grid = table[table["candidate"]]
    if scale is None:
        scale = pd.DataFrame({"mean": grid[metrics].mean(), "sd": grid[metrics].std(ddof=0)})
    for metric in metrics:
        table[f"z_{metric}"] = (table[metric] - scale.loc[metric, "mean"]) / scale.loc[metric, "sd"]
    table["score"] = table[[f"z_{m}" for m in metrics]].mean(axis=1)
    table["rank"] = table.loc[table["candidate"], "score"].rank(ascending=False, method="min")
    return table, scale


def select(table: pd.DataFrame, current: dict) -> pd.Series:
    """The candidate with the highest score; a tie goes to the setting nearest the current one."""
    grid = table[table["candidate"]]
    best = grid[np.isclose(grid["score"], grid["score"].max())]
    distance = (np.log(best["n_neighbors"] / current["n_neighbors"]).abs() + (best["min_dist"] - current["min_dist"]).abs())
    return best.loc[distance.idxmin()]


def grid_figure(table: pd.DataFrame, cfg: dict, metrics: list[str], marked: dict[str, str], path: Path) -> None:
    """The selection score and each metric over the grid: small heatmaps, each with its own scale."""
    u = cfg["candidates"]["umap"]
    neighbours, dists = u["n_neighbors"], u["min_dist"]
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL)
    indexed = table.set_index("projection")
    panels = [("score", "Selection score (mean z)")] + [(m, m.replace("holdout_", "placed: ").replace("_", " ")) for m in metrics]
    columns = 4
    fig, axes = plt.subplots(int(np.ceil(len(panels) / columns)), columns, figsize=(3.3 * columns, 3.0 * int(np.ceil(len(panels) / columns)) + 0.5))
    for ax, (column, title) in zip(axes.ravel(), panels):
        cells = np.array([[indexed.loc[P.umap_id(n, d), column] for d in dists] for n in neighbours])
        norm = Normalize(cells.min(), cells.max())
        ax.imshow(cells, cmap=cmap, norm=norm, aspect="auto")
        for i, n in enumerate(neighbours):
            for j, d in enumerate(dists):
                tag = marked.get(P.umap_id(n, d))
                ax.text(j, i, f"{cells[i, j]:.2f}" if column == "score" else f"{cells[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if norm(cells[i, j]) > 0.55 else INK, fontweight="bold" if tag else "normal")
                if tag:
                    ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, linewidth=1.4,
                                           linestyle="-" if tag == "selected" else "--"))
        ax.set_xticks(range(len(dists)), [f"{d:g}" for d in dists])
        ax.set_yticks(range(len(neighbours)), [str(n) for n in neighbours])
        ax.tick_params(length=0, colors=MUTED, labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(dists)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(neighbours)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("min_dist", fontsize=7.5, color=MUTED)
        ax.set_ylabel("n_neighbors", fontsize=7.5, color=MUTED)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.text(0.01, 0.005, "Darker is better in every panel; each panel has its own colour scale. Solid outline: the selected setting. "
             "Dashed outline: the current one. Fitted map: mean over five seeds. Placed: mean over five held-out folds.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refine", metavar="ID", help="score this candidate's longer fits on the grid's scale")
    ap.add_argument("--out-dir", help="write everything here instead of results/ (trial runs)")
    args = ap.parse_args()

    cfg, map_cfg, paths, seeds = load_config(CONFIG), P.settings(), load_config("paths"), load_config("seeds")
    mcfg, selection = map_cfg["metrics"], cfg["selection"]["metrics"]
    k, label_k = mcfg["jaccard_k"], mcfg["label_neighbours"]
    grid_specs = P.candidate_specs(cfg)
    specs = grid_specs
    if args.refine:
        base = next(s for s in grid_specs if s["id"] == args.refine)
        specs = [base, *P.refined_specs(cfg, base)]
    results = Path(args.out_dir) if args.out_dir else resolve(paths["results"])
    (results / "figures").mkdir(parents=True, exist_ok=True)
    method = cfg.get("method") or map_cfg["sets"][cfg["set"]]["method"]
    fits_dir = resolve(paths["data"]["v2"]) / "a2_candidates"
    with RunManifest(SCRIPT, results, models={method: C.methods_config()["methods"][method]},
                     params={"candidates": cfg, "metrics": mcfg, "refine": args.refine},
                     seeds={key: seeds[key] for key in ("a2_projections", "umap_v2", "a2_distance_pairs", "a2_holdout_folds")},
                     record_outputs=args.out_dir is None) as run:
        labels = C.load_labels()
        rows = P.set_rows(map_cfg, cfg["set"], labels["in_comparison_set"].to_numpy())
        sub = labels.iloc[rows].reset_index(drop=True)
        fits: dict[str, np.ndarray] = {}
        for name in ("a2_candidates.npz", *(["a2_candidates_refine.npz"] if args.refine else [])):
            fits_path = latest_artefact_path(fits_dir / name)
            run.add_input(fits_path)
            with np.load(fits_path) as z:
                fits.update({key: z[key] for key in z.files})
        assert fits["hashes"].tolist() == sub["acordao_hash"].tolist(), "fits and corpus are not aligned"
        vectors = C.method_vectors(method)[rows]
        original = M.Space(M.cosine_distances(vectors))
        pairs = M.sample_pairs(len(rows), mcfg["distance_pairs"], seeds["a2_distance_pairs"])
        allowed = retrieval_metrics.candidate_mask(sub["process_group"].to_numpy())
        codes = {label: M.class_codes(sub[column].to_numpy()) for label, column in mcfg["labels"].items()}
        held_out = [fits[f"fold_{f}"] for f in range(cfg["holdout"]["folds"])]
        kept = [np.setdiff1d(np.arange(len(rows)), h) for h in held_out]

        records = []
        for spec in specs:
            projection, by_seed, near, coords_of = spec["id"], [], {}, {}
            for seed in seeds["a2_projections"]:
                coords = coords_of[seed] = fits[f"{projection}__{seed}"]
                space = M.Space(M.euclidean_distances(coords))
                values = M.neighbourhood_metrics(original, space, [k], k, pairs)
                values.update({f"same_label@{label_k}/{label}": v for label, v in M.same_label_share(space.dist, allowed, codes, label_k).items()})
                values["nearest_point_gap"] = M.nearest_point_gap(space)
                by_seed.append(values)
                near[seed] = M.nearest(space, k)
            by_fold = [M.holdout_metrics(original.dist[np.ix_(h, t)], cdist(fits[f"{projection}__fold{f}_placed"].astype(np.float64),
                                                                              fits[f"{projection}__fold{f}_fitted"].astype(np.float64)), k)
                       for f, (h, t) in enumerate(zip(held_out, kept))]
            seed_pairs = list(itertools.combinations(seeds["a2_projections"], 2))
            record = {"projection": projection, "family": spec["family"], "candidate": spec["candidate"],
                      "n_neighbors": spec["params"].get("n_neighbors"), "min_dist": spec["params"].get("min_dist"),
                      "n_epochs": spec["params"].get("n_epochs")}
            for frame in (pd.DataFrame(by_seed), pd.DataFrame(by_fold)):
                record.update(frame.mean().to_dict())
                record.update({f"{name}_sd": value for name, value in frame.std(ddof=1).items()})
            record["procrustes_between_seeds"] = float(np.mean([M.procrustes_disparity(coords_of[a], coords_of[b]) for a, b in seed_pairs]))
            record[f"map_knn_jaccard@{k}_between_seeds"] = float(np.mean([M.neighbour_jaccard(near[a], near[b]) for a, b in seed_pairs]))
            record["repeat_identical"] = bool(np.array_equal(coords_of[seeds["umap_v2"]], fits[f"{projection}__{seeds['umap_v2']}_repeat"]))
            records.append(record)
            print(projection, {m: round(record[m], 4) for m in selection}, flush=True)
        table = pd.DataFrame(records)

        if args.refine:
            reference = pd.read_csv(latest_artefact_path(results / "a2_candidates.csv"))
            grid = reference[reference["candidate"]]
            scale = pd.DataFrame({"mean": grid[selection].mean(), "sd": grid[selection].std(ddof=0)})
            table, _ = score_table(table, selection, scale)
            path = new_artefact_path(results / "a2_candidates_refine.csv")
            table.drop(columns=["rank"]).to_csv(path, index=False, float_format="%.6g")
            run.add_output(path, f"A2 (5.1): {args.refine} fitted for longer, scored on the grid's scale")
            run.data["summary"] = {"refine": args.refine, "scores": dict(zip(table["projection"], table["score"].round(4)))}
            print(table[["projection", "n_epochs", "score", *selection]].round(4).to_string())
            return 0

        table, scale = score_table(table, selection)
        chosen, current = select(table, cfg["current"]), table[table["projection"] == P.umap_id(**cfg["current"])].iloc[0]
        table = table.sort_values(["candidate", "score"], ascending=[False, False]).reset_index(drop=True)
        path = new_artefact_path(results / "a2_candidates.csv")
        table.to_csv(path, index=False, float_format="%.6g")
        run.add_output(path, "A2 (5.1): UMAP candidates and references, metrics and selection score")

        def brief(row: pd.Series) -> dict:
            return {"projection": row["projection"], "n_neighbors": int(row["n_neighbors"]), "min_dist": float(row["min_dist"]),
                    "score": float(row["score"]), "rank": int(row["rank"]), **{m: float(row[m]) for m in selection}}

        summary = {
            "rule": "highest mean z-score over the selection metrics, standardised over the grid; labels take no part",
            "selection_metrics": selection, "candidates": int(table["candidate"].sum()),
            "selected": brief(chosen), "current": brief(current),
            "selected_minus_current": {m: {"difference": float(chosen[m] - current[m]),
                                           "sd_selected": float(chosen[f"{m}_sd"]), "sd_current": float(current[f"{m}_sd"])} for m in selection},
            "top_5": [brief(r) for _, r in table[table["candidate"]].head(5).iterrows()],
            "all_seeded_fits_repeat": bool(table["repeat_identical"].all()),
            "fits_that_do_not_repeat": table.loc[~table["repeat_identical"], "projection"].tolist(),
            "scale": scale.to_dict("index"),
        }
        path = new_artefact_path(results / "a2_candidates_selection.json")
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run.add_output(path, "A2 (5.1): the selected UMAP setting and the current one")
        fig_path = new_artefact_path(results / "figures" / "a2_candidates.pdf")
        grid_figure(table, cfg, selection, {chosen["projection"]: "selected", current["projection"]: "current"}, fig_path)
        run.add_output(fig_path, "A2 (5.1) figure: selection score and metrics over the UMAP grid")
        run.data["summary"] = summary
        print(table[["projection", "candidate", "score", "rank", *selection]].round(4).to_string())
        print(json.dumps({"selected": summary["selected"], "current": summary["current"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
