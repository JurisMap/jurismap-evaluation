"""Phase 5, A2: how well each 2D projection keeps the neighbourhoods of the embedding space.

Reads the projections fitted by scripts/50_a2_projections.py and, for the sets that have it,
version 2's map as served (the coordinates stored with the frozen corpus). The original space is
the cosine distance between a set's vectors; the map is the Euclidean distance between its points.

Per projection and seed (config/a2_map.yaml):
  trustworthiness and continuity at k = 5, 10, 20, 50; Jaccard overlap of the 10 nearest
  neighbours; Spearman correlation of the pairwise distances on 100,000 pairs (the same pairs for
  every projection of a set); silhouette of each labelling, and the share of a decision's 10 nearest
  neighbours with its label (decisions of the same process left out, as in A1).
The label metrics are also computed in the original space, next to the share expected by chance.

Seed stability: Procrustes disparity and overlap of the 10 nearest map neighbours between every
two fits of a projection; the same between the served map and each refit of version 2's setting.

Outputs (never overwritten):
  results/a2_map_metrics.csv        mean, SD, minimum and maximum over the seeds, per set, projection and metric
  results/a2_seed_stability.csv     agreement between the fits of each projection
  results/a2_map_summary.json       where version 2's setting stands in the UMAP grid; label counts
  results/figures/a2_trustworthiness_vs_k.pdf, a2_umap_grid.pdf
  data/v2/a2_runs/a2_metrics_by_seed.parquet   every fit's metrics
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

from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve  # noqa: E402
from jm_eval.manifest import RunManifest  # noqa: E402
from jm_eval.mapping import metrics as M  # noqa: E402
from jm_eval.mapping import projections as P  # noqa: E402
from jm_eval.retrieval import corpus as C  # noqa: E402
from jm_eval.retrieval import metrics as retrieval_metrics  # noqa: E402

SCRIPT = "51_a2_map_metrics"
FROZEN = "frozen_v2"
# Figure colours as in the A1 figures: the two versions carry a hue, every other projection is
# neutral; marker shapes and line styles repeat the distinction.
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#6b6a66"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # one hue, light to dark


def load_fits(name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = latest_artefact_path(resolve(load_config("paths")["data"]["v2"]) / "a2_projections" / f"{name}.npz")
    if not path.exists():
        raise FileNotFoundError(f"{path}: run scripts/50_a2_projections.py")
    with np.load(path) as z:
        return z["hashes"], {k: z[k] for k in z.files if k != "hashes"}


def label_metrics(dist: np.ndarray, allowed: np.ndarray, codes: dict[str, np.ndarray], k: int) -> dict[str, float]:
    out = {f"silhouette/{name}": M.silhouette(dist, c) for name, c in codes.items()}
    out.update({f"same_label@{k}/{name}": v for name, v in M.same_label_share(dist, allowed, codes, k).items()})
    return out


def summarise(by_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["set", "projection", "metric"]
    table = by_seed.groupby(keys, sort=False)["value"].agg(fits="count", mean="mean", sd="std", min="min", max="max").reset_index()
    return table


def grid_standing(table: pd.DataFrame, specs: list[dict], versions: dict[str, str], headline: list[str]) -> dict:
    """Per set and headline metric: where each version's setting stands among the UMAP settings."""
    grid = [s["id"] for s in specs if s["family"] == "umap"]
    out: dict = {}
    for (name, metric), sub in table[table["projection"].isin(grid) & table["metric"].isin(headline)].groupby(["set", "metric"], sort=False):
        sub = sub.sort_values("mean", ascending=False).reset_index(drop=True)
        best = sub.iloc[0]
        entry = {"best": {"projection": best["projection"], "mean": float(best["mean"]), "sd": float(best["sd"])},
                 "worst": {"projection": sub.iloc[-1]["projection"], "mean": float(sub.iloc[-1]["mean"])}}
        for version, projection in versions.items():
            row = sub[sub["projection"] == projection].iloc[0]
            entry[version] = {"projection": projection, "rank": int(sub.index[sub["projection"] == projection][0]) + 1,
                              "settings": len(sub),
                              "mean": float(row["mean"]), "sd": float(row["sd"]), "min": float(row["min"]), "max": float(row["max"]),
                              "below_best": float(best["mean"] - row["mean"])}
        out.setdefault(name, {})[metric] = entry
    return out


# ---------------------------------------------------------------------------- figures

def _axes_style(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def k_figure(table: pd.DataFrame, cfg: dict, series: list[tuple[str, str, str, str, str]], path: Path) -> None:
    """Trustworthiness (top) and continuity (bottom) against k, one panel per embedding set."""
    ks, sets = cfg["metrics"]["neighbourhood_k"], list(cfg["sets"])
    fig, axes = plt.subplots(2, len(sets), figsize=(3.3 * len(sets) + 0.6, 6.0), sharex=True, sharey="row", squeeze=False)
    indexed = table.set_index(["set", "projection", "metric"])
    for row, metric in enumerate(("trustworthiness", "continuity")):
        for col, name in enumerate(sets):
            ax = axes[row, col]
            for projection, label, colour, marker, line in series:
                values = indexed.loc[[(name, projection, f"{metric}@{k}") for k in ks]]
                ax.fill_between(ks, values["min"], values["max"], color=colour, alpha=0.18, linewidth=0)
                ax.plot(ks, values["mean"], color=colour, linestyle=line, linewidth=1.8, marker=marker, markersize=6.5,
                        markeredgecolor="white", markeredgewidth=1.0, label=label)
            ax.set_xscale("log")
            ax.set_xticks(ks, [str(k) for k in ks])
            ax.minorticks_off()
            _axes_style(ax)
            if row == 0:
                ax.set_title(f"{cfg['sets'][name]['label']}\n{int(indexed.loc[(name, series[0][0], f'{metric}@{ks[0]}'), 'decisions']):,} decisions",
                             fontsize=9, color=INK, loc="left")
            if row == 1:
                ax.set_xlabel("k (nearest neighbours)", fontsize=8, color=MUTED)
        axes[row, 0].set_ylabel(metric.capitalize(), fontsize=9, color=INK)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(series), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.01, 0.008, "Mean over five seeds; the band spans the lowest and highest seed. PCA has no seed. "
             "The two rows have different scales.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    fig.savefig(path)
    plt.close(fig)


def grid_figure(table: pd.DataFrame, cfg: dict, versions: dict[str, str], metrics: list[str], path: Path) -> None:
    """The UMAP grid as heatmaps: one row per metric (one colour scale per row), one column per set."""
    neighbours, dists, sets = cfg["projections"]["umap"]["n_neighbors"], cfg["projections"]["umap"]["min_dist"], list(cfg["sets"])
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL)
    indexed = table.set_index(["set", "projection", "metric"])["mean"]
    fig, axes = plt.subplots(len(metrics), len(sets), figsize=(3.0 * len(sets) + 1.4, 2.5 * len(metrics) + 0.9), squeeze=False)
    for row, metric in enumerate(metrics):
        cells = {name: np.array([[indexed[(name, P.umap_id(n, d), metric)] for d in dists] for n in neighbours]) for name in sets}
        norm = Normalize(min(c.min() for c in cells.values()), max(c.max() for c in cells.values()))
        for col, name in enumerate(sets):
            ax = axes[row, col]
            image = ax.imshow(cells[name], cmap=cmap, norm=norm, aspect="auto")
            for i, n in enumerate(neighbours):
                for j, d in enumerate(dists):
                    value = cells[name][i, j]
                    tag = next((f"\n{v}" for v, projection in versions.items() if projection == P.umap_id(n, d)), "")
                    ax.text(j, i, f"{value:.3f}{tag}", ha="center", va="center", fontsize=7.5, linespacing=1.3,
                            color="white" if norm(value) > 0.55 else INK)
                    if tag:
                        ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, linewidth=1.4))
            ax.set_xticks(range(len(dists)), [f"{d:g}" for d in dists])
            ax.set_yticks(range(len(neighbours)), [str(n) for n in neighbours])
            ax.tick_params(length=0, colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            # a surface-coloured gap between the cells
            ax.set_xticks(np.arange(-0.5, len(dists)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(neighbours)), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", length=0)
            if row == 0:
                ax.set_title(cfg["sets"][name]["label"], fontsize=9, color=INK, loc="left")
            if row == len(metrics) - 1:
                ax.set_xlabel("min_dist", fontsize=8, color=MUTED)
            if col == 0:
                ax.set_ylabel(f"{metric}\n\nn_neighbors", fontsize=8.5, color=INK)
        bar = fig.colorbar(image, ax=axes[row, :].tolist(), fraction=0.025, pad=0.015)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, colors=MUTED, labelsize=7.5)
    fig.text(0.01, 0.012, "Mean over five seeds. Outlined cells: the settings of version 1 (v1) and version 2 (v2). "
             "Each row has its own colour scale.", fontsize=7.5, color=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", help="write everything here instead of results/ and data/v2/a2_runs/ (trial runs)")
    args = ap.parse_args()

    cfg, paths, seeds = P.settings(), load_config("paths"), load_config("seeds")
    mcfg = cfg["metrics"]
    ks, jaccard_k, label_k = mcfg["neighbourhood_k"], mcfg["jaccard_k"], mcfg["label_neighbours"]
    specs, versions = P.specs(cfg), P.version_ids(cfg)
    spec_of = {s["id"]: s for s in specs}
    results = Path(args.out_dir) if args.out_dir else resolve(paths["results"])
    runs_dir = Path(args.out_dir) if args.out_dir else resolve(paths["data"]["v2"]) / "a2_runs"
    figures = results / "figures"
    for d in (runs_dir, figures):
        d.mkdir(parents=True, exist_ok=True)
    params = {"sets": cfg["sets"], "projections": cfg["projections"], "version_settings": cfg["version_settings"],
              "frozen": cfg["frozen"], "metrics": mcfg}
    with RunManifest(SCRIPT, results, models={m: C.methods_config()["methods"][m] for m in {s["method"] for s in cfg["sets"].values()}},
                     params=params, seeds={"a2_projections": seeds["a2_projections"], "a2_distance_pairs": seeds["a2_distance_pairs"]},
                     record_outputs=args.out_dir is None) as run:
        labels = C.load_labels()
        run.add_input(resolve(load_config("a1_labels")["corpus"]))
        frozen_all = np.vstack(C.load_corpus([cfg["frozen"]["column"]])[cfg["frozen"]["column"]].to_numpy())

        seed_rows, stability_rows, label_counts, frozen_vs_refit = [], [], {}, {}
        for name, spec in cfg["sets"].items():
            rows = P.set_rows(cfg, name, labels["in_comparison_set"].to_numpy())
            sub = labels.iloc[rows].reset_index(drop=True)
            hashes, fits = load_fits(name)
            assert hashes.tolist() == sub["acordao_hash"].tolist(), f"{name}: projections and corpus are not aligned"
            run.add_input(latest_artefact_path(resolve(paths["data"]["v2"]) / "a2_projections" / f"{name}.npz"))
            original = M.Space(M.cosine_distances(C.method_vectors(spec["method"])[rows]))
            pairs = M.sample_pairs(len(rows), mcfg["distance_pairs"], seeds["a2_distance_pairs"])
            allowed = retrieval_metrics.candidate_mask(sub["process_group"].to_numpy())
            codes = {label: M.class_codes(sub[column].to_numpy()) for label, column in mcfg["labels"].items()}
            label_counts[name] = {label: {"decisions": int((c >= 0).sum()), "classes": int(len(np.unique(c[c >= 0])))}
                                  for label, c in codes.items()}

            def add(projection: str, seed, values: dict[str, float]) -> None:
                seed_rows.extend({"set": name, "projection": projection, "seed": seed, "metric": metric, "value": value}
                                 for metric, value in values.items())

            reference = label_metrics(original.dist, allowed, codes, label_k)
            reference.update({f"same_label_chance@{label_k}/{label}": M.chance_share(allowed, c) for label, c in codes.items()})
            add("original", None, reference)

            if name in cfg["frozen"]["sets"]:
                fits[f"{FROZEN}__{seeds['umap_v2']}"] = frozen_all[rows]
            near: dict[str, dict[str, np.ndarray]] = {}
            for key, coords in fits.items():
                projection, seed = key.split("__")
                projected = M.Space(M.euclidean_distances(coords))
                values = M.neighbourhood_metrics(original, projected, ks, jaccard_k, pairs)
                values.update(label_metrics(projected.dist, allowed, codes, label_k))
                add(projection, None if seed == "none" else int(seed), values)
                near.setdefault(projection, {})[seed] = M.nearest(projected, jaccard_k)
                print(name, key, {m: round(values[m], 4) for m in mcfg["headline"]}, flush=True)

            for projection, by_seed in near.items():
                if len(by_seed) < 2:
                    continue
                pairs_of_fits = list(itertools.combinations(by_seed, 2))
                disparity = [M.procrustes_disparity(fits[f"{projection}__{a}"], fits[f"{projection}__{b}"]) for a, b in pairs_of_fits]
                overlap = [M.neighbour_jaccard(by_seed[a], by_seed[b]) for a, b in pairs_of_fits]
                stability_rows.append({"set": name, "projection": projection, "fits": len(by_seed), "pairs": len(pairs_of_fits),
                                       "procrustes_mean": np.mean(disparity), "procrustes_min": np.min(disparity),
                                       "procrustes_max": np.max(disparity), f"map_knn_jaccard@{jaccard_k}_mean": np.mean(overlap),
                                       f"map_knn_jaccard@{jaccard_k}_min": np.min(overlap),
                                       f"map_knn_jaccard@{jaccard_k}_max": np.max(overlap)})
            if FROZEN in near:                                  # the served map against each refit of version 2's setting
                served = near[FROZEN][str(seeds["umap_v2"])]
                frozen_vs_refit[name] = {
                    seed: {"procrustes": M.procrustes_disparity(frozen_all[rows], fits[f"{versions['v2']}__{seed}"]),
                           f"map_knn_jaccard@{jaccard_k}": M.neighbour_jaccard(served, refit)}
                    for seed, refit in near[versions["v2"]].items()}

        by_seed = pd.DataFrame(seed_rows)
        table = summarise(by_seed)

        def describe(frame: pd.DataFrame) -> pd.DataFrame:
            """Columns that say what each row's set and projection are."""
            frame = frame.copy()
            frame["method"] = frame["set"].map(lambda s: cfg["sets"][s]["method"])
            frame["scope"] = frame["set"].map(lambda s: cfg["sets"][s]["scope"])
            frame["decisions"] = frame["set"].map(lambda s: len(P.set_rows(cfg, s, labels["in_comparison_set"].to_numpy())))
            frame["family"] = frame["projection"].map(lambda p: spec_of[p]["family"] if p in spec_of else p)
            frame["n_neighbors"] = frame["projection"].map(lambda p: spec_of.get(p, {}).get("params", {}).get("n_neighbors"))
            frame["min_dist"] = frame["projection"].map(lambda p: spec_of.get(p, {}).get("params", {}).get("min_dist"))
            setting = {projection: version for version, projection in versions.items()} | {FROZEN: "v2 (served map)"}
            frame["version_setting"] = frame["projection"].map(setting)
            lead = ["set", "method", "scope", "decisions", "projection", "family", "n_neighbors", "min_dist", "version_setting"]
            return frame[lead + [c for c in frame.columns if c not in lead]]

        table, stability = describe(table), describe(pd.DataFrame(stability_rows))
        standing = grid_standing(table, specs, versions, mcfg["headline"])
        summary = {"labels": label_counts, "umap_grid": standing, "served_map_vs_refits_of_v2_setting": frozen_vs_refit,
                   "fits": {name: int(by_seed[(by_seed["set"] == name) & (by_seed["projection"] != "original")]
                                      .drop_duplicates(["projection", "seed"]).shape[0]) for name in cfg["sets"]}}

        for path, frame, note in ((results / "a2_map_metrics.csv", table, "A2: map metrics per set and projection (mean, SD, range over seeds)"),
                                  (results / "a2_seed_stability.csv", stability, "A2: agreement between the fits of each projection")):
            path = new_artefact_path(path)
            frame.to_csv(path, index=False, float_format="%.6g")
            run.add_output(path, note)
        summary_path = new_artefact_path(results / "a2_map_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(summary_path, "A2: version settings within the UMAP grid; served map against refits; label counts")
        by_seed_path = new_artefact_path(runs_dir / "a2_metrics_by_seed.parquet")
        by_seed.assign(seed=by_seed["seed"].astype("Int64")).to_parquet(by_seed_path, index=False)
        run.add_output(by_seed_path, "A2: metrics of every fit")

        tsne, pacmap = (next(s["id"] for s in specs if s["family"] == f) for f in ("tsne", "pacmap"))
        v1, v2 = cfg["version_settings"]["v1"], cfg["version_settings"]["v2"]
        series = [(versions["v2"], f"UMAP {v2['n_neighbors']} / {v2['min_dist']:g} (version 2)", "#2a78d6", "D", "-"),
                  (versions["v1"], f"UMAP {v1['n_neighbors']} / {v1['min_dist']:g} (version 1)", "#eb6834", "s", "-"),
                  (tsne, "t-SNE", NEUTRAL, "o", "-"), (pacmap, "PaCMAP", NEUTRAL, "^", "--"), ("pca", "PCA", NEUTRAL, "X", ":")]
        fig_path = new_artefact_path(figures / "a2_trustworthiness_vs_k.pdf")
        k_figure(table, cfg, series, fig_path)
        run.add_output(fig_path, "A2 figure: trustworthiness and continuity against k")
        fig_path = new_artefact_path(figures / "a2_umap_grid.pdf")
        grid_figure(table, cfg, versions, [f"trustworthiness@{jaccard_k}", f"continuity@{jaccard_k}"], fig_path)
        run.add_output(fig_path, "A2 figure: trustworthiness and continuity at k = 10 over the UMAP grid")

        run.data["summary"] = summary
        show = table[table["metric"].isin(mcfg["headline"]) & (table["version_setting"].notna() | ~table["family"].isin(["umap"]))]
        print(show.pivot(index=["set", "projection"], columns="metric", values="mean").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
