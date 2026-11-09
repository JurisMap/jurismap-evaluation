import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness
from sklearn.metrics import silhouette_score

from jm_eval.io import load_config
from jm_eval.mapping import metrics as M
from jm_eval.mapping import projections as P
from jm_eval.retrieval import metrics as retrieval_metrics

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _spaces(n=80, dims=12, seed=0):
    rng = np.random.default_rng(seed)
    vectors, coords = rng.normal(size=(n, dims)), rng.normal(size=(n, 2))
    return vectors, coords, M.Space(M.cosine_distances(vectors)), M.Space(M.euclidean_distances(coords))


# ---------------------------------------------------------------- neighbourhood metrics
@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_trustworthiness_and_continuity_match_sklearn(k):
    vectors, coords, original, projected = _spaces()
    assert M.trustworthiness(original, projected, k) == pytest.approx(
        sk_trustworthiness(vectors, coords, n_neighbors=k, metric="cosine"))
    # continuity: the roles swapped. On unit vectors the Euclidean order is the cosine order.
    assert M.trustworthiness(projected, original, k) == pytest.approx(
        sk_trustworthiness(coords, P.unit(vectors), n_neighbors=k, metric="euclidean"))


def test_a_map_that_keeps_every_neighbourhood_scores_one():
    coords = np.random.default_rng(1).normal(size=(60, 2))
    a = M.Space(M.euclidean_distances(coords))
    b = M.Space(M.euclidean_distances(coords @ np.array([[0.0, -1.0], [1.0, 0.0]]) * 3.0 + 5.0))   # turned, scaled, moved
    pairs = M.sample_pairs(60, 500, seed=3)
    values = M.neighbourhood_metrics(a, b, [5, 10], 10, pairs)
    assert all(v == pytest.approx(1.0) for v in values.values())
    assert set(values) == {"trustworthiness@5", "continuity@5", "trustworthiness@10", "continuity@10",
                           "knn_jaccard@10", "distance_spearman"}
    with pytest.raises(ValueError):
        M.trustworthiness(a, b, 30)                                            # k must stay below n / 2


def test_neighbour_order_ranks_and_jaccard_on_a_hand_checked_line():
    original = M.Space(M.euclidean_distances(np.array([[0.0, 0], [1, 0], [2, 0], [3, 0], [10, 0]])))
    assert original.order[0].tolist() == [1, 2, 3, 4] and original.order[2].tolist() == [1, 3, 0, 4]   # ties keep the order
    assert original.ranks[0].tolist() == [0, 1, 2, 3, 4]
    moved = M.Space(M.euclidean_distances(np.array([[0.0, 0], [1, 0], [2, 0], [3, 0], [0, 0.5]])))   # the last point lands on the first
    near_a, near_b = M.nearest(original, 2), M.nearest(moved, 2)
    assert near_a[0].tolist() == [False, True, True, False, False]
    assert near_b[0].tolist() == [False, True, False, False, True]
    # overlaps: point 0 shares 1 of 3 ({1,2} against {4,1}), points 1 to 3 keep theirs, point 4 has
    # {3,2} against {0,1}
    assert M.knn_jaccard(original, moved, 2) == pytest.approx((1 / 3 + 1 + 1 + 1 + 0) / 5)
    assert M.trustworthiness(original, moved, 1) < 1.0


def test_sample_pairs_is_seeded_and_distinct():
    i, j = M.sample_pairs(50, 300, seed=7)
    assert len(i) == 300 and np.all(i < j) and len(set(zip(i.tolist(), j.tolist()))) == 300
    again = M.sample_pairs(50, 300, seed=7)
    assert np.array_equal(i, again[0]) and np.array_equal(j, again[1])
    assert len(M.sample_pairs(10, 1000, seed=7)[0]) == 45                       # fewer pairs than asked: all of them
    d = M.euclidean_distances(np.random.default_rng(0).normal(size=(50, 2)))
    assert M.distance_spearman(d, d ** 2, (i, j)) == pytest.approx(1.0)        # rank correlation


# ---------------------------------------------------------------- labels
def test_class_codes_drop_missing_labels_and_single_decisions():
    codes = M.class_codes(np.array(["a", "b", "a", None, "c", "b", float("nan")], dtype=object))
    assert codes[[3, 4, 6]].tolist() == [-1, -1, -1]                           # missing, alone, missing
    assert codes[0] == codes[2] != codes[1] == codes[5] and codes[0] >= 0


def test_silhouette_matches_sklearn_on_the_labelled_decisions():
    rng = np.random.default_rng(2)
    coords = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(3, 0.3, (20, 2)), rng.normal(9, 1, (3, 2))])
    values = np.array(["a"] * 20 + ["b"] * 20 + ["x", None, "y"], dtype=object)
    codes = M.class_codes(values)
    assert M.silhouette(M.euclidean_distances(coords), codes) == pytest.approx(silhouette_score(coords[:40], codes[:40]))
    assert np.isnan(M.silhouette(M.euclidean_distances(coords), M.class_codes(np.array(["a"] * 43, dtype=object))))


def test_same_label_share_agrees_with_the_retrieval_benchmark():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(60, 8))
    classes = rng.choice(np.array(["a", "b", "c"], dtype=object), size=60)
    allowed = retrieval_metrics.candidate_mask(rng.integers(0, 40, size=60))    # some decisions share a process
    expected = retrieval_metrics.neighbour_agreement(retrieval_metrics.cosine_similarity(vectors), allowed, classes, k=10)
    codes = M.class_codes(classes)
    share = M.same_label_share(M.cosine_distances(vectors), allowed, {"label": codes}, 10)["label"]
    assert share == pytest.approx(expected["observed"].mean())
    assert M.chance_share(allowed, codes) == pytest.approx(expected["expected"].mean())


# ---------------------------------------------------------------- seeds
def test_procrustes_disparity_ignores_rotation_reflection_and_scale():
    a = np.random.default_rng(4).normal(size=(40, 2))
    turned = a @ np.array([[0.0, 1.0], [1.0, 0.0]]) * 2.5 - 4.0
    assert M.procrustes_disparity(a, turned) == pytest.approx(0.0, abs=1e-12)
    assert M.procrustes_disparity(a, np.random.default_rng(5).normal(size=(40, 2))) > 0.5


# ---------------------------------------------------------------- projections
def test_config_lists_the_grid_with_both_versions_settings():
    cfg = P.settings()
    specs = P.specs(cfg)
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids)) == 23 and sum(s["family"] == "umap" for s in specs) == 20
    assert P.version_ids(cfg) == {"v1": "umap_n15_d0.1", "v2": "umap_n10_d0"}      # v2.3; 30 / 0.01 until v2.2
    assert cfg["version_settings"]["v2"]["n_epochs"] == 2000
    assert set(P.version_ids(cfg).values()) <= set(ids)
    assert [s["id"] for s in specs if not s["seeded"]] == ["pca"]
    seeds = load_config("seeds")
    assert len(set(seeds["a2_projections"])) == 5 and seeds["a2_projections"][0] == seeds["umap_v2"]
    assert {s["method"] for s in cfg["sets"].values()} == {"M3", "M4"}
    comparison = np.array([True, False, True])
    assert P.set_rows(cfg, "v1_comparison", comparison).tolist() == [0, 2]
    assert P.set_rows(cfg, "v2_full", comparison).tolist() == [0, 1, 2]


def test_pca_needs_no_seed_and_a_seeded_umap_fit_repeats():
    vectors = np.random.default_rng(6).normal(size=(120, 16)).astype(np.float32)
    pca = {"id": "pca", "family": "pca", "params": {}, "seeded": False}
    assert np.array_equal(P.fit(pca, vectors, None), P.fit(pca, vectors, None))
    pytest.importorskip("umap")
    umap = {"id": "umap_n15_d0.1", "family": "umap", "seeded": True, "params": {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"}}
    first = P.fit(umap, vectors, 42)
    assert first.shape == (120, 2) and first.dtype == np.float32
    assert np.array_equal(first, P.fit(umap, vectors, 42))
    assert not np.array_equal(first, P.fit(umap, vectors, 43))


# ---------------------------------------------------------------- script 51
def test_summary_over_seeds_and_standing_in_the_grid():
    script = _load("a2_map_metrics", "scripts/51_a2_map_metrics.py")
    cfg = P.settings()
    specs, versions = P.specs(cfg), P.version_ids(cfg)
    rows = []
    for position, spec in enumerate(s for s in specs if s["family"] == "umap"):
        base = 0.9 - 0.01 * position                                           # the first setting of the grid is the best
        rows += [{"set": "v2_full", "projection": spec["id"], "seed": seed, "metric": "trustworthiness@10", "value": base + 0.001 * seed}
                 for seed in (0, 1, 2)]
    table = script.summarise(pd.DataFrame(rows))
    assert len(table) == 20 and table["fits"].eq(3).all()
    first = table.iloc[0]
    assert first["mean"] == pytest.approx(0.901) and first["sd"] == pytest.approx(0.001) and first["max"] == pytest.approx(0.902)
    standing = script.grid_standing(table, specs, versions, ["trustworthiness@10"])["v2_full"]["trustworthiness@10"]
    assert standing["best"]["projection"] == "umap_n5_d0" and standing["worst"]["projection"] == "umap_n50_d0.5"
    assert standing["v2"]["rank"] == 5 and standing["v1"]["rank"] == 11 and standing["v2"]["settings"] == 20
    assert standing["v2"]["below_best"] == pytest.approx(0.04)


# ---------------------------------------------------------------- Phase 5.1: candidates
def test_holdout_metrics_on_placed_points():
    rng = np.random.default_rng(8)
    fitted, placed = rng.normal(size=(70, 2)), rng.normal(size=(15, 2))
    from scipy.spatial.distance import cdist
    d = cdist(placed, fitted)
    same = M.holdout_metrics(d, d * 4.0, 10)                                   # a map that keeps every distance order
    assert all(v == pytest.approx(1.0) for v in same.values())
    assert set(same) == {"holdout_trustworthiness@10", "holdout_continuity@10", "holdout_knn_jaccard@10"}
    other = M.holdout_metrics(d, cdist(rng.normal(size=(15, 2)), fitted), 10)  # placed somewhere unrelated
    assert other["holdout_knn_jaccard@10"] < 0.3 and other["holdout_trustworthiness@10"] < 0.8
    # one placed point, three fitted ones: the map swaps its nearest and its farthest neighbour
    swapped = M.holdout_metrics(np.array([[1.0, 2.0, 3.0]]), np.array([[3.0, 2.0, 1.0]]), 1)
    assert swapped["holdout_knn_jaccard@1"] == 0.0
    assert swapped["holdout_trustworthiness@1"] == pytest.approx(1 - 2 * 2 / (1 * 1 * (2 * 4 - 3 - 1)))   # rank 3, penalty 2
    with pytest.raises(ValueError):
        M.holdout_metrics(d, d, 40)


def test_nearest_point_gap_and_folds():
    line = M.Space(M.euclidean_distances(np.array([[0.0, 0], [1, 0], [2, 0], [10, 0]])))
    assert M.nearest_point_gap(line) == pytest.approx(1 / 10)                  # median of 1, 1, 1, 8 over the largest distance
    parts = P.folds(23, 5, seed=1)
    assert sorted(np.concatenate(parts).tolist()) == list(range(23)) and [len(p) for p in parts] == [5, 5, 5, 4, 4]
    assert all(np.array_equal(a, b) for a, b in zip(parts, P.folds(23, 5, seed=1)))
    assert all(np.all(np.diff(p) > 0) for p in parts)


def test_candidates_config_and_refined_fits():
    cfg = load_config("a2_candidates")
    specs = P.candidate_specs(cfg)
    grid = [s for s in specs if s["candidate"]]
    assert len(grid) == 30 and all(s["family"] == "umap" for s in grid)
    assert P.umap_id(**cfg["current"]) in {s["id"] for s in grid}
    assert [s["id"] for s in specs if not s["candidate"]] == ["tsne_p30", "pacmap"]
    refined = P.refined_specs(cfg, grid[0])
    assert [s["id"] for s in refined] == ["umap_n5_d0_e1000", "umap_n5_d0_e2000"] and refined[1]["params"]["n_epochs"] == 2000
    assert "n_epochs" not in grid[0]["params"]                                  # the base setting is not changed
    assert cfg["selection"]["metrics"][:4] == P.settings()["metrics"]["headline"]
    assert not any("label" in m or "silhouette" in m for m in cfg["selection"]["metrics"])


def test_a_projected_point_lands_among_its_neighbours():
    pytest.importorskip("umap")
    rng = np.random.default_rng(9)
    centres = rng.normal(size=(3, 16)) * 6
    vectors = np.vstack([c + rng.normal(size=(60, 16)) for c in centres]).astype(np.float32)
    new = np.arange(0, 180, 9)                                                  # 20 points across the three groups
    kept = np.setdiff1d(np.arange(180), new)
    spec = {"id": "umap_n15_d0.1", "family": "umap", "seeded": True, "params": {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}}
    fitted, placed = P.fit_and_project(spec, vectors[kept], vectors[new], 42)
    assert fitted.shape == (160, 2) and placed.shape == (20, 2)
    from scipy.spatial.distance import cdist
    nearest_fitted = kept[cdist(placed, fitted).argmin(axis=1)]
    assert np.array_equal(nearest_fitted // 60, new // 60)                      # each lands in its own group
    with pytest.raises(ValueError):
        P.fit_and_project({"family": "pca", "params": {}}, vectors[kept], vectors[new], None)


def test_selection_score_and_rule():
    script = _load("a2_candidate_selection", "scripts/54_a2_candidate_selection.py")
    table = pd.DataFrame({"projection": ["a", "b", "c", "ref"], "candidate": [True, True, True, False],
                          "n_neighbors": [30, 15, 5, np.nan], "min_dist": [0.01, 0.01, 0.0, np.nan],
                          "m1": [0.90, 0.92, 0.94, 0.99], "m2": [0.50, 0.60, 0.40, 0.10]})
    scored, scale = script.score_table(table, ["m1", "m2"])
    assert scale.loc["m1", "mean"] == pytest.approx(0.92)                       # the reference takes no part in the scale
    by = scored.set_index("projection")
    assert by.loc["b", "score"] == pytest.approx((0.0 + 0.1 / np.std([0.5, 0.6, 0.4])) / 2)
    assert by["rank"].loc[["a", "b", "c"]].tolist() == [3, 1, 2] and np.isnan(by.loc["ref", "rank"])
    assert script.select(scored, {"n_neighbors": 30, "min_dist": 0.01})["projection"] == "b"
    tie = table.assign(m1=[0.9, 0.9, 0.8, 0.9], m2=[0.5, 0.5, 0.4, 0.5])
    assert script.select(script.score_table(tie, ["m1", "m2"])[0], {"n_neighbors": 30, "min_dist": 0.01})["projection"] == "a"
    rescored, _ = script.score_table(table, ["m1", "m2"], scale * 1.0)          # a given scale is used as it is
    assert rescored["score"].tolist() == pytest.approx(scored["score"].tolist())
