"""The 2D projections compared in the A2 map validation (Phase 5), from config/a2_map.yaml.

UMAP over a grid of n_neighbors and min_dist (it contains the settings of both versions), t-SNE
(openTSNE), PaCMAP and PCA. Each library keeps its defaults except for the settings the config
names. The libraries are imported when a projection is fitted (`uv sync --extra map`).
"""

from __future__ import annotations

import warnings

import numpy as np

from jm_eval.io import load_config

CONFIG = "a2_map"


def settings() -> dict:
    return load_config(CONFIG)


def umap_id(n_neighbors: int, min_dist: float, n_epochs: int | None = None) -> str:
    return f"umap_n{n_neighbors}_d{min_dist:g}" + (f"_e{n_epochs}" if n_epochs else "")


def specs(cfg: dict) -> list[dict]:
    """Every projection of the config: {id, family, params, seeded}, the UMAP grid first."""
    p = cfg["projections"]
    out = [{"id": umap_id(n, d), "family": "umap", "seeded": True,
            "params": {"n_neighbors": n, "min_dist": float(d), "metric": p["umap"]["metric"]}}
           for n in p["umap"]["n_neighbors"] for d in p["umap"]["min_dist"]]
    out.append({"id": f"tsne_p{p['tsne']['perplexity']}", "family": "tsne", "seeded": True, "params": dict(p["tsne"])})
    out.append({"id": "pacmap", "family": "pacmap", "seeded": True, "params": dict(p["pacmap"] or {})})
    out.append({"id": "pca", "family": "pca", "seeded": False, "params": dict(p["pca"] or {})})
    return out


def version_ids(cfg: dict) -> dict[str, str]:
    """The projection ID of each version's UMAP setting, e.g. {"v1": "umap_n15_d0.1", ...}."""
    return {v: umap_id(s["n_neighbors"], s["min_dist"]) for v, s in cfg["version_settings"].items()}   # the grid's cell


def candidate_specs(cfg: dict) -> list[dict]:
    """The UMAP grid of config/a2_candidates.yaml, then its reference projections (`candidate: False`)."""
    u = cfg["candidates"]["umap"]
    out = [{"id": umap_id(n, d), "family": "umap", "seeded": True, "candidate": True,
            "params": {"n_neighbors": n, "min_dist": float(d), "metric": u["metric"]}}
           for n in u["n_neighbors"] for d in u["min_dist"]]
    for family, params in (cfg.get("references") or {}).items():
        name = f"tsne_p{params['perplexity']}" if family == "tsne" else family
        out.append({"id": name, "family": family, "seeded": True, "candidate": False, "params": dict(params or {})})
    return out


def refined_specs(cfg: dict, base: dict) -> list[dict]:
    """A UMAP setting fitted for each number of epochs of `refine_n_epochs`."""
    p = base["params"]
    return [{**base, "id": umap_id(p["n_neighbors"], p["min_dist"], e), "params": {**p, "n_epochs": e}}
            for e in cfg["refine_n_epochs"]]


def unit(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fit(spec: dict, vectors: np.ndarray, seed: int | None) -> np.ndarray:
    """Coordinates (n, 2, float32) of one projection of `vectors` fitted with `seed`."""
    family, params = spec["family"], spec["params"]
    if family == "umap":
        from umap import UMAP
        with warnings.catch_warnings():                 # a seeded fit runs on one thread and says so
            warnings.simplefilter("ignore", UserWarning)
            coords = UMAP(n_components=2, random_state=seed, **params).fit_transform(vectors)
    elif family == "tsne":
        from openTSNE import TSNE
        coords = TSNE(n_components=2, random_state=seed, **params).fit(unit(vectors))
    elif family == "pacmap":
        from pacmap import PaCMAP
        coords = PaCMAP(n_components=2, random_state=seed, **params).fit_transform(unit(vectors))
    elif family == "pca":
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, svd_solver="full", **params).fit_transform(unit(vectors))
    else:
        raise ValueError(f"unknown projection family: {family}")
    return np.asarray(coords, dtype=np.float32)


def fit_and_project(spec: dict, fitted: np.ndarray, new: np.ndarray, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Fit a map on `fitted` and place `new` on it without refitting, as an uploaded case is placed.

    Returns the coordinates of both (float32). UMAP's projection uses its default transform seed.
    """
    family, params = spec["family"], spec["params"]
    if family == "umap":
        from umap import UMAP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mapper = UMAP(n_components=2, random_state=seed, **params).fit(fitted)
            coords, placed = mapper.embedding_, mapper.transform(new)
    elif family == "tsne":
        from openTSNE import TSNE
        coords = TSNE(n_components=2, random_state=seed, **params).fit(unit(fitted))
        placed = coords.transform(unit(new))
    elif family == "pacmap":
        from pacmap import PaCMAP
        mapper = PaCMAP(n_components=2, random_state=seed, save_tree=True, **params)
        coords = mapper.fit_transform(unit(fitted))
        placed = mapper.transform(unit(new), basis=unit(fitted))
    else:
        raise ValueError(f"{family} cannot place new points")
    return np.asarray(coords, dtype=np.float32), np.asarray(placed, dtype=np.float32)


def folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    """k disjoint held-out sets that cover range(n), from one seeded shuffle; each sorted."""
    return [np.sort(part) for part in np.array_split(np.random.default_rng(seed).permutation(n), k)]


def set_rows(cfg: dict, name: str, in_comparison_set: np.ndarray) -> np.ndarray:
    """Row numbers (corpus in `acordao_hash` order) of the decisions of one embedding set."""
    scope = cfg["sets"][name]["scope"]
    if scope == "comparison_set":
        return np.flatnonzero(in_comparison_set)
    if scope == "v2":
        return np.arange(len(in_comparison_set))
    raise ValueError(f"unknown scope: {scope}")
