"""Phase 5.1, A2: fit the UMAP candidates for version 2's map (config/a2_candidates.yaml).

On all of version 2's decisions, in `acordao_hash` order, for every candidate and reference:
  - one fit per seed of config/seeds.yaml (`a2_projections`), and a second fit with production's
    seed to check that the fit repeats;
  - per held-out fold: a fit on the other folds with production's seed, and the held-out decisions
    placed on it without refitting, as an uploaded case is.

With --refine ID, the same for that setting fitted for each number of epochs of `refine_n_epochs`.

Fits are kept in data/cache/a2_projections/ (shared with scripts/50), so a stopped run resumes.

Outputs (never overwritten):
  data/v2/a2_candidates/a2_candidates.npz           `hashes`, `fold_<f>` (held-out rows) and, per fit,
                                                    "<id>__<seed>", "<id>__<seed>_repeat",
                                                    "<id>__fold<f>_fitted", "<id>__fold<f>_placed"
  data/v2/a2_candidates/a2_candidates_refine.npz    the same for --refine
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from jm_eval.io import load_config, new_artefact_path, resolve, sha256_bytes
from jm_eval.manifest import RunManifest
from jm_eval.mapping import projections as P
from jm_eval.retrieval import corpus as C

SCRIPT = "53_a2_candidate_fits"
CONFIG = "a2_candidates"


def _save(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def fit_seed(spec: dict, vectors: np.ndarray, seed: int, path: Path) -> None:
    coords = P.fit(spec, vectors, seed)
    assert np.isfinite(coords).all(), f"{path.name}: bad coordinates"
    _save(path, coords=coords)


def fit_fold(spec: dict, vectors: np.ndarray, held_out: np.ndarray, seed: int, path: Path) -> None:
    kept = np.setdiff1d(np.arange(len(vectors)), held_out)
    fitted, placed = P.fit_and_project(spec, vectors[kept], vectors[held_out], seed)
    assert np.isfinite(fitted).all() and np.isfinite(placed).all(), f"{path.name}: bad coordinates"
    _save(path, fitted=fitted, placed=placed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refine", metavar="ID", help="fit this candidate for longer instead of the grid")
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()

    cfg, map_cfg, paths, seeds = load_config(CONFIG), P.settings(), load_config("paths"), load_config("seeds")
    specs = P.candidate_specs(cfg)
    if args.refine:
        specs = P.refined_specs(cfg, next(s for s in specs if s["id"] == args.refine))
    fit_seeds, production_seed = seeds["a2_projections"], seeds["umap_v2"]
    out_dir = resolve(paths["data"]["v2"]) / "a2_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    method = cfg.get("method") or map_cfg["sets"][cfg["set"]]["method"]
    params = {"set": cfg["set"], "projections": {s["id"]: s["params"] for s in specs}, "holdout": cfg["holdout"], "refine": args.refine}
    with RunManifest(SCRIPT, out_dir, models={method: C.methods_config()["methods"][method]}, params=params,
                     seeds={"a2_projections": fit_seeds, "umap_v2": production_seed, "a2_holdout_folds": seeds["a2_holdout_folds"]}) as run:
        labels = C.load_labels()
        run.add_input(resolve(load_config("a1_labels")["corpus"]))
        rows = P.set_rows(map_cfg, cfg["set"], labels["in_comparison_set"].to_numpy())
        vectors = np.ascontiguousarray(C.method_vectors(method)[rows])
        held_out = P.folds(len(vectors), cfg["holdout"]["folds"], seeds["a2_holdout_folds"])
        cache = resolve(paths["data"]["cache"]) / "a2_projections" / f"{cfg['set']}-{sha256_bytes(vectors.tobytes())[:12]}"
        cache.mkdir(parents=True, exist_ok=True)

        seed_runs = {f"{s['id']}__{seed}": (s, seed) for s in specs for seed in fit_seeds}
        seed_runs.update({f"{s['id']}__{production_seed}_repeat": (s, production_seed) for s in specs})
        fold_runs = {f"{s['id']}__fold{f}": (s, f) for s in specs for f in range(len(held_out))}
        tasks = [delayed(fit_seed)(s, vectors, seed, cache / f"{key}.npz") for key, (s, seed) in seed_runs.items()
                 if not (cache / f"{key}.npz").exists() and not (cache / f"{key}.npy").exists()]
        tasks += [delayed(fit_fold)(s, vectors, held_out[f], production_seed, cache / f"{key}.npz") for key, (s, f) in fold_runs.items()
                  if not (cache / f"{key}.npz").exists()]
        print(f"{cfg['set']}: {len(vectors):,} decisions, {len(specs)} projections, {len(seed_runs) + len(fold_runs)} fits, {len(tasks)} to do", flush=True)
        Parallel(n_jobs=args.jobs, verbose=5)(tasks)

        arrays = {"hashes": labels["acordao_hash"].to_numpy()[rows].astype(str)}
        arrays.update({f"fold_{f}": rows_out for f, rows_out in enumerate(held_out)})
        for key in seed_runs:                                    # scripts/50 kept its fits as .npy
            arrays[key] = np.load(cache / f"{key}.npy") if (cache / f"{key}.npy").exists() else np.load(cache / f"{key}.npz")["coords"]
        for key in fold_runs:
            with np.load(cache / f"{key}.npz") as z:
                arrays[f"{key}_fitted"], arrays[f"{key}_placed"] = z["fitted"], z["placed"]
        path = new_artefact_path(out_dir / ("a2_candidates_refine.npz" if args.refine else "a2_candidates.npz"))
        np.savez_compressed(path, **arrays)
        run.add_output(path, f"A2 (5.1): fits of {len(specs)} projections on {cfg['set']}, with held-out folds")
        run.data["summary"] = {"decisions": int(len(vectors)), "projections": len(specs), "fits": len(seed_runs) + len(fold_runs),
                               "fitted_in_this_run": len(tasks)}
        print(run.data["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
