"""Phase 5, A2: fit every 2D projection of the map validation.

For each embedding set of config/a2_map.yaml (version 1's vectors and version 2's on the comparison
set; version 2's on all its decisions): the UMAP grid, t-SNE, PaCMAP and PCA, each seeded
projection once per seed of config/seeds.yaml (`a2_projections`). Vectors are those of the A1
benchmark (M3, M4), so no model is called.

A finished fit is kept in data/cache/a2_projections/ and never fitted again for the same vectors,
so the script can be stopped and resumed.

Outputs (never overwritten):
  data/v2/a2_projections/<set>.npz    `hashes` and one (n, 2) array per fit, keyed "<projection>__<seed>"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from jm_eval.io import load_config, new_artefact_path, resolve, sha256_bytes
from jm_eval.manifest import RunManifest
from jm_eval.mapping import projections as P
from jm_eval.retrieval import corpus as C

SCRIPT = "50_a2_projections"


def run_key(spec: dict, seed: int | None) -> str:
    return f"{spec['id']}__{seed if spec['seeded'] else 'none'}"


def fit_cached(spec: dict, vectors: np.ndarray, seed: int | None, path: Path) -> tuple[str, float]:
    start = time.perf_counter()
    coords = P.fit(spec, vectors, seed)
    assert coords.shape == (len(vectors), 2) and np.isfinite(coords).all(), f"{path.name}: bad coordinates"
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, coords)
    tmp.rename(path)
    return spec["family"], time.perf_counter() - start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", help="default: every set of config/a2_map.yaml")
    ap.add_argument("--jobs", type=int, default=4, help="fits run at the same time (each seeded fit uses one thread)")
    args = ap.parse_args()

    cfg, paths, seeds = P.settings(), load_config("paths"), load_config("seeds")["a2_projections"]
    sets = args.sets or list(cfg["sets"])
    specs = P.specs(cfg)
    out_dir = resolve(paths["data"]["v2"]) / "a2_projections"
    cache_root = resolve(paths["data"]["cache"]) / "a2_projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"sets": {s: cfg["sets"][s] for s in sets}, "projections": cfg["projections"], "jobs": args.jobs}
    with RunManifest(SCRIPT, out_dir, models={m: C.methods_config()["methods"][m] for m in {cfg["sets"][s]["method"] for s in sets}},
                     params=params, seeds={"a2_projections": seeds}) as run:
        labels = C.load_labels()
        hashes = labels["acordao_hash"].to_numpy()
        run.add_input(resolve(load_config("a1_labels")["corpus"]))
        summary = {}
        for name in sets:
            rows = P.set_rows(cfg, name, labels["in_comparison_set"].to_numpy())
            vectors = np.ascontiguousarray(C.method_vectors(cfg["sets"][name]["method"])[rows])
            cache = cache_root / f"{name}-{sha256_bytes(vectors.tobytes())[:12]}"
            cache.mkdir(parents=True, exist_ok=True)
            runs = [(spec, seed) for spec in specs for seed in (seeds if spec["seeded"] else [None])]
            todo = [(spec, seed) for spec, seed in runs if not (cache / f"{run_key(spec, seed)}.npy").exists()]
            print(f"{name}: {len(vectors):,} decisions, {len(runs)} fits, {len(todo)} to do", flush=True)
            done = Parallel(n_jobs=args.jobs, verbose=5)(
                delayed(fit_cached)(spec, vectors, seed, cache / f"{run_key(spec, seed)}.npy") for spec, seed in todo)
            arrays = {run_key(spec, seed): np.load(cache / f"{run_key(spec, seed)}.npy") for spec, seed in runs}
            path = new_artefact_path(out_dir / f"{name}.npz")
            np.savez_compressed(path, hashes=hashes[rows].astype(str), **arrays)      # plain strings: read without pickle
            run.add_output(path, f"A2: 2D projections of {cfg['sets'][name]['label']} ({len(runs)} fits)")
            seconds: dict[str, list[float]] = {}
            for family, elapsed in done:
                seconds.setdefault(family, []).append(elapsed)
            summary[name] = {"decisions": int(len(vectors)), "fits": len(runs), "fitted_in_this_run": len(todo),
                             "mean_seconds_per_fit": {f: round(float(np.mean(s)), 1) for f, s in seconds.items()}}
        run.data["summary"] = summary
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
