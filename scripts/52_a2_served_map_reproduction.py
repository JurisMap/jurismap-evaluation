"""Phase 5, A2: what it takes to reproduce version 2's served map.

Version 2 fits its map with a fixed seed (42). This script refits it from the vectors of the
Phase 1 rebuild with version 2's setting and seed, and compares each refit with the served
coordinates. It varies the two things a seed does not fix:

  libraries   production's (the interpreters named in config/a2_map.yaml) and this repo's newer ones
  row order   the order the served map was fitted in (acordao_hash order since v2.3), and the
              reverse of it

Each refit runs in its own process and only calls umap-learn; no code of another repo is imported.

Output (never overwritten):
  results/a2_served_map_reproduction.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from jm_eval.io import REPO_ROOT, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.mapping import metrics as M
from jm_eval.mapping import projections as P

SCRIPT = "52_a2_served_map_reproduction"
WORKER = """
import json, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
import numba, sklearn, umap
vectors, order, out, params = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
np.save(out, umap.UMAP(**params).fit_transform(vectors[order]))
print(json.dumps({"python": sys.version.split()[0], "umap-learn": umap.__version__, "numba": numba.__version__,
                  "numpy": np.__version__, "scikit-learn": sklearn.__version__}))
"""


def refit(python: Path, vectors: Path, order: np.ndarray, params: dict, tmp: Path) -> tuple[np.ndarray, dict]:
    order_path, out = tmp / "order.npy", tmp / "coords.npy"
    np.save(order_path, order)
    done = subprocess.run([str(python), "-c", WORKER, str(vectors), str(order_path), str(out), json.dumps(params)],
                          capture_output=True, text=True, check=True)
    return np.load(out), json.loads(done.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    cfg, seed = P.settings(), load_config("seeds")["umap_v2"]
    results, rebuild = resolve(load_config("paths")["results"]), resolve(cfg["frozen"]["rebuild"])
    k = cfg["metrics"]["jaccard_k"]
    params = {"n_components": 2, "metric": cfg["projections"]["umap"]["metric"], "random_state": seed, **cfg["version_settings"]["v2"]}
    # not `resolve`: following a virtualenv's link to its base interpreter would lose its packages
    environments = {name: Path(os.path.abspath(REPO_ROOT / python)) for name, python in cfg["frozen"]["environments"].items()}
    environments["this repo's libraries"] = Path(sys.executable)
    with RunManifest(SCRIPT, results, params={"umap": params, "environments": {e: str(p) for e, p in environments.items()}},
                     seeds={"umap_v2": seed}) as run:
        vectors_path, served = rebuild / "embeddings_v2.npy", np.load(rebuild / "coords_v2.npy")
        hashes = (rebuild / "hashes_v2.txt").read_text().split()
        for name in ("embeddings_v2.npy", "coords_v2.npy", "hashes_v2.txt"):
            run.add_input(rebuild / name)
        orders = {"as served": np.arange(len(hashes)), "reversed": np.arange(len(hashes))[::-1].copy()}
        checks = []
        with tempfile.TemporaryDirectory() as tmp:
            for environment, python in environments.items():
                if not python.exists():
                    checks.append({"libraries": environment, "skipped": f"{python} not found"})
                    continue
                for order_name, order in orders.items():
                    coords, versions = refit(python, vectors_path, order, params, Path(tmp))
                    target = served[order]
                    overlap = M.knn_jaccard(M.Space(M.euclidean_distances(coords)), M.Space(M.euclidean_distances(target)), k)
                    checks.append({"libraries": environment, "versions": versions, "row_order": order_name,
                                   "identical": bool(np.array_equal(coords, target)),
                                   "max_abs_difference": float(np.abs(coords - target).max()),
                                   "procrustes": M.procrustes_disparity(target, coords), f"map_knn_jaccard@{k}": overlap})
                    print(json.dumps(checks[-1]), flush=True)
        path = new_artefact_path(results / "a2_served_map_reproduction.json")
        path.write_text(json.dumps({"decisions": len(hashes), "umap": params, "refits_against_the_served_map": checks}, indent=2),
                        encoding="utf-8")
        run.add_output(path, "A2: refits of version 2's served map by library versions and row order")
        run.data["summary"] = {"checks": checks}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
