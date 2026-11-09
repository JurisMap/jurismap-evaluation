"""Checks that the reference pipeline reproduces version 2's stored vectors, retrieval and map.

Offline checks (no network):
  1. fusion      the stored retrieval vectors equal the fusion of the stored full and compressed embeddings;
  2. corpus      the vectors and coordinates in the corpus file equal the .npy files (row by acordao_hash);
  3. retrieval   the top 10 of every decision is the same from the corpus file's float32 vectors as from
                 the float64 .npy vectors (the precision of the released vectors does not change a ranking);
  4. map         a UMAP refit with the frozen parameters, rows in acordao_hash order, against the served
                 coordinates: identical with the library versions of requirements.txt; otherwise the
                 Procrustes disparity and the overlap of the 10 nearest neighbours on the two maps.

Online check (`--online N`, calls the embedding model; see jurismap_reference.py for credentials):
  5. embedding   N decisions' full and compressed reports embedded again: cosine to the stored
                 embeddings, and overlap of the top 10 retrieved with the new vector and with the stored one.

Usage:
    python reference/check_reference.py [--skip-map] [--online N] [--credentials key.json] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jurismap_reference as R  # noqa: E402

MAP_LIBRARIES = ("umap-learn", "numba", "llvmlite", "pynndescent", "numpy", "scikit-learn", "scipy")


def pinned_versions() -> dict[str, str]:
    out = {}
    for line in (Path(__file__).resolve().parent / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==")
            out[name.strip()] = ver.strip()
    return out


def installed_versions() -> dict[str, str | None]:
    out = {}
    for name in MAP_LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def knn_sets(x: np.ndarray, k: int) -> list[set[int]]:
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)
    return [set(row[1:]) for row in idx]


def jaccard(a: list[set[int]], b: list[set[int]]) -> float:
    return float(np.mean([len(x & y) / len(x | y) for x, y in zip(a, b)]))


def top10_all(vectors: np.ndarray) -> np.ndarray:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    return np.argsort(-sims, axis=1, kind="stable")[:, :10]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-map", action="store_true", help="skip the UMAP refit (a few minutes)")
    ap.add_argument("--online", type=int, default=0, help="re-embed this many decisions (calls the model)")
    ap.add_argument("--credentials", help="service-account key file for --online (Vertex AI)")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--out", help="also write the result to this JSON file")
    args = ap.parse_args()

    import pandas as pd

    result: dict = {}
    hashes, fused = R.load_vectors("fused")
    _, full = R.load_vectors("full")
    _, compressed = R.load_vectors("compressed")

    # 1. fusion
    diff = float(np.abs(R.fuse_rows(full, compressed) - fused).max())
    result["fusion"] = {"decisions": len(hashes), "max_abs_difference": diff, "passed": diff < 1e-6}

    # 2. corpus file against the .npy files
    cols = ["acordao_hash", "embedding_acordao_relatorio_compressed_v2", "embedding_acordao_relatorio_full_v2",
            "embedding_acordao_relatorio_compressed_only_v2", "embedding_acordao_relatorio_compressed_2d_v2"]
    df = pd.read_parquet(R.CORPUS, columns=cols).set_index("acordao_hash").loc[hashes]
    as_array = {c: np.vstack(df[c].to_numpy()).astype(np.float64) for c in cols[1:]}
    served = np.load(R.REBUILD / "coords_v2.npy")
    corpus_check = {
        "stored_vector": float(np.abs(as_array[cols[1]] - fused).max()),
        "full_embedding": float(np.abs(as_array[cols[2]] - full).max()),
        "compressed_embedding": float(np.abs(as_array[cols[3]] - compressed).max()),
        "coordinates": float(np.abs(as_array[cols[4]] - served).max()),
    }
    result["corpus"] = {"max_abs_difference": corpus_check,
                        "passed": all(v < 1e-6 for v in corpus_check.values())}

    # 3. retrieval: float32 corpus vectors against float64 .npy vectors
    a, b = top10_all(fused), top10_all(as_array[cols[1]])
    same_list = float(np.mean([(x == y).all() for x, y in zip(a, b)]))
    same_set = float(np.mean([set(x) == set(y) for x, y in zip(a, b)]))
    result["retrieval"] = {"share_same_top10_order": same_list, "share_same_top10_set": same_set,
                           "passed": same_set == 1.0}

    # 4. map refit
    if not args.skip_map:
        pinned, installed = pinned_versions(), installed_versions()
        matches = all(installed.get(n) == pinned.get(n) for n in pinned if n in MAP_LIBRARIES)
        _, refit = R.fit_map(fused, hashes)
        from scipy.spatial import procrustes

        _, _, disparity = procrustes(served, refit)
        identical = bool(np.array_equal(refit.astype(served.dtype), served))
        result["map"] = {
            "umap": R.UMAP_PARAMS, "installed": installed, "pinned": pinned, "versions_match_pins": matches,
            "identical": identical, "max_abs_difference": float(np.abs(refit - served).max()),
            "procrustes_disparity": float(disparity),
            "map_knn_jaccard@10": jaccard(knn_sets(served, 10), knn_sets(refit, 10)),
            "passed": identical if matches else None,
        }

    # 5. online re-embedding
    if args.online:
        rng = np.random.default_rng(args.seed)
        rows = sorted(rng.choice(len(hashes), size=args.online, replace=False).tolist())
        texts = pd.read_parquet(R.CORPUS, columns=["acordao_hash", "acordao_relatorio", "acordao_relatorio_compressed_v2"]
                                ).set_index("acordao_hash").loc[[hashes[i] for i in rows]]
        client = R.embedding_client(args.credentials)
        per = []
        for i, (h, row) in zip(rows, texts.iterrows()):
            f = R.embed(client, row["acordao_relatorio"])
            c = R.embed(client, row["acordao_relatorio_compressed_v2"])
            v = R.fuse(f, c)
            cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))  # noqa: E731
            new_top = {j for j, _ in R.top_k(fused, v, 10, exclude=i)}
            old_top = {j for j, _ in R.top_k(fused, fused[i], 10, exclude=i)}
            per.append({"acordao_hash": h, "cosine_full": cos(f, full[i]), "cosine_compressed": cos(c, compressed[i]),
                        "cosine_fused": cos(v, fused[i]), "top10_overlap": len(new_top & old_top) / 10})
        result["embedding"] = {
            "decisions": per, "model": R.EMBEDDING,
            "min_cosine_fused": min(p["cosine_fused"] for p in per),
            "mean_top10_overlap": float(np.mean([p["top10_overlap"] for p in per])),
        }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    failed = [k for k, v in result.items() if isinstance(v, dict) and v.get("passed") is False]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
