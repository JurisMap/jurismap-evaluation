"""Reference pipeline for JurisMap version 2's retrieval and map, written with public libraries only.

It re-implements what the system does with a decision (and with an uploaded case), as the paper
describes it, so that the benchmarked configuration can be reproduced without the system's code:

1. Embed. `gemini-embedding-001`, 3,072 dimensions, task type SEMANTIC_SIMILARITY, the model's
   default truncation, one text per request. Two texts per decision are embedded separately: the
   full rapporteur report (`acordao_relatorio`) and its extractive compression by Gemini 3.7 Flash
   (`acordao_relatorio_compressed_v2`). The compression is not re-run here (its prompt belongs to
   the system); the corpus holds its output for every decision.
2. Fuse. The retrieval vector is the unit-length sum of the two unit-length vectors.
3. Retrieve. The top k decisions by cosine similarity to a query vector.
4. Map. UMAP with the cosine metric, n_neighbors 10, min_dist 0, 2 components, 2,000 epochs and
   random_state 42, fitted on all 2,390 decisions in `acordao_hash` order. A new case is placed on
   the fitted map with `transform`. The served coordinates are reproduced exactly only with the
   library versions in requirements.txt and the rows in that order (check_reference.py).

Data (from the data archive, extracted at the repository root): data/v2/rebuild_20261009/
`embeddings_v2.npy` (stored retrieval vectors), `embeddings_full_v2.npy`, `embeddings_compressed_v2.npy`,
`hashes_v2.txt` (their row order), `coords_v2.npy` (the served map); and the corpus
data/v2/corpus_v2_rebuilt_20261009.parquet.

Usage:
    python reference/jurismap_reference.py neighbours --hash <acordao_hash> [--k 10]
    python reference/jurismap_reference.py query --report report.txt --compressed compressed.txt [--k 10]
    python reference/jurismap_reference.py fit-map --out coords.npy

`query` calls the embedding model (Vertex AI: set GOOGLE_CLOUD_PROJECT and use application-default
credentials, or pass --credentials with a service-account key; or the Gemini API with GOOGLE_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REBUILD = ROOT / "data" / "v2" / "rebuild_20261009"
CORPUS = ROOT / "data" / "v2" / "corpus_v2_rebuilt_20261009.parquet"

EMBEDDING = {"model": "gemini-embedding-001", "dimensions": 3072, "task_type": "SEMANTIC_SIMILARITY"}
UMAP_PARAMS = {"n_neighbors": 10, "min_dist": 0.0, "n_components": 2, "metric": "cosine",
               "n_epochs": 2000, "random_state": 42}


# ------------------------------------------------------------------------------------------------ data
def load_vectors(kind: str = "fused", rebuild: Path = REBUILD) -> tuple[list[str], np.ndarray]:
    """(acordao_hash per row, vectors) of the frozen corpus. kind: fused, full or compressed."""
    name = {"fused": "embeddings_v2.npy", "full": "embeddings_full_v2.npy",
            "compressed": "embeddings_compressed_v2.npy"}[kind]
    hashes = (rebuild / "hashes_v2.txt").read_text(encoding="utf-8").split()
    vectors = np.load(rebuild / name)
    if len(hashes) != vectors.shape[0]:
        raise ValueError(f"{name}: {vectors.shape[0]} rows for {len(hashes)} hashes")
    return hashes, vectors


# ------------------------------------------------------------------------------------------- embedding
def embedding_client(credentials: str | None = None, location: str = "global"):
    """A google-genai client: Vertex AI (default) or the Gemini API when GOOGLE_API_KEY is set."""
    from google import genai

    if os.environ.get("GOOGLE_API_KEY") and not credentials:
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    if credentials:
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            credentials, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return genai.Client(vertexai=True, project=creds.project_id, location=location, credentials=creds)
    return genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=location)


def embed(client, text: str) -> np.ndarray:
    """One text, one request, the model's default truncation."""
    from google.genai import types

    config = types.EmbedContentConfig(task_type=EMBEDDING["task_type"],
                                      output_dimensionality=EMBEDDING["dimensions"])
    response = client.models.embed_content(model=EMBEDDING["model"], contents=[text], config=config)
    vector = np.asarray(response.embeddings[0].values, dtype=np.float64)
    if vector.shape != (EMBEDDING["dimensions"],):
        raise RuntimeError(f"expected {EMBEDDING['dimensions']} dimensions, got {vector.shape}")
    return vector


def fuse(full: np.ndarray, compressed: np.ndarray) -> np.ndarray:
    """Unit-length sum of the unit-length embeddings of the full and the compressed report."""
    a, b = np.asarray(full, dtype=np.float64), np.asarray(compressed, dtype=np.float64)
    total = a / np.linalg.norm(a) + b / np.linalg.norm(b)
    return total / np.linalg.norm(total)


def fuse_rows(full: np.ndarray, compressed: np.ndarray) -> np.ndarray:
    a = full / np.linalg.norm(full, axis=1, keepdims=True)
    b = compressed / np.linalg.norm(compressed, axis=1, keepdims=True)
    total = a + b
    return total / np.linalg.norm(total, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------- retrieval
def top_k(vectors: np.ndarray, query: np.ndarray, k: int = 10, exclude: int | None = None) -> list[tuple[int, float]]:
    """(row, cosine similarity) of the k rows most similar to the query, most similar first.

    Ties keep row order (stable sort). `exclude` drops one row (a decision used as its own query).
    """
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    q = np.asarray(query, dtype=np.float64)
    sims = unit @ (q / np.linalg.norm(q))
    if exclude is not None:
        sims[exclude] = -np.inf
    order = np.argsort(-sims, kind="stable")[:k]
    return [(int(i), float(sims[i])) for i in order]


# ------------------------------------------------------------------------------------------------- map
def canonical_order(hashes: list[str]) -> list[int]:
    """Row numbers that put the decisions in acordao_hash order, the order of every map fit."""
    return sorted(range(len(hashes)), key=lambda i: hashes[i])


def fit_map(vectors: np.ndarray, hashes: list[str], params: dict | None = None):
    """Fit the global UMAP on the rows in acordao_hash order. Returns (mapper, coordinates in input row order)."""
    from umap import UMAP

    order = canonical_order(hashes)
    mapper = UMAP(**(params or UMAP_PARAMS))
    coords_sorted = mapper.fit_transform(vectors[order])
    coords = np.empty_like(coords_sorted)
    coords[order] = coords_sorted
    return mapper, coords


def place(mapper, vector: np.ndarray) -> np.ndarray:
    """2D position of a new case on a fitted map."""
    return mapper.transform(np.asarray(vector, dtype=np.float64).reshape(1, -1))[0]


# ------------------------------------------------------------------------------------------------- CLI
def _titles(hashes: list[str]) -> dict[str, str]:
    import pandas as pd

    if not CORPUS.exists():
        return {}
    df = pd.read_parquet(CORPUS, columns=["acordao_hash", "acordao_titulo"])
    return dict(zip(df.acordao_hash, df.acordao_titulo))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("neighbours", help="the k decisions most similar to a decision of the corpus")
    n.add_argument("--hash", required=True)
    n.add_argument("--k", type=int, default=10)
    q = sub.add_parser("query", help="embed a new case (full text and compressed text) and retrieve")
    q.add_argument("--report", required=True, help="file with the full text")
    q.add_argument("--compressed", required=True, help="file with the compressed text")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--credentials", help="service-account key file (Vertex AI)")
    f = sub.add_parser("fit-map", help="refit the map and save the coordinates (input row order)")
    f.add_argument("--out", required=True)
    args = ap.parse_args()

    hashes, vectors = load_vectors("fused")
    if args.cmd == "neighbours":
        row = hashes.index(args.hash)
        titles = _titles(hashes)
        hits = top_k(vectors, vectors[row], args.k, exclude=row)
        print(json.dumps([{"acordao_hash": hashes[i], "similarity": round(s, 6),
                           "title": titles.get(hashes[i])} for i, s in hits], ensure_ascii=False, indent=1))
    elif args.cmd == "query":
        client = embedding_client(args.credentials)
        full = embed(client, Path(args.report).read_text(encoding="utf-8").strip())
        compressed = embed(client, Path(args.compressed).read_text(encoding="utf-8").strip())
        hits = top_k(vectors, fuse(full, compressed), args.k)
        print(json.dumps([{"acordao_hash": hashes[i], "similarity": round(s, 6)} for i, s in hits], indent=1))
    elif args.cmd == "fit-map":
        _, coords = fit_map(vectors, hashes)
        np.save(args.out, coords)
        print(f"saved {coords.shape} to {args.out}")


if __name__ == "__main__":
    main()
