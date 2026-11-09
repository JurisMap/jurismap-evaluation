"""Phase 4b, A1: Vertex AI embeddings and exact input lengths. Public CRSNSP text only.

- M3 (text-multilingual-embedding-002, full report) and M5 (gemini-embedding-001, full report) are
  embedded here: data/v2/a1_embeddings/M3.npy and M5.npy, one row per decision in acordao_hash order.
- M4 and M4b use the vectors stored in the frozen corpus. Their texts are embedded again only to
  read the input length and to check that the call reproduces the stored vector.
- Input length: every embedding response states the token count of the input and whether the
  model truncated it (limit 2,048 tokens). The count_tokens endpoint does not serve embedding
  models, so these figures replace it; they are exact.

Outputs (never overwritten):
  data/v2/a1_embeddings/M3.npy, M5.npy, a1_embedding_inputs.parquet (tokens and truncation per text)
  results/a1_input_length.csv      share of inputs truncated, per method and scope
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C
from jm_eval.retrieval.vertex_embed import VertexEmbedder

SCRIPT = "43_a1_cloud_embeddings"
VERTEX_METHODS = ["M3", "M4", "M4b", "M5"]


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def input_length_table(inputs: pd.DataFrame, shared: set[str], limit: int) -> pd.DataFrame:
    rows = []
    for scope, mask in {"comparison_set": inputs["acordao_hash"].isin(shared), "v2": inputs["acordao_hash"].notna()}.items():
        for method, sub in inputs[mask].groupby("method", sort=True):
            tokens = sub["token_count"]
            rows.append({
                "scope": scope, "method": method, "model": sub["model"].iloc[0], "text": sub["text"].iloc[0],
                "inputs": len(sub), "truncated": int(sub["truncated"].sum()),
                "share_truncated": round(float(sub["truncated"].mean()), 4),
                "tokens_median": float(tokens.median()), "tokens_mean": round(float(tokens.mean()), 1),
                "tokens_p90": float(tokens.quantile(0.9)), "tokens_max": int(tokens.max()),
                "share_of_tokens_read": round(float(tokens.clip(upper=limit).sum() / tokens.sum()), 4),
                "chars_per_token": round(float(sub["chars"].sum() / tokens.sum()), 3),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="pilot: only the first N decisions (nothing is recorded)")
    args = ap.parse_args()

    cfg, paths = C.methods_config(), load_config("paths")
    results = resolve(paths["results"])
    out_dir = C.embeddings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {m: cfg["methods"][m] for m in VERTEX_METHODS}
    models = {m: {k: s[k] for k in ("model", "dimensions", "task_type", "text")} for m, s in specs.items()}
    with RunManifest(SCRIPT, results, models=models, params={"workers": args.workers, "limit": args.limit},
                     record_outputs=not args.limit) as run:
        run.add_input(resolve(load_config("a1_labels")["corpus"]))
        shared = set(resolve(load_config("a1_labels")["comparison_set"]).read_text().split())
        frames, summary = [], {}
        for method, spec in specs.items():
            hashes, texts = C.load_texts(spec["text"])
            if args.limit:
                hashes, texts = hashes[:args.limit], texts[:args.limit]
            embedder = VertexEmbedder(spec["model"], spec["dimensions"], spec["task_type"])
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                embedded = list(pool.map(embedder.embed, texts))
            vectors = np.vstack([e.vector for e in embedded])
            frames.append(pd.DataFrame({
                "acordao_hash": hashes, "method": method, "model": spec["model"], "text": spec["text"],
                "chars": [len(t) for t in texts], "token_count": [e.token_count for e in embedded],
                "truncated": [e.truncated for e in embedded]}))
            info = {"decisions": len(hashes), "calls_not_cached": sum(not e.cached for e in embedded),
                    "credentials": dict(Counter(e.credentials for e in embedded)),
                    "input_tokens": int(sum(e.token_count or 0 for e in embedded)),
                    "input_chars": int(sum(len(t) for t in texts))}
            if spec["kind"] == "stored":
                stored = C.stored_vectors(spec["column"])[:len(hashes)]
                cos = cosine_rows(vectors, stored)
                info["cosine_with_stored_vectors"] = {"min": round(float(cos.min()), 6), "median": round(float(np.median(cos)), 6),
                                                      "below_0.999": int((cos < 0.999).sum())}
            elif not args.limit:
                path = new_artefact_path(out_dir / f"{method}.npy")
                np.save(path, vectors)
                run.add_output(path, f"A1 {method}: {spec['model']} on {spec['text']} (acordao_hash order)")
            summary[method] = info
            print(method, info, flush=True)

        inputs = pd.concat(frames, ignore_index=True)
        table = input_length_table(inputs, shared, cfg["embedding_input_limit_tokens"])
        print(table.to_string(index=False))
        if not args.limit:
            inputs_path = new_artefact_path(out_dir / "a1_embedding_inputs.parquet")
            inputs.to_parquet(inputs_path, index=False)
            run.add_output(inputs_path, "A1: token count and truncation of every embedding input")
            table_path = new_artefact_path(results / "a1_input_length.csv")
            table.to_csv(table_path, index=False)
            run.add_output(table_path, "A1: share of embedding inputs truncated, per method and scope")
        run.data["summary"] = summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
