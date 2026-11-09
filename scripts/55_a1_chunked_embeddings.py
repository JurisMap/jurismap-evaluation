"""Phase 4.1, A1: the full report embedded in chunks (M5c). Public CRSNSP text only.

The embedding model reads at most 2,048 tokens, so one call per report drops the rest of 38% of
version 2's reports (Phase 4). Here every report is cut into chunks under that limit
(config/a1_candidates.yaml, `chunking`), each chunk is embedded with version 2's model and settings,
and a decision's vector is the mean of its chunk vectors weighted by chunk length. A chunk the model
still truncates (dense text) is halved until every piece is read whole.

Outputs (never overwritten):
  data/v2/a1_embeddings/M5c.npy                 one row per decision in acordao_hash order
  data/v2/a1_embeddings/a1_chunk_inputs.parquet  characters, tokens and truncation of every chunk
  results/a1_chunking.json                      chunks per decision, tokens, truncation
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import candidates as K
from jm_eval.retrieval import corpus as C
from jm_eval.retrieval.vertex_embed import VertexEmbedder

SCRIPT = "55_a1_chunked_embeddings"
METHOD = "M5c"


def embed_whole(embedder: VertexEmbedder, chunk: str) -> list[tuple[str, object]]:
    """Embed a chunk; if the model still truncated it (dense text), halve it until every piece is read whole."""
    embedded = embedder.embed(chunk)
    if not embedded.truncated or len(chunk) < 2:
        return [(chunk, embedded)]
    return [pair for half in K.chunk_text(chunk, len(chunk) // 2 + 1) for pair in embed_whole(embedder, half)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="pilot: only the first N decisions (nothing is recorded)")
    args = ap.parse_args()

    cfg, methods, paths = K.settings(), C.methods_config(), load_config("paths")
    spec, max_chars = methods["methods"][METHOD], cfg["chunking"]["max_chars"]
    results, out_dir = resolve(paths["results"]), C.embeddings_dir()
    with RunManifest(SCRIPT, results, models={METHOD: spec}, params={"chunking": cfg["chunking"], "workers": args.workers, "limit": args.limit},
                     record_outputs=not args.limit) as run:
        run.add_input(resolve(load_config("a1_labels")["corpus"]))
        hashes, texts = C.load_texts(spec["text"])
        if args.limit:
            hashes, texts = hashes[:args.limit], texts[:args.limit]
        first_cut = [K.chunk_text(t, max_chars) for t in texts]
        embedder = VertexEmbedder(spec["model"], spec["dimensions"], spec["task_type"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            done = list(pool.map(lambda c: embed_whole(embedder, c), [c for cs in first_cut for c in cs]))
        position = iter(done)
        pairs = [[pair for _ in cs for pair in next(position)] for cs in first_cut]      # per decision: (chunk, embedding)
        chunks = [[c for c, _ in ps] for ps in pairs]
        owner = [i for i, cs in enumerate(chunks) for _ in cs]
        flat, embedded = [c for cs in chunks for c in cs], [e for ps in pairs for _, e in ps]
        inputs = pd.DataFrame({"acordao_hash": [hashes[i] for i in owner], "chunk": [j for cs in chunks for j in range(len(cs))],
                               "chars": [len(c) for c in flat], "token_count": [e.token_count for e in embedded],
                               "truncated": [e.truncated for e in embedded]})
        stacked, owner = np.vstack([e.vector for e in embedded]), np.asarray(owner)
        vectors = np.vstack([K.pool(stacked[owner == i], inputs["chars"].to_numpy()[owner == i]) for i in range(len(texts))]).astype(np.float32)
        per_decision = inputs.groupby("acordao_hash", sort=False).size()
        summary = {"decisions": len(texts), "max_chars": max_chars, "chunks": len(flat),
                   "decisions_with_one_chunk": int((per_decision == 1).sum()),
                   "chunks_per_decision": {"mean": round(float(per_decision.mean()), 2), "median": float(per_decision.median()),
                                           "p90": float(per_decision.quantile(0.9)), "max": int(per_decision.max())},
                   "chunks_halved_because_truncated": len(flat) - sum(len(cs) for cs in first_cut),
                   "chunks_truncated": int(inputs["truncated"].sum()), "tokens_max": int(inputs["token_count"].max()),
                   "input_tokens": int(inputs["token_count"].sum()), "input_chars": int(inputs["chars"].sum()),
                   "calls_not_cached": sum(not e.cached for e in embedded),
                   "tokens_not_cached": int(sum(e.token_count or 0 for e in embedded if not e.cached)),
                   "credentials": dict(Counter(e.credentials for e in embedded))}
        print(json.dumps(summary, indent=1), flush=True)
        if not args.limit:
            path = new_artefact_path(out_dir / f"{METHOD}.npy")
            np.save(path, vectors)
            run.add_output(path, f"A1 {METHOD}: {spec['model']} on the full report in chunks of at most {max_chars} characters")
            inputs_path = new_artefact_path(out_dir / "a1_chunk_inputs.parquet")
            inputs.to_parquet(inputs_path, index=False)
            run.add_output(inputs_path, "A1: characters, tokens and truncation of every chunk")
            summary_path = new_artefact_path(results / "a1_chunking.json")
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            run.add_output(summary_path, "A1: chunks per decision, tokens and truncation")
        run.data["summary"] = summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
