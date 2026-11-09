"""Phase 4b, A1: local dense retrievers on the CRSNSP corpus (M6 BGE-M3, M7 multilingual-E5-large).

--setup-only downloads what the local methods need and stops: the two models (data/cache/hf) and
the NLTK stop words and RSLP rules (data/cache/nltk_data). After it, the lexical baselines and the
SUSEP script run without network access.

Without the option, the full report of every decision is embedded on this machine:
  data/v2/a1_embeddings/M6.npy, M7.npy            one row per decision in acordao_hash order
  data/v2/a1_embeddings/a1_local_inputs.parquet   token count and truncation of every input
  results/a1_input_length_local.csv               share of inputs truncated, per method and scope
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C
from jm_eval.retrieval import dense_local as D
from jm_eval.retrieval.lexical import nltk_data_dir

SCRIPT = "44_a1_local_embeddings"
LOCAL_METHODS = ["M6", "M7"]


def setup(specs: dict) -> None:
    import nltk
    from huggingface_hub import snapshot_download

    target = nltk_data_dir()
    target.mkdir(parents=True, exist_ok=True)
    for package in ("stopwords", "rslp"):
        if not nltk.download(package, download_dir=str(target), quiet=True):
            raise SystemExit(f"NLTK package {package} could not be downloaded")
    for spec in specs.values():
        path = snapshot_download(spec["model"], cache_dir=str(D.models_dir()),
                                 ignore_patterns=["onnx/*", "openvino/*", "*.onnx", "*.onnx_data", "imgs/*"])
        print(spec["model"], "->", path)


def length_table(inputs: pd.DataFrame, shared: set[str]) -> pd.DataFrame:
    rows = []
    for scope, mask in {"comparison_set": inputs["acordao_hash"].isin(shared), "v2": inputs["acordao_hash"].notna()}.items():
        for method, sub in inputs[mask].groupby("method", sort=True):
            tokens, limit = sub["token_count"], int(sub["limit"].iloc[0])
            rows.append({"scope": scope, "method": method, "model": sub["model"].iloc[0], "text": sub["text"].iloc[0],
                         "limit_tokens": limit, "inputs": len(sub), "truncated": int(sub["truncated"].sum()),
                         "share_truncated": round(float(sub["truncated"].mean()), 4),
                         "tokens_median": float(tokens.median()), "tokens_mean": round(float(tokens.mean()), 1),
                         "tokens_p90": float(tokens.quantile(0.9)), "tokens_max": int(tokens.max()),
                         "share_of_tokens_read": round(float(tokens.clip(upper=limit).sum() / tokens.sum()), 4)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup-only", action="store_true", help="download the models and the NLTK data, then stop")
    ap.add_argument("--methods", nargs="+", default=LOCAL_METHODS, choices=LOCAL_METHODS)
    ap.add_argument("--limit", type=int, help="pilot: only the N longest reports (nothing is recorded)")
    args = ap.parse_args()

    cfg, paths = C.methods_config(), load_config("paths")
    specs = {m: cfg["methods"][m] for m in args.methods}
    if args.setup_only:
        setup({m: cfg["methods"][m] for m in LOCAL_METHODS})
        return 0

    results, out_dir = resolve(paths["results"]), C.embeddings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {m: {k: s.get(k) for k in ("model", "max_seq_length", "prefix", "text")} for m, s in specs.items()}
    with RunManifest(SCRIPT, results, models=models, params={"limit": args.limit, "precision": "float16 on GPU"},
                     record_outputs=not args.limit) as run:
        run.add_input(resolve(load_config("a1_labels")["corpus"]))
        shared = set(resolve(load_config("a1_labels")["comparison_set"]).read_text().split())
        frames, summary = [], {}
        for method, spec in specs.items():
            hashes, texts = C.load_texts(spec["text"])
            if args.limit:
                longest = np.argsort([-len(t) for t in texts])[:args.limit]
                hashes, texts = [hashes[i] for i in longest], [texts[i] for i in longest]
            model = D.load_model(spec["model"], spec["max_seq_length"], offline=True)
            started = time.perf_counter()
            vectors, counts = D.encode(model, texts, spec.get("prefix", ""), progress=True)
            seconds = round(time.perf_counter() - started, 1)
            if not np.isfinite(vectors).all():
                raise SystemExit(f"{method}: non-finite values in the vectors")
            frames.append(pd.DataFrame({"acordao_hash": hashes, "method": method, "model": spec["model"],
                                        "text": spec["text"], "limit": spec["max_seq_length"], "token_count": counts,
                                        "truncated": counts > spec["max_seq_length"]}))
            summary[method] = {"decisions": len(hashes), "seconds": seconds, "device": str(model.device),
                               "dimensions": int(vectors.shape[1])}
            print(method, summary[method])
            if not args.limit:
                path = new_artefact_path(out_dir / f"{method}.npy")
                np.save(path, vectors)
                run.add_output(path, f"A1 {method}: {spec['model']} on {spec['text']} (acordao_hash order)")
            del model

        inputs = pd.concat(frames, ignore_index=True)
        table = length_table(inputs, shared)
        print(table.to_string(index=False))
        if not args.limit:
            inputs_path = new_artefact_path(out_dir / "a1_local_inputs.parquet")
            inputs.to_parquet(inputs_path, index=False)
            run.add_output(inputs_path, "A1: token count and truncation of every input of the local models")
            table_path = new_artefact_path(results / "a1_input_length_local.csv")
            table.to_csv(table_path, index=False)
            run.add_output(table_path, "A1: share of inputs truncated by the local models")
        run.data["summary"] = summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
