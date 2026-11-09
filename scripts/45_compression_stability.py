"""Phase 4b, A1: is version 2's compression stable when it is run again?

Compression (Gemini 3.7 Flash, temperature 0) is not deterministic. 50 random reports are compressed
and embedded once more with the frozen pipeline (scrapers `crsnsp_scraper/rebuild_v2_local.py` at
the frozen tag, run as a subprocess exactly as scripts/12 ran it). For each of the 50 decisions:

- cosine between the frozen vector and the new one;
- how many of its 10 nearest neighbours stay the same when the new vector replaces the frozen one
  as the query (neighbours are searched among the frozen vectors of all other decisions);
- length of the new compressed text relative to the frozen one, and whether the two are identical.

The frozen texts and vectors remain the record; nothing here replaces them.

Outputs (never overwritten):
  data/v2/compression_stability_<tag>/      the rerun's files
  results/a1_compression_stability.csv      one row per decision
  results/a1_compression_stability.json     summary
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import date

import joblib
import numpy as np
import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C

SCRIPT = "45_compression_stability"
MODELS = load_config("models")["system_under_test"]


def top_k(query: np.ndarray, vectors: np.ndarray, exclude: int, k: int) -> list[int]:
    scores = vectors @ query
    scores[exclude] = -np.inf
    return np.argsort(-scores, kind="stable")[:k].tolist()


def compare(frozen: np.ndarray, rerun: np.ndarray, rows: list[int], k: int) -> pd.DataFrame:
    """Cosine and neighbour overlap for the decisions at `rows` of the unit-norm `frozen` matrix."""
    out = []
    for new, row in zip(rerun, rows):
        new = new / np.linalg.norm(new)
        before, after = top_k(frozen[row].copy(), frozen, row, k), top_k(new, frozen, row, k)
        out.append({"cosine": float(frozen[row] @ new), "neighbours_kept": len(set(before) & set(after)),
                    "first_neighbour_kept": before[0] == after[0]})
    return pd.DataFrame(out)


def describe(x: pd.Series) -> dict:
    return {"mean": round(float(x.mean()), 4), "median": round(float(x.median()), 4),
            "min": round(float(x.min()), 4), "p10": round(float(x.quantile(0.1)), 4), "max": round(float(x.max()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg, paths = C.methods_config(), load_config("paths")
    settings = cfg["compression_stability"]
    k = cfg["neighbours"]
    results, scrapers = resolve(paths["results"]), resolve(paths["repos"]["scrapers"])
    work_dir = resolve(paths["data"]["v2"]) / f"compression_stability_{args.tag}"
    compression = MODELS["compression"]
    with RunManifest(SCRIPT, results, models={"compression": compression, "embedding": MODELS["embedding"]},
                     params={**settings, "neighbours": k, "rebuild_script": "crsnsp_scraper/rebuild_v2_local.py"}) as run:
        columns = [cfg["texts"]["report"], cfg["texts"]["compressed_v2"], cfg["methods"]["M4"]["column"]]
        corpus = C.load_corpus(columns)
        run.add_input(resolve(load_config("a1_labels")["corpus"]))
        rows = sorted(np.random.default_rng(settings["seed"]).choice(len(corpus), settings["n"], replace=False).tolist())
        sample = corpus.iloc[rows]

        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "input_records.joblib"
        if not input_path.exists():
            joblib.dump([{"acordao_hash": h, "acordao_relatorio": t}
                         for h, t in zip(sample["acordao_hash"], sample[cfg["texts"]["report"]])], input_path)
        if not (work_dir / "embeddings_v2.npy").exists():
            env = {**os.environ,
                   "GOOGLE_CLOUD_PROJECT": paths["corpus"]["bigquery_table"].split(".")[0],
                   "GOOGLE_GENAI_USE_VERTEXAI": "True",
                   "GEMINI_PRIMARY_CREDENTIALS": str(resolve(paths["gemini_primary_credentials"])),
                   "GEMINI_PRIMARY_RETRY_ATTEMPTS": "6", "GEMINI_PRIMARY_COOLDOWN_S": "60",
                   "GENAI_LOCATION": MODELS["location"],
                   "GOOGLE_GENAI_MAIN_MODEL_NAME": compression["model"],
                   "COMPRESSION_THINKING_LEVEL": compression["thinking_level"],
                   "COMPRESSION_MAX_OUTPUT_TOKENS": str(compression["max_output_tokens"]),
                   "EMBEDDING_MODEL": MODELS["embedding"]}
            env.pop("VIRTUAL_ENV", None)
            cmd = [str(scrapers / ".venv/bin/python"), "crsnsp_scraper/rebuild_v2_local.py",
                   "--input", str(input_path), "--out-dir", str(work_dir), "--version", "v2",
                   "--workers", str(args.workers)]
            run.data["params"]["command"] = cmd[1:]
            subprocess.run(cmd, cwd=scrapers, env=env, check=True)

        hashes = (work_dir / "hashes_v2.txt").read_text(encoding="utf-8").split()
        assert hashes == sample["acordao_hash"].tolist()
        rerun = np.load(work_dir / "embeddings_v2.npy")
        cache = pd.read_json(work_dir / "compression_cache.jsonl", lines=True).set_index("acordao_hash")
        frozen = np.vstack(corpus[cfg["methods"]["M4"]["column"]].to_numpy()).astype(np.float64)
        frozen /= np.linalg.norm(frozen, axis=1, keepdims=True)

        table = compare(frozen, rerun.astype(np.float64), rows, k)
        old_text, new_text = sample[cfg["texts"]["compressed_v2"]].tolist(), cache.loc[hashes, "compressed_text"].tolist()
        table.insert(0, "acordao_hash", hashes)
        table["identical_text"] = [a == b for a, b in zip(old_text, new_text)]
        table["length_ratio"] = [len(b) / len(a) for a, b in zip(old_text, new_text)]
        table["finish_reason"] = cache.loc[hashes, "finish_reason"].tolist()

        table_path = new_artefact_path(results / "a1_compression_stability.csv")
        table.to_csv(table_path, index=False)
        run.add_output(table_path, "A1: compression rerun, one row per decision")
        usage = cache.loc[hashes]
        summary = {
            "decisions": len(table), "neighbours": k,
            "cosine_frozen_vs_rerun": describe(table["cosine"]),
            "neighbours_kept_of_10": describe(table["neighbours_kept"]),
            "share_all_neighbours_kept": round(float((table["neighbours_kept"] == k).mean()), 4),
            "share_first_neighbour_kept": round(float(table["first_neighbour_kept"].mean()), 4),
            "share_identical_text": round(float(table["identical_text"].mean()), 4),
            "length_ratio_rerun_to_frozen": describe(table["length_ratio"]),
            "finish_reasons": table["finish_reason"].value_counts().to_dict(),
            "tokens": {c: int(usage[c].sum()) for c in ("prompt_tokens", "output_tokens", "thoughts_tokens") if c in usage},
        }
        summary_path = new_artefact_path(results / "a1_compression_stability.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(summary_path, "A1: compression rerun, summary")
        run.data["summary"] = summary
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
