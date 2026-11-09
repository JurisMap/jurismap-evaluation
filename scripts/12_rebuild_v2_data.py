"""Phase 1 step 3: rebuild the v2 data (compression, embeddings, global UMAP) on the frozen corpus.

From v2.3 (Phase 5.1b): the stored vector is the fusion of the full-report and compressed-report
embeddings, the rows are in acordao_hash order, and the map parameters are given on the command
line. --compression-cache starts the rebuild from a frozen compression cache, so that no text is
compressed again and the compressed texts stay those of the earlier freeze.

The rebuild itself runs in the scrapers repo (crsnsp_scraper/rebuild_v2_local.py, branch
release/v2-iswa) with the scrapers' own environment, so this repo imports no proprietary code and
holds no system prompt. This script prepares its input, runs it, checks the outputs and records
everything (sha256 in data/MANIFEST.csv, run manifest).

Inputs:  data/v2/corpus_v2_frozen_<date>.parquet, data/v2/shared_ids.txt
Outputs: data/v2/rebuild_<tag>/            input records + the rebuild's outputs
             data_cleaned_v2.joblib, umap_mapper_v2.joblib   (backend: DATA_VERSION=v2, DATA_DIR=<this dir>)
         data/v2/corpus_v2_rebuilt_<tag>.parquet
             frozen corpus + acordao_relatorio_compressed_v2, embedding_acordao_relatorio_compressed_v2
             (the stored vector), embedding_acordao_relatorio_compressed_2d_v2, compression ratios
             (production and v2); from v2.3 also the two parts of the stored vector,
             embedding_acordao_relatorio_full_v2 and embedding_acordao_relatorio_compressed_only_v2
         results/v2_rebuild_summary.json    aggregate statistics only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import date

import joblib
import numpy as np
import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "12_rebuild_v2_data"
MODELS = {
    "compression": "gemini-3.7-flash",
    "compression_thinking_level": "low",
    "compression_max_output_tokens": 8192,
    "embedding": "gemini-embedding-001",
    "genai_location": "global",
}
REPEATED = ["acordao_relator", "acordao_recorrente", "acordao_recorrido", "acordao_advogado"]


def plain(value):
    """Convert pandas/numpy values to the Python types production's data_cleaned.joblib uses."""
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def to_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict("records"):
        rec = {k: plain(v) for k, v in row.items() if not k.startswith("embedding_")}
        for field in REPEATED:
            rec[field] = list(rec.get(field) or [])
        records.append(rec)
    return records


def describe(x: pd.Series) -> dict:
    q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(x.count()), "mean": round(float(x.mean()), 4), "sd": round(float(x.std()), 4),
            "p05": round(float(q[0.05]), 4), "p25": round(float(q[0.25]), 4), "median": round(float(q[0.5]), 4),
            "p75": round(float(q[0.75]), 4), "p95": round(float(q[0.95]), 4),
            "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", required=True, help="data/v2/corpus_v2_frozen_<date>.parquet")
    ap.add_argument("--tag", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("--limit", type=int, help="pilot: only the first N decisions (not a paper artefact)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.01)
    ap.add_argument("--umap-n-epochs", type=int, help="from v2.3; omitted: the rebuild script's default")
    ap.add_argument("--compression-cache", help="compression_cache.jsonl of an earlier rebuild to start from")
    args = ap.parse_args()

    paths = load_config("paths")
    seeds = load_config("seeds")
    data_v2 = resolve(paths["data"]["v2"])
    results_dir = resolve(paths["results"])
    scrapers = resolve(paths["repos"]["scrapers"])
    tag = args.tag + (f"_pilot{args.limit}" if args.limit else "")
    work_dir = data_v2 / f"rebuild_{tag}"
    umap_params = {"n_neighbors": args.umap_n_neighbors, "min_dist": args.umap_min_dist, "n_components": 2,
                   "metric": "cosine", "random_state": seeds["umap_v2"]}
    if args.umap_n_epochs:
        umap_params["n_epochs"] = args.umap_n_epochs

    with RunManifest(SCRIPT, work_dir, models=MODELS,
                     params={"frozen": args.frozen, "limit": args.limit, "workers": args.workers,
                             "umap": umap_params, "rebuild_script": "crsnsp_scraper/rebuild_v2_local.py"},
                     record_outputs=not args.limit) as run:
        frozen_path = resolve(args.frozen)
        run.add_input(frozen_path)
        df = pd.read_parquet(frozen_path)
        # Same rule as the corpus freeze: decisions without a rapporteur report are dropped (for v2.1:
        # a re-extraction found no report in the document)
        empty = df["acordao_relatorio"].fillna("").str.strip() == ""
        run.data["params"]["dropped_empty_relatorio"] = sorted(df.loc[empty, "acordao_hash"])
        # acordao_hash order: the order the rebuild fits the map in (from v2.3)
        df = df[~empty].sort_values("acordao_hash").reset_index(drop=True)
        if args.limit:
            df = df.head(args.limit)
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "input_records.joblib"
        if not input_path.exists():
            joblib.dump(to_records(df), input_path)
        if args.compression_cache and not (work_dir / "compression_cache.jsonl").exists():
            run.add_input(resolve(args.compression_cache))
            shutil.copyfile(resolve(args.compression_cache), work_dir / "compression_cache.jsonl")

        env = {**os.environ,
               "GOOGLE_CLOUD_PROJECT": paths["corpus"]["bigquery_table"].split(".")[0],
               "GOOGLE_GENAI_USE_VERTEXAI": "True",
               # primary Gemini credentials; errors fall back to the project's own credentials
               "GEMINI_PRIMARY_CREDENTIALS": str(resolve(paths["gemini_primary_credentials"])),
               "GEMINI_PRIMARY_RETRY_ATTEMPTS": os.environ.get("GEMINI_PRIMARY_RETRY_ATTEMPTS", "6"),
               "GEMINI_PRIMARY_COOLDOWN_S": os.environ.get("GEMINI_PRIMARY_COOLDOWN_S", "60"),
               "GENAI_LOCATION": MODELS["genai_location"],
               "GOOGLE_GENAI_MAIN_MODEL_NAME": MODELS["compression"],
               "COMPRESSION_THINKING_LEVEL": MODELS["compression_thinking_level"],
               "COMPRESSION_MAX_OUTPUT_TOKENS": str(MODELS["compression_max_output_tokens"]),
               "EMBEDDING_MODEL": MODELS["embedding"]}
        cmd = [str(scrapers / ".venv/bin/python"), "crsnsp_scraper/rebuild_v2_local.py",
               "--input", str(input_path), "--out-dir", str(work_dir), "--version", "v2",
               "--workers", str(args.workers), "--n-neighbors", str(umap_params["n_neighbors"]),
               "--min-dist", str(umap_params["min_dist"]), "--random-state", str(umap_params["random_state"])]
        if args.umap_n_epochs:
            cmd += ["--n-epochs", str(args.umap_n_epochs)]
        run.data["params"]["command"] = cmd[1:]
        subprocess.run(cmd, cwd=scrapers, env=env, check=True)

        rebuild = json.loads((work_dir / "rebuild_summary_v2.json").read_text(encoding="utf-8"))
        hashes = (work_dir / "hashes_v2.txt").read_text(encoding="utf-8").split()
        emb = np.load(work_dir / "embeddings_v2.npy")
        coords = np.load(work_dir / "coords_v2.npy")
        cache = pd.read_json(work_dir / "compression_cache.jsonl", lines=True).set_index("acordao_hash")
        assert hashes == df["acordao_hash"].tolist()

        out = df.copy()
        out["acordao_relatorio_compressed_v2"] = cache.loc[hashes, "compressed_text"].tolist()
        out["compression_finish_reason_v2"] = cache.loc[hashes, "finish_reason"].tolist()
        out["embedding_acordao_relatorio_compressed_v2"] = list(emb.astype(np.float32))
        out["embedding_acordao_relatorio_compressed_2d_v2"] = list(coords.astype(np.float32))
        parts = {"embedding_acordao_relatorio_full_v2": "embeddings_full_v2.npy",
                 "embedding_acordao_relatorio_compressed_only_v2": "embeddings_compressed_v2.npy"}
        parts = {column: work_dir / name for column, name in parts.items() if (work_dir / name).exists()}
        for column, path in parts.items():
            out[column] = list(np.load(path).astype(np.float32))
        n_orig = out["acordao_relatorio"].str.len()
        out["compression_ratio_v2"] = out["acordao_relatorio_compressed_v2"].str.len() / n_orig
        # Production compressed the report text it had (for v2.1: the pre-re-extraction text)
        n_prod = (out["acordao_relatorio_v2_0"] if "acordao_relatorio_v2_0" in out else out["acordao_relatorio"]).str.len()
        out["compression_ratio_production"] = out["acordao_relatorio_compressed"].fillna("").str.len() / n_prod

        shared = set((data_v2 / "shared_ids.txt").read_text(encoding="utf-8").split())
        is_shared = out["acordao_hash"].isin(shared)
        summary = {
            "decisions": len(out),
            "rebuild": {k: rebuild[k] for k in ("compression", "embedding", "umap")},
            "compression_ratio_chars": {
                "production_gemini-2.5-flash-lite": describe(out["compression_ratio_production"]),
                "v2_gemini-3.7-flash": describe(out["compression_ratio_v2"]),
                "shared_production": describe(out.loc[is_shared, "compression_ratio_production"]),
                "shared_v2": describe(out.loc[is_shared, "compression_ratio_v2"]),
            },
            "compressed_chars": {"production": describe(out["acordao_relatorio_compressed"].fillna("").str.len()),
                                 "v2": describe(out["acordao_relatorio_compressed_v2"].str.len())},
            "v2_ratio_above_1": int((out["compression_ratio_v2"] > 1).sum()),
            "shared_decisions": int(is_shared.sum()),
            "dropped_empty_relatorio": run.data["params"]["dropped_empty_relatorio"],
        }
        run.data["summary"] = summary

        if not args.limit:
            parquet_path = data_v2 / f"corpus_v2_rebuilt_{args.tag}.parquet"
            if parquet_path.exists():
                raise SystemExit(f"Refusing to overwrite {parquet_path}")
            out.to_parquet(parquet_path, index=False)
            run.add_output(parquet_path, "v2 frozen corpus + v2 compression, embeddings, 2D coordinates")
            for name, note in [("data_cleaned_v2.joblib", "backend DATA_VERSION=v2 data"),
                               ("umap_mapper_v2.joblib", "backend DATA_VERSION=v2 UMAP mapper"),
                               ("embeddings_v2.npy", "v2 embeddings (row order = hashes_v2.txt)"),
                               *[(path.name, "a part of the v2 embeddings (row order = hashes_v2.txt)") for path in parts.values()],
                               ("coords_v2.npy", "v2 global UMAP coordinates"),
                               ("hashes_v2.txt", "row order of the v2 arrays"),
                               ("compression_cache.jsonl", "v2 compressions with finish reasons and tokens"),
                               ("rebuild_summary_v2.json", "rebuild script summary"),
                               ("input_records.joblib", "rebuild input (frozen corpus records)")]:
                run.add_output(work_dir / name, note)
            summary_path = new_artefact_path(results_dir / "v2_rebuild_summary.json")
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            run.add_output(summary_path, "aggregate v2 rebuild statistics")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
