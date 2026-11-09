"""Phase 1 (v2.1): re-extract the rapporteur reports and votes of the frozen v2 corpus.

Production's extractor sent only the first 50,000 characters of each document to the model, which
cut long reports and dropped many votes. The re-extraction runs in the scrapers repo
(crsnsp_scraper/reextract_local.py, release/v2-iswa) with Gemini 3.7 Flash, the production prompt
and no practical input limit, on the stored source files. Only acordao_relatorio and
acordao_voto_relator are replaced; every other field keeps its frozen value.

Inputs:  data/v2/corpus_v2_frozen_<date>.parquet, data/v2/corpus.parquet (v1 texts, diagnostics),
         data/v2/v1_v2_id_map.csv
Outputs: data/v2/reextract_<tag>/            the re-extraction's cache, sources and document texts
         data/v2/corpus_v2_1_frozen_<tag>.parquet   frozen corpus with the re-extracted texts
             (previous texts kept as acordao_relatorio_v2_0 / acordao_voto_relator_v2_0)
         results/v2_1_reextraction_summary.json      aggregate diagnostics only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from datetime import date

import joblib
import numpy as np
import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "16_reextract_v2_reports"
MODELS = {"extraction": "gemini-3.7-flash", "thinking_level": "low", "max_input_chars": 1_000_000,
          "genai_location": "global"}
FIELDS = ("acordao_relatorio", "acordao_voto_relator")
PUNCT = tuple('.;:)"”»')


def words(text) -> list[str]:
    return re.findall(r"\w+", str(text or "").lower())


def ngram_support(text, source_ngrams: set, n: int = 8) -> float:
    """Share of the text's word n-grams that occur in the source document (1.0 = verbatim)."""
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return float(np.mean([g in source_ngrams for g in grams])) if grams else float("nan")


def describe(x) -> dict:
    x = pd.Series(x, dtype=float).dropna()
    if x.empty:
        return {"n": 0}
    q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(x.count()), "mean": round(float(x.mean()), 4), "p05": round(float(q[0.05]), 4),
            "p25": round(float(q[0.25]), 4), "median": round(float(q[0.5]), 4),
            "p75": round(float(q[0.75]), 4), "p95": round(float(q[0.95]), 4)}


def text_stats(series: pd.Series) -> dict:
    s = series.fillna("").astype(str).str.strip()
    return {"empty": int((s == "").sum()), "chars": describe(s.str.len()),
            "ends_with_punctuation": round(float(s[s != ""].str.endswith(PUNCT).mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--tag", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("--hashes-file", help="pilot: only these acordao_hash values (not a paper artefact)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retry-failed", action="store_true", help="re-process decisions that failed in an earlier run")
    args = ap.parse_args()

    paths = load_config("paths")
    data_v2 = resolve(paths["data"]["v2"])
    scrapers = resolve(paths["repos"]["scrapers"])
    pilot = bool(args.hashes_file)
    work_dir = data_v2 / (f"reextract_{args.tag}" + ("_pilot" if pilot else ""))
    work_dir.mkdir(parents=True, exist_ok=True)

    with RunManifest(SCRIPT, work_dir, models=MODELS, record_outputs=not pilot,
                     params={"frozen": args.frozen, "workers": args.workers, "pilot": pilot,
                             "replaced_fields": list(FIELDS)}) as run:
        frozen_path = resolve(args.frozen)
        run.add_input(frozen_path)
        df = pd.read_parquet(frozen_path)
        if pilot:
            wanted = set(resolve(args.hashes_file).read_text().split())
            df = df[df["acordao_hash"].isin(wanted)].reset_index(drop=True)
        input_path = work_dir / "input_records.joblib"
        if not input_path.exists():
            joblib.dump(df[["acordao_hash", "gcs_file_path"]].to_dict("records"), input_path)

        env = {**os.environ,
               "GOOGLE_CLOUD_PROJECT": paths["corpus"]["bigquery_table"].split(".")[0],
               "BIGQUERY_TABLE": paths["corpus"]["bigquery_table"],  # names list only (SELECT)
               "GOOGLE_GENAI_USE_VERTEXAI": "True",
               # primary Gemini credentials; errors fall back to the project's own credentials
               "GEMINI_PRIMARY_CREDENTIALS": str(resolve(paths["gemini_primary_credentials"])),
               "GEMINI_PRIMARY_RETRY_ATTEMPTS": os.environ.get("GEMINI_PRIMARY_RETRY_ATTEMPTS", "6"),
               "GEMINI_PRIMARY_COOLDOWN_S": os.environ.get("GEMINI_PRIMARY_COOLDOWN_S", "60"),
               "GENAI_LOCATION": MODELS["genai_location"],
               "GOOGLE_GENAI_MAIN_MODEL_NAME": MODELS["extraction"],
               "GOOGLE_GENAI_FALLBACK_MODEL_NAME": MODELS["extraction"],
               "EXTRACTION_THINKING_LEVEL": MODELS["thinking_level"],
               "EXTRACTION_MAX_INPUT_CHARS": str(MODELS["max_input_chars"])}
        cmd = [str(scrapers / ".venv/bin/python"), "crsnsp_scraper/reextract_local.py",
               "--input", str(input_path), "--out-dir", str(work_dir), "--workers", str(args.workers),
               *(["--retry-failed"] if args.retry_failed else [])]
        proc = subprocess.run(cmd, cwd=scrapers, env=env)
        run.data["params"]["reextract_exit_code"] = proc.returncode

        cache = pd.read_json(work_dir / "extraction_cache.jsonl", lines=True).drop_duplicates(
            "acordao_hash", keep="last").set_index("acordao_hash")
        missing = sorted(set(df["acordao_hash"]) - set(cache.index))
        failed = [h for h in cache.index if cache.at[h, "fields"] is None]

        out = df.copy()
        for field in FIELDS:
            out[f"{field}_v2_0"] = out[field]
            new = out["acordao_hash"].map(lambda h: (cache.at[h, "fields"] or {}).get(field)
                                          if h in cache.index else None)
            # Keep the previous text only where the re-extraction failed outright
            out[field] = new.where(out["acordao_hash"].isin(cache.index) & ~out["acordao_hash"].isin(failed),
                                   out[field])
        out["reextracted"] = out["acordao_hash"].isin(cache.index) & ~out["acordao_hash"].isin(failed)

        # Faithfulness: share of word 8-grams of the new texts that occur in the document text
        support = {f: [] for f in FIELDS}
        for h, row in out.set_index("acordao_hash").iterrows():
            txt_path = work_dir / "sources" / f"{h}.txt"
            if not txt_path.exists():
                continue
            w = words(txt_path.read_text(encoding="utf-8"))
            grams = {tuple(w[i:i + 8]) for i in range(len(w) - 7)}
            for f in FIELDS:
                support[f].append(ngram_support(row[f], grams))

        attempts = [a for lst in cache["attempts"] for a in lst]
        idmap = pd.read_csv(data_v2 / "v1_v2_id_map.csv").dropna(subset=["v2_acordao_hash"])
        v1 = pd.read_parquet(data_v2 / "corpus.parquet", columns=["acordao_hash", *FIELDS]).set_index("acordao_hash")
        shared = idmap[idmap["v2_acordao_hash"].isin(out["acordao_hash"])]
        new_by_hash = out.set_index("acordao_hash")
        ratio = {f: (new_by_hash.loc[shared["v2_acordao_hash"], f].fillna("").str.len().to_numpy()
                     / v1.loc[shared["v1_acordao_hash"], f].fillna("").str.len().replace(0, np.nan).to_numpy())
                 for f in FIELDS}
        summary = {
            "decisions": len(out),
            "reextracted": int(out["reextracted"].sum()),
            "missing": len(missing),
            "failed": len(failed),
            "attempts": {"total": len(attempts),
                         "finish_reasons": dict(Counter(str(a.get("finish_reason")) for a in attempts)),
                         "errors": sum(bool(a.get("error")) for a in attempts),
                         "input_truncated": sum(bool(a.get("input_truncated")) for a in attempts)},
            "document_chars": describe(cache["document_chars"]),
            "documents_over_50000_chars": int((cache["document_chars"] > 50_000).sum()),
            **{f: {"v2_0": text_stats(out[f"{f}_v2_0"]), "v2_1": text_stats(out[f]),
                   "identical_to_v2_0": int((out[f].fillna("") == out[f"{f}_v2_0"].fillna("")).sum()),
                   "source_8gram_support": describe(support[f]),
                   "length_ratio_to_v1_shared": describe(ratio[f])} for f in FIELDS},
            "v1_shared_decisions": len(shared),
        }
        run.data["summary"] = summary

        if not pilot:
            parquet_path = new_artefact_path(data_v2 / f"corpus_v2_1_frozen_{args.tag}.parquet")
            out.to_parquet(parquet_path, index=False)
            run.add_output(parquet_path, "v2.1 frozen corpus: v2 frozen corpus with re-extracted reports and votes")
            for name, note in [("extraction_cache.jsonl", "re-extraction results and call metadata"),
                               ("existing_names.json", "names list used in the extraction prompt"),
                               ("input_records.joblib", "re-extraction input")]:
                run.add_output(work_dir / name, note)
            summary_path = new_artefact_path(resolve(paths["results"]) / "v2_1_reextraction_summary.json")
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            run.add_output(summary_path, "aggregate re-extraction diagnostics")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not (missing or failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
