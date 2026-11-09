"""Phase 0: export the JurisMap v2 corpus to data/v2/corpus.parquet.

Inputs (read-only):
  - app/backend/data_cleaned.joblib           current v2 corpus (list of dicts, compressed-text embeddings)
  - app/backend@v1-expert-eval:data_cleaned.joblib
        v1 snapshot (scraper export of 30 Mar 2025), read from git without a checkout.
        Used to enrich v2 with session and URL fields and to reconcile the 1,531 count.
  - optional (--bigquery): the production BigQuery table, read-only.

Outputs:
  - data/v2/corpus.parquet (+ 00_export_corpus.manifest.json), hash recorded in data/MANIFEST.csv
  - results/corpus_summary.json   aggregate counts only (public CRSNSP data)

Duplicates: records with the same acordao_hash are verified to be identical in
every field and then kept once; `n_source_records` says how many copies there were.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve, sha256_bytes, sha256_file
from jm_eval.manifest import RunManifest

SCRIPT = "00_export_corpus"
EMB = "embedding_acordao_relatorio_compressed"
UNKNOWN = "Desconhecido"

# e.g. "15414.604792/2016-77", "15414.604792 2016-77", "15414.000494/98-24", "5414.613049/2020-94"
PROC_RE = re.compile(r"(?<![\d.])(\d{4,5})\.(\d{6,7})\s*[/ ]\s*(\d{2,4})\s*-\s*(\d{2})(?!\d)")
RECURSO_RE = re.compile(r"Recurso\s*(?:n[ºo°.]*\s*)?(\d+)", re.IGNORECASE)

# Fields taken from the v1 snapshot, joined on acordao_hash.
V1_FIELDS = {
    "acordao_url": "acordao_url",
    "url": "sessao_url",
    "sessao_hash": "sessao_hash",
    "sessao_nr_int": "sessao_nr_int",
    "sessao_data_fim": "sessao_data_fim",
    "ano_sessoes": "ano_sessoes",
    "acordao_base": "acordao_base",
    "acordao_penalidade": "acordao_penalidade",
    "acordao_ementa": "acordao_ementa",
    "acordao_resultado": "acordao_resultado",
    "acordao_resultado_curated": "v1_acordao_resultado_curated",
    "acordao_titulo": "v1_acordao_titulo",
    "export_timestamp": "v1_export_timestamp",
}


def find_processes(text: str | None) -> list[dict]:
    out = []
    for m in PROC_RE.finditer(text or ""):
        a, b, c, d = m.groups()
        out.append({
            "formatted": f"{a}.{b}/{c}-{d}",
            "digits": a + b + c + d,
            "valid": len(a) == 5 and len(b) == 6 and len(c) == 4,
        })
    return out


def process_fields(rec: dict) -> dict:
    title_procs = find_processes(rec["acordao_titulo"])
    source, procs = "titulo", title_procs
    if not procs:
        for field in ("acordao_relatorio", "acordao_voto_relator"):
            procs = find_processes(rec.get(field))
            if procs:
                source = field
                break
    m = RECURSO_RE.search(rec["acordao_titulo"])
    primary = procs[0] if procs else None
    return {
        "process_number": primary["formatted"] if primary else None,
        "process_number_digits": primary["digits"] if primary else None,
        "process_number_valid_format": primary["valid"] if primary else None,
        "process_number_source": source if primary else None,
        "process_numbers_in_title": [p["digits"] for p in title_procs],
        "recurso_number": m.group(1) if m else None,
    }


def records_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        if k == EMB:
            if not np.array_equal(np.asarray(a[k]), np.asarray(b[k])):
                return False
        elif a[k] != b[k]:
            return False
    return True


def load_v1_from_git(paths: dict) -> tuple[list[dict], dict]:
    spec = paths["corpus"]["v1_joblib_git"]
    repo = resolve(paths["repos"][spec["repo"]])
    ref = f"{spec['rev']}:{spec['path']}"
    blob = subprocess.run(["git", "-C", str(repo), "show", ref], capture_output=True, check=True).stdout
    tag_sha = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{spec['rev']}^{{commit}}"],
                             capture_output=True, text=True, check=True).stdout.strip()
    info = {"path": f"{repo}@{ref}", "commit": tag_sha, "sha256": sha256_bytes(blob)}
    return joblib.load(io.BytesIO(blob)), info


def load_bigquery(table: str) -> pd.DataFrame:
    """Read-only SELECT of session/URL fields from the production table."""
    from google.cloud import bigquery  # optional dependency: `uv sync --extra gcp`

    client = bigquery.Client(project=table.split(".")[0])
    sql = (f"SELECT acordao_hash, acordao_url, sessao_hash, sessao_nr_int, sessao_data_fim, ano_sessoes "
           f"FROM `{table}`")
    return client.query(sql).to_dataframe()


def counter(values) -> dict:
    return dict(Counter(values).most_common())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bigquery", action="store_true", help="also read the production table (read-only)")
    args = ap.parse_args()

    paths = load_config("paths")
    v2_path = resolve(paths["corpus"]["v2_joblib"])
    out_dir = resolve(paths["data"]["v2"])
    results_dir = resolve(paths["results"])

    with RunManifest(SCRIPT, out_dir, params={"dedupe": "exact duplicates by acordao_hash",
                                              "bigquery": args.bigquery}) as run:
        # ---- v2 corpus
        run.add_input(v2_path)
        raw: list[dict] = joblib.load(v2_path)
        by_hash: dict[str, list[dict]] = {}
        for rec in raw:
            by_hash.setdefault(rec["acordao_hash"], []).append(rec)
        dup_groups = {h: rs for h, rs in by_hash.items() if len(rs) > 1}
        non_identical = [h for h, rs in dup_groups.items() if not all(records_equal(rs[0], r) for r in rs[1:])]
        if non_identical:
            raise SystemExit(f"Duplicate hashes with differing content: {non_identical}")
        unique = [rs[0] for rs in by_hash.values()]

        # ---- v1 snapshot (from git tag)
        v1, v1_info = load_v1_from_git(paths)
        run.add_input(v1_info["path"], sha256=v1_info["sha256"], commit=v1_info["commit"])
        v1_by_hash = {r["acordao_hash"]: r for r in v1}

        # ---- build table
        rows = []
        for rec in unique:
            row = {k: v for k, v in rec.items()}
            row["acordao_titulo"] = rec["acordao_titulo"].strip()
            row["n_source_records"] = len(by_hash[rec["acordao_hash"]])
            row.update(process_fields(rec))
            v1rec = v1_by_hash.get(rec["acordao_hash"])
            row["in_v1_snapshot"] = v1rec is not None
            for src, dst in V1_FIELDS.items():
                row[dst] = v1rec.get(src) if v1rec else None
            row["relatorio_same_as_v1"] = (v1rec.get("acordao_relatorio") == rec["acordao_relatorio"]) if v1rec else None
            rows.append(row)
        df = pd.DataFrame(rows)
        df["sessao_data_fim"] = pd.to_datetime(df["sessao_data_fim"], utc=True)
        df["v1_export_timestamp"] = pd.to_datetime(df["v1_export_timestamp"], utc=True)

        if args.bigquery:
            bq = load_bigquery(paths["corpus"]["bigquery_table"]).drop_duplicates("acordao_hash")
            df = df.merge(bq.add_prefix("bq_").rename(columns={"bq_acordao_hash": "acordao_hash"}),
                          on="acordao_hash", how="left")

        E = np.vstack(df[EMB].to_numpy()).astype(np.float64)
        df[EMB] = list(E)

        out = new_artefact_path(out_dir / "corpus.parquet")
        df.to_parquet(out, index=False)
        run.add_output(out, note="v2 corpus, exact duplicates removed, v1 session/URL fields joined")

        # ---- summary (aggregate only)
        v1_hashes = set(v1_by_hash)
        v2_hashes = set(by_hash)
        v1_only = [v1_by_hash[h] for h in sorted(v1_hashes - v2_hashes)]
        n_proc = df["process_number_digits"].nunique()
        proc_counts = df["process_number_digits"].value_counts()
        summary = {
            "v2_source": {"path": str(v2_path), "sha256": sha256_file(v2_path)},
            "v2_records_in_source": len(raw),
            "v2_unique_decisions": len(unique),
            "v2_exact_duplicate_groups": len(dup_groups),
            "v2_duplicate_titles": sorted(rs[0]["acordao_titulo"].strip() for rs in dup_groups.values()),
            "outcome_distribution_source_records": counter(r["acordao_resultado_curated"] for r in raw),
            "outcome_distribution_unique": counter(df["acordao_resultado_curated"]),
            "ementa_unknown_source_records": sum(r["acordao_ementa_curated"] == UNKNOWN for r in raw),
            "ementa_unknown_unique": int((df["acordao_ementa_curated"] == UNKNOWN).sum()),
            "embedding": {"field": EMB, "dim": int(E.shape[1]), "n": int(E.shape[0]),
                          "nan": int(np.isnan(E).sum()),
                          "norm_min": float(np.linalg.norm(E, axis=1).min()),
                          "norm_max": float(np.linalg.norm(E, axis=1).max())},
            "process_numbers": {
                "found": int(df["process_number"].notna().sum()),
                "by_source": counter(df["process_number_source"].fillna("none")),
                "invalid_format": int((df["process_number_valid_format"] == False).sum()),  # noqa: E712
                "titles_with_multiple_processes": int((df["process_numbers_in_title"].map(len) > 1).sum()),
                "distinct_primary_processes": int(n_proc),
                "processes_with_multiple_decisions": int((proc_counts > 1).sum()),
                "decisions_sharing_a_process": int(proc_counts[proc_counts > 1].sum()),
            },
            "entities_unique_excl_unknown": {
                k: len({x for xs in df[k] for x in xs} - {UNKNOWN})
                for k in ["acordao_relator_curated", "acordao_recorrente_curated",
                          "acordao_advogado_curated", "acordao_recorrido_curated"]
            },
            "sessions": {
                "distinct_sessao_hash": int(df["sessao_hash"].nunique()),
                "date_min": str(df["sessao_data_fim"].min().date()),
                "date_max": str(df["sessao_data_fim"].max().date()),
                "decisions_per_year": {int(k): int(v) for k, v in df["ano_sessoes"].value_counts().sort_index().items()},
            },
            "v1_enrichment": {
                "joined": int(df["in_v1_snapshot"].sum()),
                "relatorio_identical_to_v1": int(df["relatorio_same_as_v1"].sum()),
                "acordao_base_nonempty": int(df["acordao_base"].fillna("").str.strip().ne("").sum()),
                "acordao_penalidade_nonempty": int(df["acordao_penalidade"].fillna("").str.strip().ne("").sum()),
                "outcome_label_changed_v1_to_v2": int((df["acordao_resultado_curated"] != df["v1_acordao_resultado_curated"]).sum()),
                "outcome_crosstab_v1_to_v2": {
                    f"{a} -> {b}": int(n) for (a, b), n in
                    df.groupby(["v1_acordao_resultado_curated", "acordao_resultado_curated"]).size().items() if a != b
                },
            },
            "reconciliation_1531": {
                "v1_snapshot": {**v1_info, "records": len(v1), "unique_decisions": len(v1_hashes),
                                "export_timestamp": str(max(r["export_timestamp"] for r in v1)),
                                "sessions": len({r["sessao_hash"] for r in v1}),
                                "outcome_distribution_unique": counter(
                                    r["acordao_resultado_curated"] for r in v1_by_hash.values())},
                "in_v1_not_v2": len(v1_only),
                "in_v2_not_v1": len(v2_hashes - v1_hashes),
                "in_v1_not_v2_empty_relatorio": sum(not (r.get("acordao_relatorio") or "").strip() for r in v1_only),
                "in_v1_not_v2_titles": sorted(r["acordao_titulo"].strip() for r in v1_only),
            },
        }
        run.data["summary"] = summary
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = new_artefact_path(results_dir / "corpus_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(summary_path, note="aggregate corpus counts")

    print(json.dumps({k: summary[k] for k in ["v2_records_in_source", "v2_unique_decisions",
                                              "outcome_distribution_source_records",
                                              "ementa_unknown_source_records"]}, ensure_ascii=False, indent=1))
    print(f"wrote {out}\nmanifest {run.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
