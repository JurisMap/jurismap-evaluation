"""Phase 1 step 1a: freeze the v2 corpus from the production BigQuery table (read-only).

Inputs (read-only):
  - your-gcp-project.crsnsp.scraper_output   production table (SELECT only)
  - data/v2/corpus.parquet                          Phase 0 export (the 1,523 decisions shared with v1)

Outputs (never overwritten; sha256 in data/MANIFEST.csv):
  - data/v2/corpus_snapshot_<date>.parquet   every row of the table, as read
  - data/v2/corpus_v2_frozen_<date>.parquet  one row per decision: rows without acordao_hash
        (session entries with no decision) dropped, deduplicated by acordao_hash, empty
        rapporteur reports dropped, sessions dated after the snapshot date dropped
  (the v1 ∩ v2 comparison set is built by 11_match_v1_v2.py: acordao_hash is a content hash that
   changed when production was re-scraped, so the hash join below finds no shared decisions)
  - results/v2_corpus_freeze_summary.json    aggregate counts only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timezone

import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "10_freeze_v2_corpus"


def is_blank(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or not str(x).strip()


def coverage(df: pd.DataFrame, field: str) -> dict:
    filled = ~df[field].map(lambda v: is_blank(v) or str(v).strip() == "Desconhecido")
    by_year = df.assign(_f=filled).groupby("ano_sessoes")["_f"].agg(["sum", "count"])
    return {
        "filled": int(filled.sum()),
        "total": len(df),
        "by_year": {str(int(y)): {"filled": int(r["sum"]), "total": int(r["count"])}
                    for y, r in by_year.iterrows() if pd.notna(y)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-date", default=date.today().isoformat(),
                    help="sessions dated after this day are excluded (default: today)")
    args = ap.parse_args()

    from google.cloud import bigquery  # optional dependency: `uv sync --extra gcp`

    paths = load_config("paths")
    table = paths["corpus"]["bigquery_table"]
    out_dir = resolve(paths["data"]["v2"])
    results_dir = resolve(paths["results"])
    tag = args.snapshot_date.replace("-", "")
    cutoff = pd.Timestamp(args.snapshot_date, tz="UTC") + pd.Timedelta(days=1)

    with RunManifest(SCRIPT, out_dir, params={"table": table, "snapshot_date": args.snapshot_date}) as run:
        client = bigquery.Client(project=table.split(".")[0])
        meta = client.get_table(table)
        raw = client.query(f"SELECT * FROM `{table}`").to_dataframe()
        run.data["summary"]["table_last_modified_utc"] = meta.modified.isoformat()
        run.data["summary"]["read_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        snap_path = new_artefact_path(out_dir / f"corpus_snapshot_{tag}.parquet")
        raw.to_parquet(snap_path, index=False)
        run.add_output(snap_path, f"raw read-only snapshot of {table}")

        steps = {"rows": len(raw)}
        df = raw[raw["acordao_hash"].notna()]
        steps["rows_with_acordao_hash"] = len(df)
        steps["rows_without_acordao_hash"] = len(raw) - len(df)
        df = df.sort_values("export_timestamp", ascending=False).drop_duplicates("acordao_hash")
        steps["unique_acordao_hash"] = len(df)
        empty = df["acordao_relatorio"].map(is_blank)
        steps["dropped_empty_relatorio"] = int(empty.sum())
        df = df[~empty]
        future = df["sessao_data_fim"] >= cutoff
        steps["dropped_future_sessions"] = int(future.sum())
        df = df[~future].sort_values(["sessao_data_fim", "acordao_hash"]).reset_index(drop=True)
        steps["frozen_decisions"] = len(df)

        frozen_path = new_artefact_path(out_dir / f"corpus_v2_frozen_{tag}.parquet")
        df.to_parquet(frozen_path, index=False)
        run.add_output(frozen_path, "v2 frozen corpus: one row per decision")

        p0_path = out_dir / "corpus.parquet"
        run.add_input(p0_path)
        p0 = set(pd.read_parquet(p0_path, columns=["acordao_hash"])["acordao_hash"])
        frozen_hashes = set(df["acordao_hash"])
        shared = sorted(frozen_hashes & p0)

        # Same decision published under two hashes? (same title and same report text)
        dup_content = int(df.duplicated(["acordao_titulo", "acordao_relatorio"]).sum())
        with_session = raw[raw["sessao_data_fim"].notna()]
        summary = {
            "table": table,
            "table_last_modified_utc": meta.modified.isoformat(),
            "snapshot_date": args.snapshot_date,
            "steps": steps,
            "same_title_and_report_under_other_hash": dup_content,
            "frozen": {
                "decisions": len(df),
                "sessions": int(df["sessao_nr_int"].nunique()),
                "sessions_with_null_number": int(df["sessao_nr_int"].isna().sum()),
                "date_min": str(df["sessao_data_fim"].min().date()),
                "date_max": str(df["sessao_data_fim"].max().date()),
                "decisions_by_year": {str(int(k)): int(v) for k, v in
                                      sorted(Counter(df["ano_sessoes"].dropna()).items())},
                "outcome": dict(Counter(df["acordao_resultado"].fillna("<null>")).most_common()),
                "with_production_embedding": int(df["embedding_acordao_relatorio_compressed"]
                                                 .map(lambda v: v is not None and len(v) > 0).sum()),
                "export_timestamp_max": str(df["export_timestamp"].max()),
            },
            "hash_join_with_phase0_corpus": {
                "phase0_decisions": len(p0),
                "shared": len(shared),
                "phase0_not_in_frozen": len(p0 - frozen_hashes),
                "frozen_not_in_phase0": len(frozen_hashes - p0),
            },
            "field_coverage": {f: coverage(df, f) for f in ("acordao_base", "acordao_penalidade")},
            "raw_session_dates_max": str(with_session["sessao_data_fim"].max()),
        }
        run.data["summary"] = summary
        summary_path = new_artefact_path(results_dir / "v2_corpus_freeze_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(summary_path, "aggregate counts of the v2 corpus freeze")

    print(json.dumps({k: summary[k] for k in ("steps", "frozen", "hash_join_with_phase0_corpus")}, indent=2))
    for f, c in summary["field_coverage"].items():
        print(f, c["filled"], "/", c["total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
