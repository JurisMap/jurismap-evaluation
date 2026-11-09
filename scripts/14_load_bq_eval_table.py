"""Phase 1 step 1: BigQuery table for the v2 server's runtime reads.

The v2 backend still reads BigQuery at runtime (/API_get_filters always; filter_data_by_id and
filter_data_by_id_and_fields as fallbacks). So that those reads see the frozen v2 corpus, never the
production table, this loads the rebuilt v2 corpus into your-gcp-project.crsnsp_eval.scraper_output_v2
with the production table's schema: the columns acordao_relatorio_compressed,
embedding_acordao_relatorio_compressed and embedding_acordao_relatorio_compressed_2d hold the v2 values.

Never writes to the production dataset `crsnsp`; refuses to overwrite an existing table (WRITE_EMPTY).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from jm_eval.io import load_config, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "14_load_bq_eval_table"
DATASET = "crsnsp_eval"
DEFAULT_TABLE = "scraper_output_v2"
V2_COLUMNS = {
    "acordao_relatorio_compressed": "acordao_relatorio_compressed_v2",
    "embedding_acordao_relatorio_compressed": "embedding_acordao_relatorio_compressed_v2",
    "embedding_acordao_relatorio_compressed_2d": "embedding_acordao_relatorio_compressed_2d_v2",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuilt", required=True, help="data/v2/corpus_v2_rebuilt_<tag>.parquet")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="table name in crsnsp_eval (never overwritten)")
    args = ap.parse_args()

    from google.cloud import bigquery  # optional dependency: `uv sync --extra gcp`

    paths = load_config("paths")
    prod_table = paths["corpus"]["bigquery_table"]
    project, prod_dataset, _ = prod_table.split(".")
    assert DATASET != prod_dataset, "refusing to write to the production dataset"
    table_id = f"{project}.{DATASET}.{args.table}"
    out_dir = resolve(paths["data"]["v2"])

    with RunManifest(SCRIPT, out_dir, params={"table": table_id, "schema_from": prod_table}) as run:
        rebuilt_path = resolve(args.rebuilt)
        run.add_input(rebuilt_path)
        client = bigquery.Client(project=project)
        prod = client.get_table(prod_table)  # schema only (read-only)

        df = pd.read_parquet(rebuilt_path)
        for column, v2_column in V2_COLUMNS.items():
            df[column] = df[v2_column].map(lambda v: None if v is None else [float(x) for x in v]) \
                if column.startswith("embedding") else df[v2_column]
        schema_names = [f.name for f in prod.schema]
        df = df[schema_names]

        dataset = bigquery.Dataset(f"{project}.{DATASET}")
        dataset.location = prod.location
        dataset.description = "JurisMap ISWA evaluation (never production). Frozen v2 corpus."
        client.create_dataset(dataset, exists_ok=True)
        job = client.load_table_from_dataframe(
            df, table_id,
            job_config=bigquery.LoadJobConfig(schema=prod.schema, write_disposition="WRITE_EMPTY"))
        job.result()
        table = client.get_table(table_id)
        summary = {"table": table_id, "rows": table.num_rows, "location": table.location,
                   "created_utc": table.created.isoformat(), "load_job": job.job_id,
                   "v2_columns": V2_COLUMNS}
        run.data["summary"] = summary
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
