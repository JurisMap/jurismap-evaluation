"""Phase 1 step 7: write config/v2_freeze.json, the definition of JurisMap v2 as evaluated.

Records model IDs and settings, sha256 of every prompt file, sha256 of the v2 data files, the git
SHAs of the repos (which must be clean) and the tag name to create, the BigQuery table, the UMAP
reproducibility check and the smoke test run. Refuses to write if a repo has uncommitted changes
to tracked files or if the freeze file already exists (a new freeze gets a new tag and file).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from dotenv import dotenv_values

from jm_eval.io import CONFIG_DIR, load_config, resolve, sha256_file
from jm_eval.manifest import git_state



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild-dir", required=True, help="data/v2/rebuild_<tag>")
    ap.add_argument("--frozen", required=True, help="data/v2/corpus_v2_frozen_<date>.parquet")
    ap.add_argument("--rebuilt", required=True, help="data/v2/corpus_v2_rebuilt_<tag>.parquet")
    ap.add_argument("--snapshot", required=True, help="data/v2/corpus_snapshot_<date>.parquet")
    ap.add_argument("--smoke", required=True, help="results/v2_smoke_test*.json")
    ap.add_argument("--bq-table", default="your-gcp-project.crsnsp_eval.scraper_output_v2")
    ap.add_argument("--version", default="v2", help="v2, v2.1, ...: tag <version>-eval-frozen, file <version>_freeze.json")
    ap.add_argument("--reextracted", help="data/v2/corpus_v2_1_frozen_<tag>.parquet (v2.1 and later)")
    ap.add_argument("--tag-frontend", action="store_true", help="the freeze changes the React frontend too (v2.4)")
    args = ap.parse_args()

    tag = f"{args.version}-eval-frozen"
    out_path = CONFIG_DIR / f"{args.version}_freeze.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: a new freeze needs a new version, tag and file")

    paths = load_config("paths")
    backend = resolve(paths["repos"]["backend"])
    scrapers = resolve(paths["repos"]["scrapers"])
    repos = {name: git_state(resolve(p)) for name, p in paths["repos"].items()}
    tagged = ["backend", "scrapers"] + (["frontend_v2"] if args.tag_frontend else [])
    for name in tagged:
        if repos[name]["dirty"]:
            raise SystemExit(f"{name} has uncommitted changes to tracked files")

    rebuild_dir = resolve(args.rebuild_dir)
    rebuild = json.loads((rebuild_dir / "rebuild_summary_v2.json").read_text(encoding="utf-8"))
    runtime_env = {k: v for k, v in dotenv_values(backend / "config/runtime.env").items() if v is not None}
    server_env = {k: v for k, v in dotenv_values(CONFIG_DIR / "v2_server.env").items() if v is not None}
    smoke = json.loads(resolve(args.smoke).read_text(encoding="utf-8"))

    def hashes(directory, pattern):
        return {p.name: sha256_file(p) for p in sorted(directory.glob(pattern))}

    freeze = {
        "version": args.version,
        "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {name: {k: repos[name].get(k) for k in ("sha", "branch", "dirty")} for name in repos},
        "tagged_repos": tagged,
        "untagged_repos_note": ("frontend_v2 is tagged too: it shows the cases the backend pre-selects; " if args.tag_frontend
                                else "frontend_v2 unchanged (v2 is evaluated through the backend HTTP API); ")
                               + "eval is the harness; frontend_v1 is version 1.",
        "models": {
            "backend_runtime_env": runtime_env,
            "corpus_compression": rebuild["compression"],
            "corpus_embedding": {k: rebuild["embedding"][k] for k in ("model", "vector", "shape") if k in rebuild["embedding"]},
            "embedding_task_type_corpus": "SEMANTIC_SIMILARITY",
            "embedding_task_type_chat_query": "SEMANTIC_SIMILARITY",
            "legal_search_tool": "Google Search grounding (built-in tool); the Tavily tool is not attached",
            "adk": "google-adk 1.3.0 (pinned) + GoogleSearchGroundingTool shim for Gemini 3",
        },
        "prompts_sha256": {
            "backend": hashes(backend / "prompts", "*.yaml"),
            "scrapers": hashes(scrapers / "crsnsp_scraper/prompts", "*.yaml"),
        },
        "data": {
            "snapshot": {"path": args.snapshot, "sha256": sha256_file(resolve(args.snapshot))},
            "frozen_corpus": {"path": args.frozen, "sha256": sha256_file(resolve(args.frozen))},
            **({"reextracted_corpus": {"path": args.reextracted, "sha256": sha256_file(resolve(args.reextracted))}}
               if args.reextracted else {}),
            "rebuilt_corpus": {"path": args.rebuilt, "sha256": sha256_file(resolve(args.rebuilt))},
            "shared_ids": {"path": "data/v2/shared_ids.txt",
                           "sha256": sha256_file(resolve(paths["data"]["v2"]) / "shared_ids.txt")},
            "backend_files": {name: sha256_file(rebuild_dir / name)
                              for name in ("data_cleaned_v2.joblib", "umap_mapper_v2.joblib")},
            "decisions": rebuild["decisions"],
            "bigquery_table": args.bq_table,
        },
        "umap": rebuild["umap"],
        "server_env": server_env,
        "smoke_test": {"file": args.smoke, "run_dir": smoke["run_dir"], "failures": smoke["failures"]},
    }
    out_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
