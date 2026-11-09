"""Phase 2 step 1: the reproducibility table (settings of v1 and v2, side by side).

Every value in the table is tied to evidence in the tagged code or the harness's own records: v1 is
read at the `v1-expert-eval` tag (backend, scrapers, frontend_v1), v2 at the tag of the current
freeze file (config/v2.7_freeze.json; the React frontend at the SHA that file records), or the backend at
`--v2-backend-rev` (the v2.7 tag deployed on 2026-10-25, measured in Phase 9). Each
evidence item is a regular expression that must match the file at that revision; the script
records `repo@rev:path:line` for it and refuses to write the table if any item does not match, so
the table cannot drift from the code silently. Package versions are parsed from the pinned
requirement files, not typed in.

`.env` files are only read line by line for `*MODEL*`, `*REGION*` and `*LOCATION*` variables
(ground rule 3); nothing from them is printed.

Latency and cost come from the latest results/a6_latency_cost*.csv (Phase 9, A6).

Outputs: results/reproducibility_table.csv and .md (never overwritten: a rerun writes _2, ...),
         results/20_settings_table.manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from jm_eval.io import CONFIG_DIR, REPO_ROOT, latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "20_settings_table"
FREEZE_FILE = "v2.7_freeze.json"
ENV_KEEP = re.compile(r"^\s*[A-Z_]*(MODEL|REGION|LOCATION)[A-Z_]*\s*=")

# paper_scope values (ground rule 9: data corrections are not version differences)
DIFF = "version difference"
SAME = "same in both"
SHARED = "shared evaluation setting (ground rule 9)"
REPRO = "reproducibility only (not a version difference)"
V2ONLY = "v2 evaluation instrumentation"


@dataclass
class Ev:
    """Evidence: `pattern` must match `path` in `repo` at the version's revision ("eval": working tree)."""

    repo: str
    path: str
    pattern: str


@dataclass
class Row:
    section: str
    setting: str
    v1: str
    v2: str
    scope: str
    ev1: list[Ev] = field(default_factory=list)
    ev2: list[Ev] = field(default_factory=list)
    note: str = ""


class Sources:
    """Reads files at a revision with `git show`, caching the text."""

    def __init__(self, repos: dict[str, Path], revs: dict[tuple[str, str], str]):
        self.repos = repos
        self.revs = revs  # (version, repo) -> rev
        self._cache: dict[tuple[str, str, str], str] = {}

    def rev(self, version: str, repo: str) -> str | None:
        return None if repo == "eval" else self.revs[(version, repo)]

    def text(self, version: str, repo: str, path: str) -> str:
        rev = self.rev(version, repo)
        key = (repo, rev or "", path)
        if key not in self._cache:
            if rev is None:
                content = (REPO_ROOT / path).read_text(encoding="utf-8")
            else:
                out = subprocess.run(["git", "-C", str(self.repos[repo]), "show", f"{rev}:{path}"],
                                     capture_output=True, check=True)
                content = out.stdout.decode("utf-8", errors="replace")
            if Path(path).name == ".env":
                # ground rule 3: only model/region lines of an env file are kept in memory
                content = "\n".join(line for line in content.splitlines() if ENV_KEEP.match(line))
            self._cache[key] = content
        return self._cache[key]

    def check(self, version: str, ev: Ev) -> tuple[str | None, str]:
        """Return (error or None, `repo@rev:path:line`)."""
        rev = self.rev(version, ev.repo)
        where = f"{ev.repo}@{rev or 'worktree'}:{ev.path}"
        try:
            text = self.text(version, ev.repo, ev.path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return f"cannot read {where}: {e}", where
        m = re.search(ev.pattern, text, re.MULTILINE)
        if not m:
            return f"pattern not found in {where}: {ev.pattern!r}", where
        return None, f"{where}:{text.count(chr(10), 0, m.start()) + 1}"

    def pins(self, version: str, repo: str, path: str, names: list[str]) -> str:
        """`name==version` for each package, from a pinned requirements file (first pin wins)."""
        found: dict[str, str] = {}
        for line in self.text(version, repo, path).splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*(==|>=)\s*([^\s;#]+)", line)
            if m and m.group(1).lower() in {n.lower() for n in names}:
                found.setdefault(m.group(1).lower(), f"{m.group(1)}{m.group(2)}{m.group(3)}")
        return ", ".join(found.get(n.lower(), f"{n} (not listed)") for n in names)

    def npm(self, version: str, repo: str, path: str, names: list[str]) -> str:
        pkg = json.loads(self.text(version, repo, path))
        deps = {**pkg.get("devDependencies", {}), **pkg.get("dependencies", {})}
        return ", ".join(f"{n} {deps.get(n, '(not listed)')}" for n in names)


A6_LABELS = {"map_view": "map view", "case_upload": "case upload", "chat": "chat",
             "v2.7_deployed": "defense as deployed (no legal search)", "v2.7_full": "defense in full mode"}


def a6_table() -> tuple[Path, list[dict]]:
    """The latest A6 summary (scripts/94): one row per endpoint or defense arm."""
    path = latest_artefact_path(resolve(load_config("paths")["results"]) / "a6_latency_cost.csv")
    with open(path, encoding="utf-8") as f:
        return path, [r for r in csv.DictReader(f) if r["unit"] in A6_LABELS]


def a6_text(kind: str) -> str:
    _, rows = a6_table()
    if kind == "latency":
        parts = [f"{A6_LABELS[r['unit']]} {float(r['latency_p50_s']):,.1f} / {float(r['latency_p95_s']):,.1f} s" for r in rows]
        return "; ".join(parts) + " (30 calls each, Phase 9)"
    parts = [f"{A6_LABELS[r['unit']]} {int(float(r['prompt_tokens_mean'])):,} in + {int(float(r['output_tokens_mean'])):,} out "
             f"tokens, ${float(r['usd_mean']):.3f}" for r in rows if float(r["model_calls_mean"]) > 0]
    return "; ".join(parts) + " (means per call; output includes thinking); the map view makes no model call"


def a6_evidence() -> list[Ev]:
    path, rows = a6_table()
    rel = str(path.relative_to(REPO_ROOT))
    return [Ev("eval", rel, rf"^{re.escape(r['unit'])},") for r in rows]


def build_rows(src: Sources) -> list[Row]:
    B, S, F1, F2, E = "backend", "scrapers", "frontend_v1", "frontend_v2", "eval"
    corpus_freeze = json.loads((REPO_ROOT / "results/v2_corpus_freeze_summary.json").read_text(encoding="utf-8"))
    frozen = corpus_freeze["frozen"]
    v1_backend_pins = src.pins("v1", B, "requirements.txt", [
        "Flask", "langgraph", "langchain", "langchain-community", "langchain-core", "google-cloud-aiplatform",
        "vertexai", "google-genai", "umap-learn", "pynndescent", "numba", "scikit-learn", "numpy", "pandas"])
    v2_backend_pins = src.pins("v2", B, "requirements.txt", [
        "Flask", "google-adk", "google-genai", "google-cloud-aiplatform", "langchain", "langchain-community",
        "langgraph", "umap-learn", "pynndescent", "numba", "scikit-learn", "numpy", "pandas"])
    v1_scraper_pins = src.pins("v1", S, "requirements.txt", ["Scrapy", "google-genai", "PyMuPDF"])
    v2_scraper_pins = src.pins("v2", S, "crsnsp_scraper/requirements.txt",
                               ["Scrapy", "google-genai", "PyMuPDF", "umap-learn", "numpy"])
    v1_fe = src.pins("v1", F1, "requirements.txt", ["streamlit", "streamlit-authenticator", "plotly"])
    v2_fe = src.npm("v2", F2, "package.json", ["react", "react-dom", "@mui/material", "@mui/x-charts"])

    rows = [
        # ------------------------------------------------------------------ corpus
        Row("Corpus", "Decisions",
            "1,531 unique decisions (1,537 records in the snapshot)",
            f"{2390:,} decisions ({frozen['decisions']:,} frozen; 1 without a report dropped after re-extraction)",
            DIFF,
            [Ev(E, "PHASE_0_SUMMARY.md", r"1,537 records, which is \*\*1,531 unique decisions")],
            [Ev(E, f"config/{FREEZE_FILE}", r'"decisions": 2390'),
             Ev(E, "results/v2_corpus_freeze_summary.json", r'"frozen_decisions": 2391')]),
        Row("Corpus", "Sessions and period",
            "44 sessions, Nov 2019 – Feb 2025 (no 2020 decisions)",
            f"{frozen['sessions']} sessions, {frozen['date_min']} – {frozen['date_max']}",
            DIFF,
            [Ev(E, "PHASE_0_SUMMARY.md", r"44 sessions, Nov 2019 – Feb 2025, with no 2020 decisions")],
            [Ev(E, "results/v2_corpus_freeze_summary.json", r'"sessions": 90')]),
        Row("Corpus", "Snapshot date",
            "BigQuery export of 30 Mar 2025 (`data_cleaned.joblib` committed at the tag)",
            f"Production table read on {corpus_freeze['snapshot_date']} (table last modified "
            f"{corpus_freeze['table_last_modified_utc'][:10]}); reports and votes re-extracted 27 Sep 2026 (v2.1)",
            DIFF,
            [Ev(E, "PHASE_0_SUMMARY.md", r"exported 30 Mar 2025")],
            [Ev(E, "results/v2_corpus_freeze_summary.json", r'"snapshot_date": "2026-09-26"')]),
        Row("Corpus", "Comparison set (v1 vs v2)",
            "1,503 decisions present in both corpora, matched 1:1 by acórdão URL (1,491) or process number (12)",
            "same", SAME,
            [Ev(E, "results/v1_v2_match_summary.json", r'"matched": 1503')],
            [Ev(E, "results/v1_v2_match_summary.json", r'"url": 1491,\s*"process": 12')]),
        Row("Corpus", "Report and vote texts used in the evaluation",
            "v2.1 re-extraction (as v2)",
            "Re-extracted from the stored source documents with gemini-3.7-flash (temperature 0, thinking level "
            "low, max 65,535 output tokens, no input cut); long sections restored verbatim from the document",
            SHARED,
            [Ev(E, "DECISIONS.md", r"Both the v1 baseline methods and v2 are evaluated on the re-extracted reports")],
            [Ev(S, "crsnsp_scraper/llm_extractor.py", r"EXTRACTION_THINKING_LEVEL = os.environ.get\(\"EXTRACTION_THINKING_LEVEL\", \"low\"\)"),
             Ev(S, "crsnsp_scraper/llm_extractor.py", r"temperature=0,"),
             Ev(S, "crsnsp_scraper/llm_extractor.py", r"max_output_tokens=65535")],
            "Ground rule 9: the re-extraction is a data correction, applied to both versions' methods."),
        Row("Corpus", "Report extraction as deployed",
            "Scrapy spider: regular expressions and XPath on the acórdão HTML/PDF (PyMuPDF), no LLM",
            "LLM extraction (gemini-3.7-flash) of every field, as above",
            REPRO,
            [Ev(S, "crsnsp_scraper/spiders/crsnsp_spider.py", r"'acordao_relatorio': r\"TÓRIO\(\.\*\?\)Documento assinado\""),
             Ev(S, "crsnsp_scraper/spiders/crsnsp_spider.py", r"^import fitz")],
            [Ev(S, "crsnsp_scraper/llm_extractor.py", r"self.flash_model = os.environ.get\(\"GOOGLE_GENAI_MAIN_MODEL_NAME\", \"gemini-3.7-flash\"\)")]),

        # ------------------------------------------------------------------ models by role
        Row("Models by role", "Name curation (recorrente, recorrido, relator, advogado → `*_curated`)",
            "gemini-2.0-flash-001 (scrapers `data_pipeline/curate_names.py`, BigQuery pipeline)",
            "No separate curation step: the extraction prompt (gemini-3.7-flash) receives the names already in "
            "the corpus and must reuse their spelling",
            DIFF,
            [Ev(S, "data_pipeline/curate_names.py", r'model="gemini-2\.0-flash-001"'),
             Ev(E, "DECISIONS.md", r"v1 name curation: Gemini")],
            [Ev(B, "app.py", r'^\s*"acordao_recorrente",$'),
             Ev(S, "crsnsp_scraper/prompts/extraction_prompt.yaml", r"\{existing_names\}")],
            "Confirmed by the author (DECISIONS 2026-09-27). The GPT-4o script in the backend (`utils/curateNames.py`, "
            "Azure OpenAI) is MongoDB-era and did not produce the v1 snapshot."),
        Row("Models by role", "Outcome and ementa curation",
            "gemini-2.0-flash-001 (`data_pipeline/curate_resultado.py`, `curate_ementa.py`) → "
            "`acordao_resultado_curated`, `acordao_ementa_curated`",
            "Extracted with the other fields by gemini-3.7-flash; the outcome is validated against a closed label set",
            DIFF,
            [Ev(S, "data_pipeline/curate_resultado.py", r'model="gemini-2\.0-flash-001"'),
             Ev(S, "data_pipeline/curate_ementa.py", r'model="gemini-2\.0-flash-001"')],
            [Ev(S, "crsnsp_scraper/llm_extractor.py", r"def validate_resultado")]),
        Row("Models by role", "Report compression (retrieval text)",
            "n/a: v1 embeds the full report",
            "gemini-3.7-flash, temperature 0, thinking level low, max 8,192 output tokens (thinking included), "
            "extractive prompt (delete words, keep order); median 36% of the report's characters",
            DIFF,
            [Ev(S, "data_pipeline/create_vectors.py", r'"SELECT acordao_hash, acordao_relatorio FROM')],
            [Ev(B, "config/runtime.env", r"^COMPRESSION_MODEL=gemini-3\.7-flash"),
             Ev(B, "config/runtime.env", r"^COMPRESSION_THINKING_LEVEL=low"),
             Ev(B, "config/runtime.env", r"^COMPRESSION_MAX_OUTPUT_TOKENS=8192"),
             Ev(S, "crsnsp_scraper/embedding_generator.py", r"GenerateContentConfig\(temperature=0\.0"),
             Ev(E, "results/v2_rebuild_summary_2.json", r'"median": 0\.3587')]),
        Row("Models by role", "Corpus embeddings",
            "text-multilingual-embedding-002, 768 dimensions, task SEMANTIC_SIMILARITY, input = full report "
            "(single call; the model's input limit applies)",
            "gemini-embedding-001, 3,072 dimensions, task SEMANTIC_SIMILARITY. The stored vector is the unit-length "
            "sum of two unit-length embeddings: the full report (single call; the model's input limit applies) and "
            "the compressed report. An uploaded case gets its vector the same way",
            DIFF,
            [Ev(S, "data_pipeline/create_vectors.py", r'model="text-multilingual-embedding-002"'),
             Ev(S, "data_pipeline/create_vectors.py", r'task_type="SEMANTIC_SIMILARITY"'),
             Ev(S, "data_pipeline/create_vectors.py", r"output_dimensionality=768")],
            [Ev(S, "crsnsp_scraper/embedding_generator.py", r'task_type="SEMANTIC_SIMILARITY"'),
             Ev(S, "crsnsp_scraper/embedding_generator.py", r"output_dimensionality=3072"),
             Ev(S, "crsnsp_scraper/vector_map.py", r"total = a / norm_a \+ b / norm_b"),
             Ev(S, "crsnsp_scraper/embedding_generator.py",
                r"embeddings = self\.generate_fused_embeddings_batch\(list\(original_texts\), list\(valid_compressed\)\)"),
             Ev(B, "app.py", r"embeddings = fuse_embeddings\(full_embedding or None, compressed_embedding or None\)"),
             Ev(B, "config/runtime.env", r"^EMBEDDING_MODEL=gemini-embedding-001")],
            "Selected in Phase 4.1 on a development half of the queries (results/a1_candidates_selection.json). "
            "Until the v2.2 freeze the stored vector was the compressed report's embedding alone."),
        Row("Models by role", "Chat query embedding",
            "text-multilingual-embedding-002, task type not set (API default; the corpus used SEMANTIC_SIMILARITY)",
            "gemini-embedding-001, SEMANTIC_SIMILARITY, 3,072 dimensions (as the corpus)",
            DIFF,
            [Ev(B, "utils/gemini_config.py", r"response = embeddings_model\.get_embeddings\(\[text\]\)"),
             Ev(B, ".env", r"^EMBEDDING_MODEL=text-multilingual-embedding-002")],
            [Ev(B, "app.py", r'get_embedding\(user_message, task_type="SEMANTIC_SIMILARITY", output_dimensionality=3072\)')]),
        Row("Models by role", "Chat generation",
            "gemini-2.0-flash (`CHATBOT_MODEL`), Vertex AI us-central1",
            "gemini-3.7-flash (`CHATBOT_MODEL`), Vertex AI global endpoint",
            DIFF,
            [Ev(B, ".env", r"^CHATBOT_MODEL=gemini-2\.0-flash$"),
             Ev(B, "utils/gemini_config.py", r'region = os\.getenv\("VERTEX_REGION", "us-central1"\)')],
            [Ev(B, "config/runtime.env", r"^CHATBOT_MODEL=gemini-3\.7-flash"),
             Ev(B, "config/runtime.env", r"^GENAI_LOCATION=global"),
             Ev(B, "utils/runtime_config.py", r'"chat": "CHATBOT_MODEL"')]),
        Row("Models by role", "Defense agents",
            "Structure interpreter, case interpreter, legal searcher: gemini-2.0-flash (`CHATBOT_MODEL`); "
            "defense creator and supervisor review: gemini-2.5-flash-preview-04-17 (`CHATBOT_REASONING_MODEL`)",
            "Supervisor, case interpreter, legal searcher, structure interpreter, defense creator: gemini-3.7-flash "
            "(`CHATBOT_MODEL`); merger: gemini-3.7-flash (`CHATBOT_REASONING_MODEL`); single-prompt ablation: "
            "gemini-3.7-flash",
            DIFF,
            [Ev(B, ".env", r"^CHATBOT_REASONING_MODEL=gemini-2\.5-flash-preview-04-17$"),
             Ev(B, "agents/router.py", r'state, session_variables, create_gemini_client\("reasoning"\)\)\),'),
             Ev(B, "agents/router.py", r'reasoning_llm = create_gemini_client\(model_type="reasoning"\)')],
            [Ev(B, "config/runtime.env", r"^CHATBOT_REASONING_MODEL=gemini-3\.7-flash"),
             Ev(B, "utils/runtime_config.py", r'"merger": "CHATBOT_REASONING_MODEL"'),
             Ev(B, "utils/runtime_config.py", r'"agent": "CHATBOT_MODEL"')]),
        Row("Models by role", "Chat case summaries (hierarchical mode)",
            "n/a: mode does not exist in v1",
            "gemini-3.7-flash (`CHATBOT_MODEL`), one call per retrieved case, 8 in parallel, cached per process",
            V2ONLY, [Ev(B, "app.py", r"top_k = min\(10, len\(filtered_data_with_embeddings\)\)")],
            [Ev(B, "utils/runtime_config.py", r'"summary": "CHATBOT_MODEL"'),
             Ev(B, "app.py", r"CASE_SUMMARY_WORKERS = 8")],
            "Evaluation-only context mode (Phases 6–7); production uses full_text."),
        Row("Models by role", "OCR of uploaded PDFs and images",
            "gemini-2.0-flash-001 (hard-coded); used when a PDF yields < 50 characters of text, and for images",
            "gemini-3.7-flash (`OCR_MODEL`); same trigger",
            DIFF,
            [Ev(B, "utils/file_parsers.py", r'GenerativeModel\("gemini-2\.0-flash-001"\)'),
             Ev(B, "utils/file_parsers.py", r"if len\(content\.strip\(\)\) < 50:")],
            [Ev(B, "config/runtime.env", r"^OCR_MODEL=gemini-3\.7-flash"),
             Ev(B, "utils/file_parsers.py", r"if len\(content\.strip\(\)\) < 50:")]),

        # ------------------------------------------------------------------ generation settings
        Row("Generation settings", "Temperature",
            "Not set anywhere in the app (model defaults) for chat, agents and OCR",
            "Not set (model default) for chat, agents, OCR and case summaries; 0 for compression and corpus extraction",
            DIFF,
            [Ev(B, "app.py", r"response = gemini_model\.generate_content\(full_prompt\)"),
             Ev(B, "agents/router.py", r"response = chat\.send_message\(full_prompt\)")],
            [Ev(B, "app.py", r'role="chat", contents=full_prompt, config=chat_grounding\.generation_config\(chat_strategy\)'),
             Ev(B, "utils/chat_grounding.py", r'DEFAULT_CHAT_STRATEGY = "quoted_full"'),
             Ev(B, "utils/chat_grounding.py", r'"quoted_full": \{"rules": True, "quotes": True, "complete": True\}'),
             Ev(B, "app.py", r"temperature=0\.0,  # Zero temperature for consistency")],
            note="The v2.5 freeze set the chat's temperature to 0; v2.6 selects a chat strategy that leaves it at the "
                 "model default (DECISIONS 2026-10-12)."),
        Row("Generation settings", "Max output tokens",
            "Not set (model defaults)",
            "Not set (model default) except compression (8,192) and corpus extraction (65,535); thinking tokens count "
            "toward the limit",
            DIFF,
            [Ev(B, "agents/CaseInterpreterAgent.py", r"response = chat\.send_message\(messages_content\)")],
            [Ev(B, "app.py", r"max_output_tokens=compression_max_output_tokens\(\)")]),
        Row("Generation settings", "Thinking",
            "gemini-2.0-flash: no thinking; gemini-2.5-flash-preview (creator, review): model default (dynamic)",
            "gemini-3.7-flash always thinks (cannot be switched off); level not set (model default) for chat and "
            "agents; `low` for compression and corpus extraction",
            DIFF,
            [Ev(B, ".env", r"^CHATBOT_REASONING_MODEL=gemini-2\.5-flash-preview-04-17$")],
            [Ev(B, "utils/runtime_config.py", r'DEFAULT_THINKING_LEVELS = \{"compression": "low"\}')]),
        Row("Generation settings", "Retries and timeouts",
            "None in the code (SDK defaults); an agent error ends the run; gunicorn `--timeout 0` (no limit)",
            "HTTP retries on 408/429/5xx (6 attempts, exponential backoff 2–60 s); 240 s per call; primary key "
            "(6 attempts) then fallback credentials, primary skipped for 60 s after a failure; a blocked "
            "(RECITATION/SAFETY) response is retried once; gunicorn and Cloud Run request timeout 600 s (300 s until "
            "v2.7-eval-frozen)",
            DIFF,
            [Ev(B, "Dockerfile", r"--timeout 0 app:app")],
            [Ev(B, "utils/runtime_config.py", r"RETRY_STATUS_CODES = \[408, 429, 500, 502, 503, 504\]"),
             Ev(B, "utils/runtime_config.py", r'attempts=int\(os\.getenv\("GENAI_RETRY_ATTEMPTS"\) or 6\)'),
             Ev(B, "utils/runtime_config.py", r'os\.getenv\("GENAI_TIMEOUT_S"\) or 240'),
             Ev(B, "agents_adk/common.py", r"for attempt in range\(2\):"),
             Ev(B, "Dockerfile", r"--timeout 600"), Ev(B, "deploy-flags.yaml", r"--timeout: 600")]),

        # ------------------------------------------------------------------ map
        Row("Map", "Projection",
            "Global: one UMAP fitted offline on all 1,537 snapshot vectors (`utils/precompute_umap_mapper.py`); each "
            "map request projects the filtered subset with `mapper.transform()`; not refitted per subset",
            "Global: one UMAP fitted on the 2,390 v2 vectors; the map shows the stored fit coordinates; an uploaded "
            "case is projected with `mapper.transform()`",
            DIFF,
            [Ev(B, "utils/umap_plot.py", r"embeddings_2D = self\.mapper\.transform\("),
             Ev(B, "utils/umap_plot.py", r"self\.mapper = self\.reducer\.fit\(embedding_array\)")],
            [Ev(B, "utils/umap_plot.py", r'doc\.get\("embedding_acordao_relatorio_compressed_2d", \[0, 0\]\)\[0\]'),
             Ev(B, "app.py", r"coordinates_2d = umap_helper\.mapper\.transform\(embeddings_array\)")],
            "Global by design in both versions: the map is refitted when the dataset is updated, never on filtering "
            "(the author, DECISIONS 2026-09-27). Checked on the pickled mapper at the tag: "
            "1,537 × 768 training vectors; projected points land within ~0.2% of the map's span of their fitted "
            "positions whatever the subset."),
        Row("Map", "UMAP parameters",
            "cosine, n_neighbors 15, min_dist 0.1, 2 components, random_state None (fit not seeded); "
            "transform_seed 42 (umap-learn 0.5.7 default)",
            "cosine, n_neighbors 10, min_dist 0, 2,000 epochs, 2 components, random_state 42; rows in "
            "`acordao_hash` order",
            DIFF,
            [Ev(B, "utils/umap_plot.py", r"self\.reducer = UMAP\(metric='cosine', \n\s*n_neighbors=15, \n\s*min_dist=0\.1, \n\s*n_components=2\)")],
            [Ev(S, "crsnsp_scraper/vector_map.py", r'"n_neighbors": 10,\n\s*"min_dist": 0\.0,\n\s*"n_components": 2,\n\s*"metric": "cosine",\n\s*"n_epochs": 2000,\n\s*"random_state": 42'),
             Ev(S, "crsnsp_scraper/umap_precompute.py", r"self\.umap_params = dict\(UMAP_PARAMS\)"),
             Ev(S, "crsnsp_scraper/umap_precompute.py", r"ORDER BY acordao_hash"),
             Ev(E, f"config/{FREEZE_FILE}", r'"n_neighbors": 10,'),
             Ev(E, f"config/{FREEZE_FILE}", r'"n_epochs": 2000,')],
            "Selected in Phase 5.1 (results/a2_candidates_selection_2.json). Until the v2.2 freeze: n_neighbors 30, "
            "min_dist 0.01, the library's default 500 epochs, rows in the order of the input."),
        Row("Map", "Input vectors and reproducibility",
            "768-d embeddings of the full report; the fit cannot be reproduced (no seed)",
            "The 3,072-d stored vectors (full report + compressed report); a seed-42 refit in `acordao_hash` order "
            "returns the stored coordinates exactly with the pinned libraries (umap-learn 0.5.7; numba 0.61.0 and "
            "0.62.1 both checked), and a different map with another row order or newer libraries",
            DIFF,
            [Ev(B, "utils/umap_plot.py", r"self\.embedding_field = 'embedding_acordao_relatorio'")],
            [Ev(E, f"config/{FREEZE_FILE}", r'"refit_identical": true'),
             Ev(E, "results/a2_served_map_reproduction_2.json", r'"libraries": "backend environment",(?s:.{0,400}?)"row_order": "as served",\s*"identical": true'),
             Ev(S, "crsnsp_scraper/rebuild_v2_local.py", r"if not np\.array_equal\(coords, coords_refit\):")]),

        # ------------------------------------------------------------------ chat (RAG)
        Row("Chat (RAG)", "Candidate cases",
            "The cases matching the dashboard filters; the frontend refuses more than 10",
            "After a case is uploaded, the 10 decisions most similar to it by cosine similarity of the retrieval "
            "vectors are pre-selected; the user confirms or changes the selection by clicking on the map, at most "
            "10 (frontend); evaluation: `case_hashes` (EVAL_MODE=1)",
            DIFF,
            [Ev(F1, "app.py", r"if len\(st\.session_state\['filtered_data_ids'\]\) > 10:")],
            [Ev(B, "utils/similar_cases.py", r"PRESELECTED_CASES = 10"),
             Ev(B, "app.py", r"umap_helper\.most_similar_cases\(embeddings, PRESELECTED_CASES\)"),
             Ev(F2, "src/App.js", r"setSelectedAccords\(accords\.slice\(0, 10\)\)"),
             Ev(F2, "src/components/Home.js", r"onPreselectAccords\(similarCasesToAccords\(uploaded\)\)"),
             Ev(F2, "src/App.js", r"if \(prevSelectedAccords\.length >= 10\)"),
             Ev(B, "utils/runtime_config.py", r'raise PermissionError\("case_hashes is only accepted when EVAL_MODE=1"\)')],
            "Until the v2.3 freeze the user selected every case by clicking on the map (the author, DECISIONS 2026-10-10)."),
        Row("Chat (RAG)", "Retrieval (`top_k`)",
            "min(10, n) cases by cosine similarity of query and report embeddings; with ≤ 10 candidates every case is "
            "used, so retrieval only orders them",
            "Same (default 10); evaluation: `top_k` 1–100 per request, at most 30 in hierarchical mode",
            SAME,
            [Ev(B, "app.py", r"top_k = min\(10, len\(filtered_data_with_embeddings\)\)")],
            [Ev(B, "utils/runtime_config.py", r"DEFAULT_CHAT_TOP_K = 10"),
             Ev(B, "utils/runtime_config.py", r"MAX_CHAT_TOP_K = 100"),
             Ev(B, "utils/runtime_config.py", r"HIERARCHICAL_MAX_CASES = 30")]),
        Row("Chat (RAG)", "Context format",
            "One prompt: system prompt, data dictionary, then 17 labelled fields per case (session, parties, "
            "rapporteur, lawyer, ementa, penalty, legal basis, acórdão, participants, full report, full vote, outcome; "
            "curated names, ementa and outcome), then the whole conversation history and the question",
            "Same format and fields, with names, ementa and outcome as extracted (no `*_curated` copies); "
            "evaluation-only `hierarchical_summary` mode: a ≤ 300-token summary per case plus 12 metadata fields",
            DIFF,
            [Ev(B, "app.py", r'"acordao_recorrente_curated",'),
             Ev(B, "app.py", r'f"Contexto de Casos Relevantes \(Top \{top_k\}\):\\n\{context\}"')],
            [Ev(B, "app.py", r'f"Contexto de Casos Relevantes \(Top \{top_k\}\):\\n\{context\}"'),
             Ev(B, "prompts/chat_case_summary.yaml", r"300 tokens")]),
        Row("Chat (RAG)", "Chat system prompt",
            "`prompts/system_prompt.yaml`",
            "Same file plus one rule (defense strategies only as found in the cases, and whether they worked), "
            "followed by rules that prevail over it (v2.6). Six grounding rules: every statement names its case; "
            "nothing beyond what the cases state; a statement about several cases only if each supports it; values "
            "and outcomes as written; no evaluation or interpretation; say when the cases do not tell. Three quoting "
            "rules: after each statement about a case, a literal quote of that case of at most 30 words, copied "
            "unchanged; no statement without a quote. Three completeness rules: every part of the question and every "
            "main point of each pertinent case; nothing left out for brevity; a closing comparison built on quoted "
            "statements",
            DIFF,
            [Ev(B, "prompts/system_prompt.yaml", r"^your_role:")],
            [Ev(B, "prompts/system_prompt.yaml", r"estratégias de defesa"),
             Ev(B, "prompts/chat_grounding.yaml", r"^grounding_rules:"),
             Ev(B, "prompts/chat_grounding.yaml", r"^quote_rules:"),
             Ev(B, "prompts/chat_grounding.yaml", r"^completeness_rules:"),
             Ev(B, "utils/chat_grounding.py", r"base_prompt\.rstrip\(\) \+ .*grounding_heading"),
             Ev(B, "app.py", r"system_prompt = chat_grounding\.system_prompt\(chat_strategy, load_system_prompt\(\)\)")],
            note="The rules were added to reduce statements that the cases do not support (Phase 6.1; v2.5 had the "
                 "grounding rules and a rule preferring a short answer, at temperature 0)."),
        Row("Chat (RAG)", "Check of the answer",
            "None",
            "Every quote of the answer is looked up, literally (case, accents and punctuation aside), in the text "
            "given to the model: in the case the sentence names, or in any supplied case when it names none or several. "
            "A sentence with a quote that is not found is removed and the answer says how many were removed (v2.6)",
            DIFF,
            [Ev(B, "app.py", r"response = gemini_model\.generate_content\(full_prompt\)")],
            [Ev(B, "utils/chat_grounding.py", r"def check_quotes\(answer: str, cases: dict\[str, str\]\)"),
             Ev(B, "app.py", r"assistant_message, quote_check = chat_grounding\.check_quotes\(assistant_message, case_texts\)"),
             Ev(B, "prompts/chat_grounding.yaml", r"^removed_note:")]),

        # ------------------------------------------------------------------ defense generation
        Row("Defense generation", "Agent framework and order",
            "LangGraph `StateGraph` with a fixed sequence: structure interpreter → case interpreter → legal "
            "searcher → defense creator → supervisor review → END",
            "Google ADK: SupervisorAgent (clarification) → SequentialAgent[ParallelAgent(case interpreter, legal "
            "searcher, structure interpreter) → defense creator → merger]",
            DIFF,
            [Ev(B, "agents/router.py", r"builder = StateGraph\(MessagesState\)"),
             Ev(B, "agents/router.py", r"builder\.add_edge\(last_agent_name, END\)")],
            [Ev(B, "agents_adk/workflow_coordinator.py", r"parallel_research = ParallelAgent\("),
             Ev(B, "agents_adk/workflow_coordinator.py", r"self\.main_agent = SequentialAgent\(")]),
        Row("Defense generation", "Model calls per draft (no retries)",
            "6: structure, case, legal-search query extraction, legal-search answer, creator, review",
            "6 with the supervisor, 5 without (`skip_supervisor`, EVAL_MODE=1); single-prompt ablation: 1; "
            "the fact controls (v2.7) add 2 (case fact sheet, premise check); without the legal search, 1 fewer",
            DIFF,
            [Ev(B, "agents/LegalSearcherAgent.py", r"search_response = chat\.send_message\(search_prompt\)"),
             Ev(B, "agents/LegalSearcherAgent.py", r"final_response = chat\.send_message\(messages_content\)")],
            [Ev(E, "results/v2_smoke_test_6.json", r'(?s)"defense_full": \{.{0,400}?"llm_calls": 5,'),
             Ev(E, "results/v2_smoke_test_6.json", r'(?s)"defense_full_via_supervisor": \{.{0,400}?"llm_calls": 6,'),
             Ev(B, "utils/fact_controls.py", r"def extract_case_facts"), Ev(B, "utils/fact_controls.py", r"def check_premises")]),
        Row("Defense generation", "Precedents given to the agents",
            "The selected cases (≤ 10), 17 fields each (full report and vote), inserted only in the case "
            "interpreter's prompt; later agents see its output",
            "The selected cases (≤ 10), same 17 fields, inserted in the prompts of the supervisor, case interpreter, "
            "legal searcher, defense creator and merger",
            DIFF,
            [Ev(B, "prompts/case_interpreter_prompt.yaml", r"\{context_cases\}")],
            [Ev(B, "prompts/merger_consolidated.yaml", r"\{context_cases\}"),
             Ev(B, "prompts/legal_searcher_consolidated.yaml", r"\{context_cases\}"),
             Ev(B, "prompts/defense_creator_consolidated.yaml", r"\{context_cases\}")]),
        Row("Defense generation", "Uploaded case (notice) given to the agents",
            "Case interpreter prompt; appended to the defense creator's input; later agents see earlier outputs",
            "Supervisor and case interpreter prompts only (`{case_file_content}`); the legal searcher, creator and "
            "merger prompts interpolate only the typed description (`{case_description}`)",
            DIFF,
            [Ev(B, "prompts/case_interpreter_prompt.yaml", r"\{case_file_content\}"),
             Ev(B, "agents/DefenseCreatorAgent.py", r"Case File Content: \{case_file_content\}")],
            [Ev(B, "prompts/case_interpreter_consolidated.yaml", r"\{case_file_content\}"),
             Ev(B, "prompts/legal_searcher_consolidated.yaml", r"^  \{case_description\}")],
            "See results/agent_protocol.md: with an uploaded notice and no typed description, v2's legal searcher "
            "does not see the notice. Phase 8 supplies notices as uploads only, as users do (DECISIONS 2026-09-27)."),
        Row("Defense generation", "Template",
            "Uploaded template; if none, `agents/defense_structure.json`",
            "Uploaded template; if none, the prompt's `default_template` (also the `no_template` ablation)",
            DIFF,
            [Ev(B, "agents/DefenseStructureInterpreterAgent.py", r'"defense_structure\.json"')],
            [Ev(B, "prompts/defense_structure_consolidated.yaml", r"^default_template:")]),
        Row("Defense generation", "Fact grounding of the draft",
            "None",
            "Prompt rule (creator, merger) plus a deterministic check (`utils/fact_guard.py`): unsupported case "
            "identifiers replaced by `[A PREENCHER: ...]`, unsupported dates, amounts and legal acts flagged",
            DIFF,
            [Ev(B, "agents/router.py", r"def run_agents")],
            [Ev(B, "utils/fact_guard.py", r"def check_draft"),
             Ev(B, "prompts/merger_consolidated.yaml", r"^case_facts_rule:")]),
        Row("Defense generation", "Fact controls (v2.7)",
            "None",
            "Three controls, on by default (`FACT_CONTROLS`): a premise rule (a ground taken from a precedent states "
            "its factual premise as `[VERIFICAR: ...]` unless the case gives it), a case fact sheet extracted from "
            "the uploaded case before drafting (one model call), and a premise check that tags unsupported factual "
            "passages of the final draft with `[VERIFICAR: ...]` (one model call). With all three off, v2.6's prompts",
            DIFF,
            [Ev(B, "agents/router.py", r"def run_agents")],
            [Ev(B, "utils/fact_controls.py", r'^FACT_CONTROLS = \("premise_rule", "case_facts", "premise_check"\)'),
             Ev(B, "config/runtime.env", r"^FACT_CONTROLS=premise_rule,case_facts,premise_check")],
            "Added after the v2.6 evaluation (Phase 8 improvement round, PHASE_8_SUMMARY)."),
        Row("Defense generation", "Output to the client",
            "Server-sent events: the last message after each node (5), plus a system message on errors",
            "Server-sent events: one message per ADK event (model responses are not token-streamed), a "
            "'Verificação de Fatos' message after the final draft, a system message at the end",
            DIFF,
            [Ev(B, "agents/router.py", r'stream_mode="values"')],
            [Ev(B, "agents_adk/workflow_coordinator.py", r"yield fact_check_message\(fact_report\)")]),
        Row("Defense generation", "Ablation switches",
            "n/a (v1 is not modified)",
            "`ablation_mode` ∈ {full, single_prompt, no_legal_search, no_precedents, no_template}; code default full; "
            "deployed with `ABLATION_MODE=no_legal_search` (tag v2.7, 2026-10-25)",
            V2ONLY,
            [Ev(B, "agents/router.py", r"def create_agent_graph")],
            [Ev(B, "utils/runtime_config.py", r'ABLATION_MODES = \("full", "single_prompt", "no_legal_search", "no_precedents", "no_template"\)'),
             Ev(B, "config/runtime.env", r"^ABLATION_MODE=no_legal_search")]),

        # ------------------------------------------------------------------ legal search
        Row("Legal search", "Tool and parameters",
            "Tavily (`langchain_community` TavilySearchResults): search_depth advanced, max_results 10, "
            "include_domains planalto.gov.br and the CRSNSP legislation page; up to 3 queries that the model "
            "extracts from the previous agent's output",
            "Google Search grounding (Gemini built-in tool), no domain restriction, queries chosen by the model. "
            "A Tavily tool (max_results 5, advanced) is still constructed but not attached to the agent",
            DIFF,
            [Ev(B, "agents/LegalSearcherAgent.py", r'search_depth="advanced"'),
             Ev(B, "agents/LegalSearcherAgent.py", r"max_results=10"),
             Ev(B, "agents/LegalSearcherAgent.py", r'include_domains=\["planalto\.gov\.br"'),
             Ev(B, "agents/LegalSearcherAgent.py", r"search_queries = queries\[:3\]")],
            [Ev(B, "agents_adk/legal_searcher_agent.py", r"tools: list\[Any\] = \[google_search_grounding\]"),
             Ev(B, "agents_adk/common.py", r"types\.Tool\(google_search=types\.GoogleSearch\(\)\)"),
             Ev(E, f"config/{FREEZE_FILE}", r"the Tavily tool is not attached")],
            "The plan lists Tavily for v2; v2 uses Google Search grounding (recorded in the v2.1 CHANGELOG entry)."),

        # ------------------------------------------------------------------ software
        Row("Software", "Python and web server",
            "Python 3.12 (Docker `python:3.12-slim`), Flask, gunicorn 1 worker × 8 threads",
            "Python 3.12 (Docker `python:3.12-slim`), Flask, gunicorn 1 worker × 8 threads",
            SAME,
            [Ev(B, "Dockerfile", r"^FROM python:3\.12-slim"), Ev(B, "Dockerfile", r"--workers 1 --threads 8")],
            [Ev(B, "Dockerfile", r"^FROM python:3\.12-slim"), Ev(B, "Dockerfile", r"--workers 1 --threads 8")]),
        Row("Software", "Backend packages (pinned)", v1_backend_pins, v2_backend_pins, REPRO,
            [Ev(B, "requirements.txt", r"^langgraph==")], [Ev(B, "requirements.txt", r"^google-adk==")]),
        Row("Software", "Data pipeline packages", v1_scraper_pins, v2_scraper_pins, REPRO,
            [Ev(S, "requirements.txt", r"(?i)^scrapy")], [Ev(S, "crsnsp_scraper/requirements.txt", r"(?i)^scrapy")]),
        Row("Software", "Frontend", f"Streamlit ({v1_fe})", f"React ({v2_fe})", DIFF,
            [Ev(F1, "requirements.txt", r"^streamlit==")], [Ev(F2, "package.json", r'"react":')]),

        # ------------------------------------------------------------------ operation
        Row("Operation", "Where it ran for the evaluation",
            "Cloud Run production service, used by the experts 10 May – 11 Jun 2025",
            "Local Flask server of the frozen tag (`EVAL_MODE=1`, frozen data files, `crsnsp_eval` table) calling "
            "Vertex AI; not production Cloud Run",
            DIFF,
            [Ev(E, "DECISIONS.md", r"Version 1 is taken as it was when the experts used it")],
            [Ev(E, "config/v2_server.env", r"^EVAL_MODE=1")]),
        Row("Operation", "Latency per request (p50 / p95)",
            "n/a: v1's generation models are shut down (Gemini 2.0 Flash, 1 Jun 2026), so v1 cannot be timed",
            a6_text("latency"),
            DIFF, [Ev(E, "config/models.yaml", r"generation: gemini-2\.0-flash\s+# shut down 1 Jun 2026")],
            a6_evidence()),
        Row("Operation", "Tokens and cost per request",
            "n/a: as above",
            a6_text("cost"),
            DIFF, [Ev(E, "config/models.yaml", r"shut down 1 Jun 2026")], [*a6_evidence(), Ev(E, "config/prices.yaml", r"gemini-3\.7-flash:")],
            "Measured on a local server calling the cloud APIs, one request at a time, not on production Cloud Run "
            "(Phase 9). Costs: the model calls the server logged, at Standard PayGo prices of `config/prices.yaml`; "
            "not counted: Google Search grounding in full mode ($14 per 1,000 queries; the queries are not recorded) "
            "and the embeddings of an upload (at most $0.0006)."),
    ]
    return rows


def write_markdown(path: Path, rows: list[Row], revs_text: str) -> None:
    lines = [
        "# Reproducibility table: JurisMap v1 and v2",
        "",
        "Generated by `scripts/20_settings_table.py`; do not edit by hand. Every value is checked against the code or",
        "records named in `reproducibility_table.csv` (columns `evidence_v1`, `evidence_v2`).",
        "",
        revs_text,
        "",
        "`Scope` says how the paper may use a row: a *version difference*, a setting *shared* by both versions'",
        "evaluations (ground rule 9: data corrections are not version differences), *reproducibility only*, or",
        "*v2 evaluation instrumentation* (absent from production behaviour).",
    ]
    section = None
    for r in rows:
        if r.section != section:
            section = r.section
            lines += ["", f"## {section}", "", "| Setting | v1 | v2 | Scope |", "| --- | --- | --- | --- |"]
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
        lines.append(f"| {esc(r.setting)} | {esc(r.v1)} | {esc(r.v2)} | {r.scope} |")
    notes = [r for r in rows if r.note]
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- **{r.setting}:** {r.note}" for r in notes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", default=FREEZE_FILE, help="freeze file in config/ defining v2")
    ap.add_argument("--v2-backend-rev", help="the backend of v2 at this revision instead of the freeze tag (v2.7: the deployed tag)")
    args = ap.parse_args()

    paths = load_config("paths")
    freeze = json.loads((CONFIG_DIR / args.freeze).read_text(encoding="utf-8"))
    v1_tag, v2_tag = paths["v1_tag"], freeze["tag"]
    repos = {name: resolve(p) for name, p in paths["repos"].items()}
    revs = {
        ("v1", "backend"): v1_tag, ("v1", "scrapers"): v1_tag, ("v1", "frontend_v1"): v1_tag,
        ("v2", "backend"): args.v2_backend_rev or v2_tag, ("v2", "scrapers"): v2_tag,
        ("v2", "frontend_v2"): freeze["git"]["frontend_v2"]["sha"],
    }
    src = Sources(repos, revs)
    out_dir = resolve(paths["results"])

    with RunManifest(SCRIPT, out_dir, params={"freeze_file": args.freeze, "revisions": {f"{v}/{r}": rev for (v, r), rev in revs.items()}},
                     record_outputs=True) as run:
        rows = build_rows(src)
        errors, evidence = [], []
        for r in rows:
            refs = {}
            for version, evs in (("v1", r.ev1), ("v2", r.ev2)):
                found = []
                for ev in evs:
                    err, where = src.check(version, ev)
                    if err:
                        errors.append(f"{r.section} / {r.setting} ({version}): {err}")
                    found.append(where)
                refs[version] = found
            evidence.append(refs)
        if errors:
            raise SystemExit("Evidence check failed; nothing written:\n  " + "\n  ".join(errors))

        csv_path = new_artefact_path(out_dir / "reproducibility_table.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["section", "setting", "v1", "v2", "paper_scope", "evidence_v1", "evidence_v2", "note"])
            for r, refs in zip(rows, evidence):
                w.writerow([r.section, r.setting, r.v1, r.v2, r.scope, "; ".join(refs["v1"]), "; ".join(refs["v2"]), r.note])
        md_path = csv_path.with_suffix(".md")
        backend_text = (f"backend `{freeze['git']['backend']['sha'][:7]}`" if not args.v2_backend_rev else
                        f"backend at `{args.v2_backend_rev}` "
                        f"(`{subprocess.run(['git', '-C', str(repos['backend']), 'rev-parse', '--short=7', args.v2_backend_rev + '^{commit}'], capture_output=True, text=True, check=True).stdout.strip()}`)")
        revs_text = (f"v1 = tag `{v1_tag}` (backend, scrapers, Streamlit frontend). v2 = tag `{v2_tag}` "
                     f"({backend_text}, scrapers `{freeze['git']['scrapers']['sha'][:7]}`; "
                     f"React frontend `{revs[('v2', 'frontend_v2')][:7]}`), defined by `config/{args.freeze}`.")
        write_markdown(md_path, rows, revs_text)
        run.add_output(csv_path, "reproducibility table (Phase 2)")
        run.add_output(md_path, "reproducibility table, Markdown")
        n_items = sum(len(r.ev1) + len(r.ev2) for r in rows)
        run.data["summary"] = {"rows": len(rows), "evidence_items_checked": n_items,
                               "pending": [r.setting for r in rows if "pending" in (r.v1 + r.v2)]}
        print(f"{len(rows)} rows, {n_items} evidence items checked -> {csv_path.name}, {md_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
