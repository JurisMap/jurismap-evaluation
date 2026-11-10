# JurisMap evaluation: code, data and results

Code, data and results of the technical evaluation of JurisMap reported in the paper *JurisMap: A Visual
Analytics and Generative AI Approach to Automating Administrative Legal Defense Production* (Mentzingen,
António and Bacao). JurisMap retrieves and maps decisions of Brazil's National Private Insurance System
Appeals Board (CRSNSP), answers questions about selected decisions and drafts administrative appeals.

The paper reports two versions of the system:

- **Version 1** was evaluated by five legal experts (B2, B3). Its expert questionnaire is not released;
  the results give the aggregate statistics and the quotations the paper uses.
- **Version 2** is evaluated here: retrieval (A1), the similarity map (A2), the chatbot (A3) and its context
  size (A5), the defense drafts and an ablation of the drafting pipeline (A4), and latency and cost (A6).

The system's own code (backend, frontend, data pipeline) and its prompts are not part of this release.
This repository talks to the system only through its HTTP API and exported data files, and it contains a
**reference pipeline** (`reference/`) that reproduces version 2's retrieval vectors and map with public
libraries.

## Contents

| Path | Contents |
| --- | --- |
| `src/jm_eval/` | The evaluation package: run manifests, LLM response cache and judge clients (`llm.py`), statistics, and one subpackage per experiment (`retrieval/`, `mapping/`, `chatbot/`, `drafting/`) |
| `scripts/` | Numbered entry points, one per step (see below) |
| `config/` | Every setting of every step, fixed before the step ran: models and prices, label definitions, methods, candidates and selection rules, question templates, rubric, pacing; `v2.*_freeze.json` define the frozen system versions |
| `reference/` | The reference pipeline: embedding call, fusion, top-k cosine retrieval, UMAP with the frozen parameters; `check_reference.py` |
| `results/` | Every table, figure and example file behind the paper, with the run manifests that wrote them |
| `tests/` | Unit tests (`pytest`) |
| `data/` | Empty in git except `MANIFEST.csv` and `README.md`: the data archive (Zenodo) is extracted here |
| `analysis/b2/` | The statistics of the expert questionnaire (B2), as computed for version 1's evaluation |

**Judge prompts and rubric.** The prompts of the LLM judges, the verifiers and the generators of test
material are in the code, next to where they are used: `src/jm_eval/chatbot/judging.py` (claim
decomposition and verification, completeness, key points and reference points), `src/jm_eval/drafting/judging.py`
(the drafting rubric, argument recall, fact sorting) and `src/jm_eval/drafting/notices.py` (reconstructed
notices and the appellants' argument lists). The rubric's scales and the judging settings are in
`config/a4_drafting.yaml` and `config/a3_chatbot.yaml`.

**The system's prompts and agent protocol** are described in `results/prompt_specs.md` and
`results/agent_protocol.md` (version 1, and version 2 up to its final freeze, v2.7, with the configuration
deployed), quoting at most 25 words of any prompt. The agent graphs are in `results/figures/`.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/JurisMap/jurismap-evaluation.git && cd jurismap-evaluation
uv sync --frozen                 # analysis of the released data
uv sync --frozen --extra map     # A2 projections (umap-learn, openTSNE, PaCMAP)
uv sync --frozen --extra local   # local models: BGE-M3, multilingual-E5, the NLI model (downloads on first use)
uv sync --frozen --extra gcp     # model calls on Vertex AI
uv run pytest
```

Then download the data archive `jurismap-evaluation-data.zip` from Zenodo
(https://doi.org/10.5281/zenodo.24065101), extract it at the repository root and check it:

```bash
unzip jurismap-evaluation-data.zip    # files land under data/
uv run python scripts/verify_data.py
```

## What can be reproduced, and how

Every script writes its outputs without overwriting: a rerun writes `<name>_<n>.<ext>` next to the released
file, and the scripts read the newest version of each input. Each released result has the run manifest
that wrote it (`results/*.manifest.json`, and `*.manifest.json` in the data archive): the git commits,
model IDs and parameters, config hashes, seeds, timestamps and package versions. A manifest's `argv` is
the command that produced the file.

**1. From the released data, with no model call.** These scripts rebuild the tables and figures from the
data archive. They were rerun from a copy of this release with the network blocked, and their outputs
compared with the released ones (see "Verification" below).

| Experiment | Script | Main outputs in `results/` |
| --- | --- | --- |
| A1 silver labels | `40_silver_labels.py` with `corpus: data/v2/corpus_v2_rebuilt_20260927.parquet` in `config/a1_labels.yaml` (a) | `a1_label_coverage_*.csv`, `a1_label_summary_*.json`, `data/labels/a1_silver_labels_*.parquet` |
| A1 retrieval benchmark | `46_a1_retrieval.py` | `a1_retrieval_main_*.csv`, `a1_retrieval_contrasts_*.csv`, `a1_outcome_consistency_*.csv`, `a1_retrieval_by_complexity*.csv`, `figures/a1_retrieval_*.pdf` |
| A1 retrieval vector of version 2 | `56_a1_candidate_selection.py` | `a1_candidates.csv`, `a1_candidates_contrasts.csv`, `a1_candidates_selection.json`, `figures/a1_candidates.pdf` |
| A1 costs | `48_phase4_costs.py` (b) | `costs_phase4_*.csv` |
| A2 map validation | `51_a2_map_metrics.py` | `a2_map_metrics_*.csv`, `a2_seed_stability_*.csv`, `a2_map_summary_*.json`, `figures/a2_*.pdf` |
| A2 map configuration of version 2 | `54_a2_candidate_selection.py` | `a2_candidates_*.csv`, `a2_candidates_selection_*.json`, `figures/a2_candidates_*.pdf` |
| A3 case sets | `60_a3_case_sets.py` | `a3_case_sets_*.csv` |
| A3 chatbot | `68_a3_report.py` | `a3_chatbot_*.csv`, `a3_claims_*.parquet`, `a3_judge_agreement_*.csv`, `a3_examples_*.md`, `costs_phase6_*.csv` |
| A3 chat strategies (development sets) | `70_a3_candidate_selection.py` | `a3_candidates_*.csv`, `a3_candidates_selection_*.json` |
| A3 audit of the verifiers | `71_a3_claim_audit.py --apply results/a3_claim_audit_3.xlsx` | `a3_claim_audit.csv`, `a3_claim_audit.json` |
| A5 context size | `73_a5_report.py` | `a5_context_size_*.csv`, `a5_context_size_paired_*.csv`, `figures/a5_context_size_*.pdf`, `costs_phase7_*.csv` |
| A4 automatic metrics | `84_a4_auto_metrics.py` | `a4_drafting_*.csv`, `a4_drafting_by_mode_*.csv`, `a4_citations_unresolved_*.csv`, `a4_statutes_review_*.csv` (and `a4_v27_*`) |
| A4 ablation and judges | `88_a4_report.py` | `a4_drafting_scores_*.csv`, `a4_ablation_stats_*.csv`, `a4_judge_agreement_*.csv`, `failure_modes_*.csv`, `figures/a4_ablation_*.pdf` (and `a4_v27_*`) |
| A4 invented facts | `90_a4_invented_facts.py` | `a4_fact_audit_agreement_*.csv`, `a4_invented_facts_*.csv` (and `a4_v27_*`) |
| A4 development arms | `92_a4_dev_selection.py` | `a4_dev_arms_*.csv` |
| A6 latency and cost | `94_a6_report.py` | `a6_latency_cost.csv`, `a6_latency_paired.csv`, `a6_latency_by_round.csv`, `costs_phase9.csv` |
| A4 drafting before and after the fact controls (v2.6 against v2.7) | `97_a4_version_comparison.py` | `a4_versions.csv` |

(a) The labels were computed on the version 2.2 corpus file (same decisions and texts, rows in another
order); the 60-decision development sample is drawn by row order, so the labels file is reproduced only
from that file, which the data archive includes. (b) The cost tables sum every run manifest of their phase. `costs_phase6_*.csv` (from `68_a3_report.py
--costs-only`) read the whole LLM response cache, which is not released.

**The paper's numbers, tables and figures.** Two scripts read only the result files above, pinned by name in
`config/paper_assets.yaml` (they stop if a newer version of a pinned file exists, as after a rerun: pin it
first). `98_results_and_quotes.py` writes `results/RESULTS.md`, every number the paper quotes with the
file, selector, script and run manifest it comes from (`config/paper_numbers.yaml`), and `results/QUOTES.md`,
the Portuguese quotations with their English translations. `99_build_paper_assets.py --out-dir <folder>`
writes the paper's tables (CSV) and figures (PDF, PNG) with an index of their sources (`tables/ASSETS.csv`).
The B2 table and numbers need the questionnaire's workbook, which is not released; both scripts skip them
and say so.

Steps that compute without calling a model but take long are reproducible the same way: the A2 projections
(`50_a2_projections.py`, `53_a2_candidate_fits.py`; their fitted outputs are in the data archive) and the
local embeddings of A1 (`44_a1_local_embeddings.py`, `--setup-only` downloads the models).

**2. With model calls.** These steps call the judges, the labelling models, the embedding model or a web
service. The release carries their outputs (in the data archive), not the LLM response cache, so running
them again calls the models and costs money; model versions behind an API also change, so new outputs
may differ. Credentials go in a git-ignored `.env` (Azure OpenAI: `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_KEY`, `AZURE_OPENAI_MODEL`) and in `config/models.yaml` and `config/paths.yaml` (your Google
Cloud project, `JM_EVAL_GCP_PROJECT`, and an optional service-account key).

- A1: `41_infraction_category.py` (category labels), `43_a1_cloud_embeddings.py`, `55_a1_chunked_embeddings.py`
  (embeddings), `45_compression_stability.py`, `49_label_review.py` (second-model review of the labels).
- A3 and A5: `61_a3_questions.py` (reference points), `64_a3_faithfulness.py` (claims, evidence, verifiers;
  the NLI model is local), `65_a3_completeness.py`, `69_a3_judge_controls.py`, `72_a5_settings.py`.
- A4: `80_a4_notices.py`, `81_a4_notice_rewrite.py`, `91_a4_dev_notices.py` (test material),
  `85_a4_statutes.py` (statute texts from the web), `86_a4_judging_dev.py`, `87_a4_judging.py`,
  `89_a4_fact_kinds.py` (judges).
- `17_judges_smoke.py` (the judges' smoke test).

**3. With the system or private data (not runnable from this release).**

- The system's outputs: chatbot answers (`62_a3_chat_runs.py`), defense drafts (`83_a4_drafts.py`) and the
  latency runs (`93_a6_latency_cost.py`) were produced by a local server of version 2 at its frozen tag
  (`jm_eval/server.py`, `config/v2_server.env`); `12`–`16` and `20`–`21` built, froze, smoke-tested and
  documented that version from its repositories. Their outputs are released.
- Production data access: `00_export_corpus.py`, `10_freeze_v2_corpus.py`, `11_match_v1_v2.py` (the frozen
  corpus they produced is released).
- Private data: the SUSEP gold standard (`47_a1_susep.py`, local models only; aggregate results in
  `results/a1_retrieval_susep.csv` and `a1_susep_reliability.json`) and the expert questionnaire (`30_b2_compare.py`,
  `31_b3_coding_sheet.py`, `35_b3_results.py`; results in `b2_reproduction_check.json`,
  `b3_prevalence_matrix_2.csv`, `b3_quotes.csv`).
- `52_a2_served_map_reproduction.py` refits the served map with the system's environments; the reference
  pipeline does the same with pinned public libraries (below).
- Manual checks: `42_labels_audit.py`, `61_a3_questions.py`, `71_a3_claim_audit.py` (without `--apply`),
  `80_a4_notices.py`, `81_a4_notice_rewrite.py` and `89_a4_fact_kinds.py` write sheets for the authors'
  checks; the marked sheets are in `results/` (`*.xlsx`) and the scripts that read them are listed above.

## Reference pipeline

`reference/jurismap_reference.py` re-implements version 2's retrieval and map: the embedding call
(`gemini-embedding-001`, 3,072 dimensions, `SEMANTIC_SIMILARITY`), the fusion of the full report's and
the compressed report's embeddings into the stored retrieval vector, top-k cosine retrieval, and the
global UMAP (cosine, n_neighbors 10, min_dist 0, 2,000 epochs, seed 42, rows in `acordao_hash` order).
The extractive compression is not re-run (its prompt belongs to the system); the corpus holds its output.

```bash
uv venv .venv-reference --python 3.12
uv pip install --python .venv-reference/bin/python -r reference/requirements.txt
.venv-reference/bin/python reference/check_reference.py            # offline checks, map refit included (about 2 min)
.venv-reference/bin/python reference/check_reference.py --online 5 # also re-embeds 5 decisions (Vertex AI or GOOGLE_API_KEY)
.venv-reference/bin/python reference/jurismap_reference.py neighbours --hash <acordao_hash> --k 10
```

The pinned library versions are those that fitted the served map. With them, the refit reproduces the
served coordinates exactly; with other versions of umap-learn, numba or numpy, or another row order, the
map differs (`results/a2_served_map_reproduction_2.json`).

## Verification

Before release, the two trees were checked:

- the test suite, and `scripts/verify_data.py` against the data archive;
- the scripts of group 1 rerun from a fresh copy of the release, with the network blocked and no LLM
  response cache, each with the arguments of its released manifest, and every output compared with the
  released file (tables cell by cell, tolerance 1e-9);
- `reference/check_reference.py` in a fresh environment from `reference/requirements.txt`, with five
  decisions re-embedded;
- secret scanning (gitleaks) and a scan for file names and content that must not be released.

Outcome (v1.1.0, 10 Nov 2026): the 278 tests passed; all files of the data archive matched `data/MANIFEST.csv`;
the 17 scripts of group 1 were rerun once for each released run manifest that wrote a released file (22 runs): 21 reproduced every output exactly (tables cell by cell within 1e-9; figures written but not compared), and 1 differs for a known reason: one development report of A3
(`data/v2/a3_dev/report/`) written before the development sets were extended, which is not a result of the
paper. The reference pipeline reproduced the stored vectors exactly (maximum difference 1e-16), gave the
same top 10 for every decision from the released float32 and float64 vectors, refitted the served map
exactly (maximum coordinate difference 0) with the pinned libraries, and re-embedded five decisions with
cosine similarity 1.0 to the stored embeddings. Gitleaks found nothing in either tree.

The scripts of group 1 reread a few LLM responses (the draft translations of the A3 examples, the judges'
answers on the A4 development notices): those entries of the response cache are in the data archive
(`data/cache/llm/`), with the response and usage but not the prompt. The A1 lexical methods need the NLTK
Portuguese stop words and RSLP stemmer: `uv run python scripts/setup_nltk_data.py`. `92_a4_dev_selection.py`
uses the local NLI model (`64_a3_faithfulness.py --setup-only` downloads it).

## System versions

The paper calls them version 1 and version 2. The evaluation used these internal freezes of version 2
(each defined in `config/v2.*_freeze.json`, with the commits, model IDs, prompt hashes and data hashes):

| Freeze | Used for |
| --- | --- |
| v2.3 | The frozen corpus, retrieval vectors and map (A1, A2, and the data of every later freeze) |
| v2.6 | The chatbot (A3, A5) and the drafts compared in A4 |
| v2.7 | The drafts with fact controls (A4) and the latency and cost measurements (A6), as deployed |

All model calls of the system under test used Gemini 3.7 Flash (`global` endpoint) and `gemini-embedding-001`.
The judges were GPT-5.4 (Azure OpenAI) and Mistral Medium 3 (Vertex AI), with Gemini 3.8 Flash reported
separately as a same-family check (`config/models.yaml`). Prices and their sources are in `config/prices.yaml`.

## Notes

- Comments in the code and configs refer in places to `DECISIONS.md`, `PHASE_<n>_SUMMARY.md` or "the
  implementation plan". These are the authors' internal working records and are not released; the
  decisions that matter for the results are described in the paper and fixed in `config/`.
- Portuguese texts that the paper quotes carry an English translation (`results/*_examples*.md`,
  `results/b3_quotes.csv`); `results/QUOTES.md` collects them, and marks which translations the authors reviewed.
- Data protection: the corpus reproduces the decisions as the board published them, with the names of
  parties, lawyers and board members (`LICENSE-DATA.md`).

## Licence and citation

Code: MIT (`LICENSE`). Data and results: CC BY 4.0 (`LICENSE-DATA.md`). Please cite the paper, this
code (https://doi.org/10.5281/zenodo.24081246; `CITATION.cff`) and the data archive
(https://doi.org/10.5281/zenodo.24065101).
