# data/

The data are not in the git repository. They are in the data archive `jurismap-evaluation-data.zip`
on Zenodo: https://doi.org/10.5281/zenodo.24065101. Extract it at the repository root (`unzip jurismap-evaluation-data.zip`):
its files land under `data/`, at the paths the scripts read. Then check them:

```bash
python scripts/verify_data.py        # sha256 and size of every file against data/MANIFEST.csv
```

`MANIFEST.csv` lists every file of the archive with its sha256 and size.

| Path | Contents |
| --- | --- |
| `v2/corpus_v2_rebuilt_20261009.parquet` | The frozen corpus of version 2: 2,390 CRSNSP decisions (sessions of Jan 2017 to Jun 2026) with the extracted fields, the rapporteur reports, the compressed reports, the embeddings (full report, compressed report, stored retrieval vector) and the map coordinates |
| `v2/shared_ids.txt`, `v2/v1_v2_id_map.csv` | The 1,503 decisions present in both versions' corpora, and how they were matched |
| `v2/rebuild_20261009/` | The stored retrieval vectors, both embeddings and the served map coordinates as arrays, with their row order (`hashes_v2.txt`); read by `reference/` |
| `labels/` | A1 silver labels (legal basis, infringed provisions, penalty, outcome), infraction categories (Gemini 3.7 Flash, and blind by Mistral Medium 3), the label review |
| `v2/a1_embeddings/`, `v2/a1_runs/` | A1: vectors of the retrieval methods (cloud and local models) and per-query metrics |
| `v2/a2_projections/`, `v2/a2_candidates/`, `v2/a2_runs/` | A2: the fitted projections (UMAP grid, t-SNE, PaCMAP, PCA; five seeds) and the map candidates |
| `v2/a3_questions.jsonl`, `v2/a3/`, `v2/a3_dev/` | A3: the frozen question set, case sets, the chatbot's answers, deterministic scores, claims with their evidence and verifier labels, judge controls; development sets and the chat strategies compared on them |
| `v2/a5/` | A5: the same at context sizes 5, 10, 20 and 30 decisions and in hierarchical mode |
| `v2/a4/`, `v2/a4_v27/`, `v2/a4_dev/` | A4: reconstructed notices, precedents supplied, every draft (with the server's record of its model calls and agent outputs), judgements, statute texts, fact sorting; for the two system versions compared and the development notices |
| `v2/a6/` | A6: one row per timed call, and the responses |
| `v2/corpus_v2_rebuilt_20260927.parquet` | The version 2.2 corpus file: the same decisions and texts, rows in another order; the A1 silver labels and the Phase 4.1 candidate selection were computed on it |
| `v2/compression_stability_20261007/` | A1: 50 reports compressed again with the frozen pipeline |
| `v2/reextract_20260927/sources/<acordao_hash>.txt` | The raw text of each of the 2,390 decisions, extracted from the PDF the board published (linked by `acordao_url` in the corpus). The corpus sections were extracted from these texts; `scripts/84` reads the ruling heading ("ACÓRDÃO CRSNSP n/yyyy") from them |
| `cache/llm/` | The LLM responses that the report scripts read again (no prompt text: model, prompt hash, parameters, response, usage) |

Each directory also holds the run manifests (`*.manifest.json`) of the scripts that wrote its files:
git commits, model IDs and parameters, config hashes, seeds, timestamps and package versions. Paths in
them are relative to the repository; `../app/...`, `../scrapers` and `../frontend...` are the system's
repositories, which are not public.

Not included: server logs (they hold the system's prompts), the rest of the LLM response cache, the
decisions' PDFs (linked by `acordao_url` in the corpus), SUSEP data and the expert questionnaire answers.
