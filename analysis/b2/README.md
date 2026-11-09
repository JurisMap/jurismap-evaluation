# B2: expert-questionnaire statistics (plan B2, Phase 3)

The B2 analysis was run in a cloud session on 26 Sep 2026 (`ISWA/B2/` in the paper folder). Phase 3 copies it here and reruns it locally, and `scripts/30_b2_compare.py` checks that the local outputs are identical to the cloud ones.

**Status (2026-10-05): reproduced.** The local rerun's workbook is identical to the cloud workbook: 5 sheets, 486 cells, 0 mismatches (`results/b2_reproduction_check.json`).

- `b2_expert_statistics.py` here is a byte-identical copy of `ISWA/B2/b2_expert_statistics.py`.
- Local packages: scipy 1.18.1, matplotlib 3.11.2, numpy 2.5.3, pandas 3.0.6 (Python 3.12).
- The rerun's figure shows the same bars, medians, means and asterisks as the cloud figure. The files are not byte-identical: text is placed a few pixels differently (font rendering). Use the cloud figure in `ISWA/B2/` for the paper.
- Respondents are E1-E5 in the export's row order, as in `scripts/31_b3_coding_sheet.py`.

## How it was run

```bash
cp "<paper folder>/ISWA/B2/b2_expert_statistics.py" analysis/b2/
cp "<paper folder>/Avaliações/Avaliação do JurisMap_5resp_20250920.xlsx" data/private/   # private, never committed
uv run python analysis/b2/b2_expert_statistics.py "data/private/Avaliação do JurisMap_5resp_20250920.xlsx" data/b2_rerun
uv run python scripts/30_b2_compare.py \
    --reference "<paper folder>/ISWA/B2/b2_statistics.xlsx" --rerun data/b2_rerun/b2_statistics.xlsx
```

The script overwrites `b2_statistics.xlsx` in its output folder, so a further rerun needs a new folder.

## Construct results (workbook, sheet `Constructs`)

Per-respondent construct scores, E1–E5, with the one-sided exact Wilcoxon p against the neutral point 3:

| Construct | Scores (E1–E5) | Mean | Median | Bootstrap 95% CI | Exact Wilcoxon p (> 3) | r_wg(j) |
| --- | --- | --- | --- | --- | --- | --- |
| Value proposition | 5.00, 4.50, 4.75, 4.25, 4.25 | 4.55 | 4.50 | 4.30–4.80 | 0.031 | 0.92 |
| Core functionality | 5.00, 4.25, 4.00, 4.00, 3.75 | 4.20 | 4.00 | 3.90–4.60 | 0.031 | 0.91 |
| Interface quality | 3.33, 5.00, 3.67, 4.00, 4.33 | 4.07 | 4.00 | 3.60–4.60 | 0.031 | 0.83 |
| Scenario relevance | 3.75, 5.00, 5.00, 3.00, 4.50 | 4.25 | 4.50 | 3.55–4.90 | 0.062 | 0.82 |
| Overall (15 items) | 4.33, 4.67, 4.40, 3.80, 4.20 | 4.28 | 4.33 | 4.01–4.52 | — | — |

With N = 5, the smallest possible exact one-sided p is 0.031 (all five above neutral).

The revision plan's B2 table gives the overall interval as 4.03–4.51. The workbook (cloud and rerun) gives 4.013–4.520; the other rows agree. The paper should take its numbers from the workbook.
