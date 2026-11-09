"""Phase 3, B3: the outputs for the paper, from the final coding of the experts' open answers.

- results/b3_prevalence_matrix.csv: theme × expert (0/1) with the number of experts per theme. A
  theme is present for an expert when at least one segment of the expert's answers carries it.
- results/b3_quotes.csv (with --quotes): the quotes selected for the paper, in Portuguese with a free
  English translation. Only rows marked `use = 1` are exported, and each must have a translation
  marked as reviewed. A quote must be verbatim text of its segment; "[...]" marks an omission and
  "[sic]" may follow a slip in the original.

Experts appear only as E1-E5. The coding sheets are private (data/private/b3/); the two CSV files
hold no other text of the answers.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from jm_eval.b3 import KEY, load_codebook, prevalence_matrix, private_dir, read_sheet, theme_ids
from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "35_b3_results"
OMISSION = re.compile(r"\s*(?:\[\.\.\.\]|\[…\]|\(\.\.\.\))\s*")
SIC = re.compile(r"\s*\[sic\]", re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def is_verbatim(quote: str, segment: str) -> bool:
    """Every part of the quote (split at omission marks) occurs in the segment, in order; "[sic]" is ignored."""
    pos, seg = 0, _norm(segment)
    for part in OMISSION.split(_norm(SIC.sub("", str(quote)))):
        part = part.strip().strip("\"“”«»")
        if not part:
            continue
        found = seg.find(part, pos)
        if found == -1:
            return False
        pos = found + len(part)
    return True


def quotes_table(selection: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    chosen = selection[selection["use"] == 1]
    problems, rows = [], []
    for _, r in chosen.iterrows():
        sid = r[KEY]
        if sid not in final.index:
            problems.append(f"{sid}: not in the final coding")
            continue
        if pd.isna(r["quote_en"]) or not str(r["quote_en"]).strip():
            problems.append(f"{sid}: no English translation")
        if r["translation_reviewed"] != 1:
            problems.append(f"{sid}: translation not marked as reviewed")
        if pd.isna(r["quote_pt"]) or not is_verbatim(r["quote_pt"], final.at[sid, "segment"]):
            problems.append(f"{sid}: quote_pt is not verbatim text of the segment")
        rows.append({"expert": final.at[sid, "expert"], "themes": r["themes"], "theme_labels": r["theme_labels"],
                     "quote_pt": _norm(r["quote_pt"]), "quote_en": _norm(r["quote_en"])})
    if problems:
        raise SystemExit("quotes not exported:\n  " + "\n  ".join(problems))
    out = pd.DataFrame(rows, columns=["expert", "themes", "theme_labels", "quote_pt", "quote_en"])
    out.insert(0, "quote_id", [f"B3-Q{i:02d}" for i in range(1, len(out) + 1)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", help="final coding sheet (default: data/private/b3/b3_coding_final.xlsx)")
    ap.add_argument("--quotes", help="quote selection sheet (e.g. data/private/b3/b3_quotes_selection.xlsx)")
    ap.add_argument("--results-dir", help="default: results/")
    args = ap.parse_args()

    codebook = load_codebook()
    final_path = Path(args.final) if args.final else private_dir() / "b3_coding_final.xlsx"
    results = Path(args.results_dir) if args.results_dir else resolve(load_config("paths")["results"])
    with RunManifest(SCRIPT, results, params={"codebook_version": codebook.get("version"),
                                              "codebook_status": codebook.get("status")},
                     record_outputs=args.results_dir is None) as run:
        run.add_input(final_path)
        final = read_sheet(final_path, theme_ids(codebook))
        matrix = prevalence_matrix(final, codebook)
        m_path = new_artefact_path(results / "b3_prevalence_matrix.csv")
        matrix.to_csv(m_path, index=False)
        run.add_output(m_path, "B3 theme x expert matrix (final coding)")
        summary = {"segments": len(final), "themes_present": int((matrix["n_experts"] > 0).sum())}
        print(matrix.drop(columns=["label"]).to_string(index=False))
        if args.quotes:
            run.add_input(args.quotes)
            quotes = quotes_table(pd.read_excel(args.quotes, sheet_name="quotes"), final)
            q_path = new_artefact_path(results / "b3_quotes.csv")
            quotes.to_csv(q_path, index=False)
            run.add_output(q_path, "B3 quotes for the paper (PT + EN)")
            summary["quotes"] = len(quotes)
            print(f"{len(quotes)} quotes -> {q_path.name}")
        run.data["summary"] = summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
