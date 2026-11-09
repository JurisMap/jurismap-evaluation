"""Phase 3, B3: the coding sheet for the experts' open answers.

Reads the questionnaire (a private file: the answers identify the five experts, so it stays in
data/private/), splits each open answer into segments (sentences by default), and writes one row per
(expert, open question, segment) with one 0/1 column per theme of config/b3_codebook.yaml. The coder
enters 1 where a theme applies; a blank cell counts as 0.

The sheet contains the answers' text, so it is written to data/private/b3/ (git-ignored and blocked
by the pre-commit hook). Only the theme × expert matrix and selected quotes go to results/
(scripts/35_b3_results.py).

    uv run python scripts/31_b3_coding_sheet.py --questionnaire "data/private/Avaliação do JurisMap_5resp_20250920.xlsx"

Output: data/private/b3/b3_coding_sheet_coder1.xlsx (never overwritten), manifest there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jm_eval.b3 import (CODEBOOK, FIXED_COLS, build_rows, load_codebook, open_question_columns,  # noqa: F401
                        private_dir, questionnaire_rows, segments, write_sheet)
from jm_eval.io import new_artefact_path, sha256_file
from jm_eval.manifest import RunManifest

SCRIPT = "31_b3_coding_sheet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questionnaire", required=True, help="questionnaire .xlsx (keep it in data/private/)")
    ap.add_argument("--out-dir", default=None, help="default: data/private/b3")
    ap.add_argument("--coders", nargs="+", default=["coder1"])
    args = ap.parse_args()

    codebook = load_codebook()
    cfg = codebook["questionnaire"]
    q_path = Path(args.questionnaire)
    out_dir = Path(args.out_dir) if args.out_dir else private_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if codebook.get("status") != "approved":
        print(f"NOTE: codebook status is {codebook.get('status')!r}; approve it before coding starts.")

    with RunManifest(SCRIPT, out_dir, params={"codebook_version": codebook.get("version"),
                                              "codebook_status": codebook.get("status"),
                                              "segment": cfg.get("segment")},
                     record_outputs=False) as run:
        run.add_input(q_path)
        run.add_input(CODEBOOK)
        rows, questions, how = questionnaire_rows(q_path, codebook)
        for coder in args.coders:
            path = new_artefact_path(out_dir / f"b3_coding_sheet_{coder}.xlsx")
            write_sheet(path, rows, codebook, coder, questions, how)
            run.data["outputs"].append({"path": str(path), "sha256": sha256_file(path)})
            print(f"{path}: {len(rows)} segments × {len(codebook['themes'])} themes")
        run.data["summary"] = {"segments": len(rows), "questions": len(questions), "question_source": how,
                               "segments_per_expert": pd.Series([r["expert"] for r in rows]).value_counts().to_dict()}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
