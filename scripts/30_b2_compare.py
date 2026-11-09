"""Phase 3, B2: check that the local rerun of the B2 statistics reproduces the cloud version.

The B2 analysis itself is `analysis/b2/b2_expert_statistics.py`, copied unchanged from
`ISWA/B2/` in the paper folder (see analysis/b2/README.md). This script compares the workbook
produced by the local rerun with the reference `ISWA/B2/b2_statistics.xlsx`, sheet by sheet and
cell by cell:

- every sheet of the reference must exist in the rerun, with the same used range;
- numbers must agree within --rtol / --atol (default 1e-9 / 1e-12: "identical" up to float noise);
- text must be equal after stripping; empty cells must stay empty.

Writes results/b2_reproduction_check.json (counts and the first mismatches) and exits 1 on any
mismatch. The comparison holds only aggregates; no questionnaire answer is copied.
"""

from __future__ import annotations

import argparse
import json
import math
from numbers import Number
from pathlib import Path

from openpyxl import load_workbook

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "30_b2_compare"
MAX_REPORTED = 50


def _norm(v):
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


def same(a, b, rtol: float, atol: float) -> bool:
    a, b = _norm(a), _norm(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, Number) and isinstance(b, Number):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b


def compare(reference: Path, rerun: Path, rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    ref = load_workbook(reference, data_only=True)
    new = load_workbook(rerun, data_only=True)
    report = {"reference": str(reference), "rerun": str(rerun), "rtol": rtol, "atol": atol,
              "sheets": {}, "missing_sheets": [], "extra_sheets": sorted(set(new.sheetnames) - set(ref.sheetnames)),
              "mismatches": [], "cells_compared": 0}
    for name in ref.sheetnames:
        if name not in new.sheetnames:
            report["missing_sheets"].append(name)
            continue
        ws_ref, ws_new = ref[name], new[name]
        rows = max(ws_ref.max_row, ws_new.max_row)
        cols = max(ws_ref.max_column, ws_new.max_column)
        n_bad = 0
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                a, b = ws_ref.cell(r, c).value, ws_new.cell(r, c).value
                report["cells_compared"] += 1
                if not same(a, b, rtol, atol):
                    n_bad += 1
                    if len(report["mismatches"]) < MAX_REPORTED:
                        report["mismatches"].append({"sheet": name, "cell": ws_ref.cell(r, c).coordinate,
                                                     "reference": a if not isinstance(a, float) else repr(a),
                                                     "rerun": b if not isinstance(b, float) else repr(b)})
        report["sheets"][name] = {"range": f"{rows}x{cols}", "mismatches": n_bad,
                                  "same_shape": (ws_ref.max_row, ws_ref.max_column) == (ws_new.max_row, ws_new.max_column)}
    report["identical"] = (not report["missing_sheets"]
                           and all(s["mismatches"] == 0 and s["same_shape"] for s in report["sheets"].values()))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="ISWA/B2/b2_statistics.xlsx (the cloud run)")
    ap.add_argument("--rerun", required=True, help="the workbook written by the local rerun")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--results-dir", help="default: results/")
    args = ap.parse_args()

    results = Path(args.results_dir) if args.results_dir else resolve(load_config("paths")["results"])
    with RunManifest(SCRIPT, results, params={"rtol": args.rtol, "atol": args.atol},
                     record_outputs=args.results_dir is None) as run:
        run.add_input(args.reference)
        run.add_input(args.rerun)
        script = Path(__file__).resolve().parents[1] / "analysis/b2/b2_expert_statistics.py"
        if script.exists():
            run.add_input(script)
        report = compare(Path(args.reference), Path(args.rerun), args.rtol, args.atol)
        out = new_artefact_path(results / "b2_reproduction_check.json")
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        run.add_output(out, "B2 rerun vs cloud reference")
        run.data["summary"] = {"identical": report["identical"], "cells_compared": report["cells_compared"]}
        print(json.dumps({k: report[k] for k in ("identical", "cells_compared", "missing_sheets", "sheets")}, indent=2))
        for m in report["mismatches"][:10]:
            print("MISMATCH", m)
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
