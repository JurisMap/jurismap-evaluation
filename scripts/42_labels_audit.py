"""Phase 4a, A1: audit of the silver labels by an author (plan, Phase 4a, item 3).

Step 1 (no option): draws 60 random decisions outside the development sample and writes
results/labels_audit.xlsx. Each row shows the source fields next to the labels built from them:
the parsed legal basis, the penalty type, the infringed provision read from the report and vote,
and the infraction category. The author marks each label 1 (correct) or 0 (wrong), as an author.

Step 2 (--apply FILLED.xlsx): reads the filled sheet and writes results/a1_label_audit.json with
the accuracy of each label type (share correct, Wilson 95% interval).

The sheet holds public CRSNSP text only. The category columns need the output of
scripts/41_infraction_category.py. --without-category writes the sheet for the other labels alone;
once the categories exist, --carry FILLED.xlsx writes the full sheet for the same 60 decisions with
the marks already given copied in, so that only the category columns are left to fill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import labels as L
from jm_eval.stats import wilson_interval

SCRIPT = "42_labels_audit"
CELL_LIMIT = 32000
# label type -> (column the author fills, column that holds the label)
CHECKS = {
    "legal_basis": ("ok_base", "label_base"),
    "penalty_type": ("ok_penalty", "label_penalty"),
    "infringed_provision": ("ok_infringed", "label_infringed_primary"),
    "infraction_category": ("ok_category", "label_category"),
}
README = [
    "Audit of the A1 silver labels (author check, not an evaluation by an expert panel).",
    "",
    "Each row is one decision. White columns are the source text; the label_* columns are the labels built from it.",
    "Fill the yellow ok_* columns with 1 (correct) or 0 (wrong). Leave a cell blank only where its label is empty.",
    "",
    "ok_base: label_base lists every provision of acordao_base at article level, with the right act, number and year,",
    "  and nothing else. Paragraphs, incisos and items are dropped on purpose. A provision in brackets is generic or",
    "  belongs to a sanction regulation: it is parsed but not used to decide relevance.",
    "ok_penalty: label_penalty names the penalty types that acordao_penalidade states, and no other.",
    "ok_infringed: label_infringed_primary is a provision the party was accused of infringing, according to the",
    "  report (it need not be the only one).",
    "ok_category: label_category is the right category of the taxonomy tab for the main infraction.",
    "  If it is wrong, write the right category id in correct_category.",
    "notes: anything worth recording (optional).",
]


def sample_ids(labels: pd.DataFrame, n: int, seed: int, excluded: set[str] = frozenset()) -> list[str]:
    """`n` decisions outside the development sample and `excluded`, drawn from the hashes in sorted order."""
    pool = sorted(set(labels.loc[~labels["in_development_sample"], "acordao_hash"]) - set(excluded))
    picks = np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(picks)]


def _marked(ids: list[str], used: set[str]) -> str:
    return "; ".join(i if i in used else f"[{i}]" for i in ids)


def audit_table(labels: pd.DataFrame, corpus: pd.DataFrame, categories: pd.DataFrame | None,
                taxonomy: dict | None, ids: list[str]) -> pd.DataFrame:
    lab, cor = labels.set_index("acordao_hash"), corpus.set_index("acordao_hash")
    names = {c["id"]: c["label"] for c in (taxonomy or {}).get("categories", [])}
    cat = categories.set_index("acordao_hash") if categories is not None else None
    text = lambda v: "" if not isinstance(v, str) else v.strip()[:CELL_LIMIT]  # noqa: E731  (missing -> "")
    rows = []
    for n, h in enumerate(ids, 1):
        row = {
            "audit_id": f"A{n:02d}", "acordao_hash": h, "acordao_titulo": text(cor.at[h, "acordao_titulo"]),
            "acordao_url": text(cor.at[h, "acordao_url"]), "acordao_ementa": text(cor.at[h, "acordao_ementa"]),
            "acordao_base": text(cor.at[h, "acordao_base"]),
            "label_base": _marked(list(lab.at[h, "base_provisions"]), set(lab.at[h, "base_specific"])),
            "ok_base": None,
            "acordao_penalidade": text(cor.at[h, "acordao_penalidade"]),
            "label_penalty": " + ".join(lab.at[h, "penalty_types"]), "ok_penalty": None,
            "label_infringed_primary": text(lab.at[h, "text_primary"]),
            "label_infringed_all": "; ".join(lab.at[h, "text_infringed"]), "ok_infringed": None,
        }
        if cat is not None:
            primary = text(cat.at[h, "primary"]) if h in cat.index else ""
            row |= {"label_category": f"{primary}: {names.get(primary, '')}" if primary else "",
                    "label_category_secondary": "; ".join(cat.at[h, "secondary"]) if h in cat.index else "",
                    "category_evidence": text(cat.at[h, "evidence"]) if h in cat.index else "",
                    "ok_category": None, "correct_category": None}
        row |= {"notes": None, "acordao_relatorio": text(cor.at[h, "acordao_relatorio"])}
        rows.append(row)
    return pd.DataFrame(rows)


def carry_marks(table: pd.DataFrame, filled: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Copy the marks of an earlier sheet into `table` where the decision and the label are unchanged."""
    table, old, copied = table.copy(), filled.set_index("acordao_hash"), 0
    same = lambda a, b: (pd.isna(a) and pd.isna(b)) or str(a).strip() == str(b).strip()  # noqa: E731
    for ok_col, label_col in CHECKS.values():
        if ok_col not in table.columns or ok_col not in old.columns:
            continue
        table[ok_col] = table[ok_col].astype(object)
        for i, h in table["acordao_hash"].items():
            if h in old.index and pd.notna(old.at[h, ok_col]) and same(old.at[h, label_col], table.at[i, label_col] or None):
                table.at[i, ok_col] = old.at[h, ok_col]
                copied += 1
    if "notes" in old.columns:
        table["notes"] = table["acordao_hash"].map(old["notes"]).astype(object).where(lambda s: s.notna(), None)
    return table, copied


def write_workbook(path: Path, table: pd.DataFrame, taxonomy: dict | None) -> None:
    widths = {"acordao_ementa": 60, "acordao_base": 50, "label_base": 45, "acordao_penalidade": 40,
              "label_infringed_all": 40, "category_evidence": 45, "acordao_relatorio": 110, "notes": 30,
              "label_category": 34, "acordao_url": 30, "acordao_titulo": 30, "label_infringed_primary": 28}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="audit", index=False)
        if taxonomy:
            pd.DataFrame([{"id": c["id"], "label": c["label"], "definition": c["definition"]}
                          for c in taxonomy["categories"]]).to_excel(writer, sheet_name="taxonomy", index=False)
            ws = writer.sheets["taxonomy"]
            ws.column_dimensions["B"].width, ws.column_dimensions["C"].width = 45, 140
        pd.DataFrame({"README": README}).to_excel(writer, sheet_name="README", index=False)
        writer.sheets["README"].column_dimensions["A"].width = 130
        ws = writer.sheets["audit"]
        fill, label_fill = PatternFill("solid", fgColor="FFF2CC"), PatternFill("solid", fgColor="E2EFDA")
        for j, name in enumerate(table.columns, 1):
            letter = get_column_letter(j)
            ws.column_dimensions[letter].width = widths.get(name, 16)
            ws.cell(row=1, column=j).font = Font(bold=True)
            colour = fill if name.startswith("ok_") or name == "correct_category" else (
                label_fill if name.startswith("label_") else None)
            for i in range(1, len(table) + 2):
                cell = ws.cell(row=i, column=j)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if colour:
                    cell.fill = colour
        ws.freeze_panes = "C2"


def accuracy(filled: pd.DataFrame) -> dict:
    """Share of labels marked correct per label type; rows with an empty label are not judged."""
    out, problems = {}, []
    for name, (ok_col, label_col) in CHECKS.items():
        if ok_col not in filled.columns:
            continue
        has_label = filled[label_col].fillna("").astype(str).str.strip().ne("")
        marks = pd.to_numeric(filled[ok_col], errors="coerce")
        bad = filled.loc[filled[ok_col].notna() & ~marks.isin([0, 1]), "audit_id"].tolist()
        missing = filled.loc[has_label & filled[ok_col].isna(), "audit_id"].tolist()
        if bad:
            problems.append(f"{ok_col}: values other than 0 or 1 in {bad}")
        if missing:
            problems.append(f"{ok_col}: not filled in {missing}")
        judged = marks[has_label & marks.isin([0, 1])]
        n, correct = int(judged.size), int(judged.sum())
        low, high = wilson_interval(correct, n)
        out[name] = {"audited": int(len(filled)), "with_label": int(has_label.sum()), "judged": n, "correct": correct,
                     "accuracy": round(correct / n, 4) if n else None,
                     "wilson_95": [round(low, 4), round(high, 4)] if n else None}
    if problems:
        raise SystemExit("audit sheet not usable:\n  " + "\n  ".join(problems))
    if "correct_category" in filled.columns:
        fixes = filled.loc[filled["correct_category"].notna(), ["label_category", "correct_category"]]
        out["infraction_category"]["corrections"] = [
            {"labelled": str(a).split(":")[0], "correct": str(b).strip()} for a, b in fixes.itertuples(index=False)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="default: the newest data/labels/a1_silver_labels*.parquet")
    ap.add_argument("--category", help="default: the newest data/labels/a1_infraction_category*.parquet")
    ap.add_argument("--without-category", action="store_true", help="audit the other labels alone")
    ap.add_argument("--carry", metavar="FILLED", help="earlier audit sheet whose marks are copied into the new one")
    ap.add_argument("--apply", metavar="FILLED", help="filled audit sheet: compute the accuracy of each label type")
    ap.add_argument("--results-dir", help="default: results/")
    args = ap.parse_args()

    cfg, paths = L.settings(), load_config("paths")
    results = Path(args.results_dir) if args.results_dir else resolve(paths["results"])
    labels_dir = resolve(paths["data"]["labels"])
    audit_params = {"n": cfg["audit"]["n"], "seed": cfg["audit"]["seed"],
                    "excluded": len(cfg["audit"].get("excluded_ids") or []) + len(cfg["development"]["ids"])}
    with RunManifest(SCRIPT, results, params={"audit": audit_params, "apply": bool(args.apply)},
                     record_outputs=args.results_dir is None) as run:
        if args.apply:
            run.add_input(args.apply)
            result = accuracy(pd.read_excel(args.apply, sheet_name="audit"))
            out = new_artefact_path(results / "a1_label_audit.json")
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            run.add_output(out, "A1 silver labels: accuracy in the author's audit")
            run.data["summary"] = result
            for name, r in result.items():
                print(f"{name:22s} {r['correct']}/{r['judged']} correct ({r['accuracy']}), 95% {r['wilson_95']}")
            return 0

        labels_path = Path(args.labels) if args.labels else latest_artefact_path(labels_dir / "a1_silver_labels.parquet")
        category_path = (Path(args.category) if args.category
                         else latest_artefact_path(labels_dir / "a1_infraction_category.parquet"))
        categories = taxonomy = None
        if not args.without_category:
            if not category_path.exists():
                raise SystemExit(f"{category_path} not found: run scripts/41_infraction_category.py first, "
                                 "or pass --without-category.")
            categories, taxonomy = pd.read_parquet(category_path), load_config("a1_taxonomy")
            run.add_input(category_path)
        run.add_input(labels_path)
        labels = pd.read_parquet(labels_path)
        ids = sample_ids(labels, cfg["audit"]["n"], cfg["audit"]["seed"], set(cfg["audit"].get("excluded_ids") or []))
        corpus = pd.read_parquet(resolve(cfg["corpus"]), columns=[
            "acordao_hash", "acordao_titulo", "acordao_url", "acordao_ementa", "acordao_base",
            "acordao_penalidade", "acordao_relatorio"])
        table = audit_table(labels, corpus[corpus["acordao_hash"].isin(ids)], categories, taxonomy, ids)
        copied = None
        if args.carry:
            run.add_input(args.carry)
            table, copied = carry_marks(table, pd.read_excel(args.carry, sheet_name="audit"))
        out = new_artefact_path(results / "labels_audit.xlsx")
        write_workbook(out, table, taxonomy)
        run.add_output(out, "A1 silver labels: audit sheet for the author (60 decisions)")
        run.data["summary"] = {"decisions": len(table), "with_category": categories is not None,
                               "marks_carried": copied,
                               "in_comparison_set": int(labels.set_index("acordao_hash").loc[ids, "in_comparison_set"].sum())}
        print(f"{len(table)} decisions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
