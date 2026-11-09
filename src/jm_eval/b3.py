"""B3: coding of the experts' open answers. Shared helpers for scripts/31-35.

The codebook is config/b3_codebook.yaml. A coding sheet has one row per (expert, open question,
segment) and one 0/1 column per theme; a blank theme cell counts as 0 and an "x" as 1. Sheets hold the experts'
answers, so they live in data/private/b3/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from jm_eval.io import CONFIG_DIR, load_config, resolve

CODEBOOK = CONFIG_DIR / "b3_codebook.yaml"
KEY = "segment_id"
FIXED_COLS = [KEY, "expert", "question_no", "question", "segment_no", "segment"]
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀ-Ý0-9\"'(«-])|\n+")


def private_dir() -> Path:
    return resolve(load_config("paths")["data"]["private"]) / "b3"


def load_codebook(path: Path = CODEBOOK) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def theme_ids(codebook: dict) -> list[str]:
    return [t["id"] for t in codebook["themes"]]


def theme_labels(codebook: dict) -> dict[str, str]:
    return {t["id"]: t["label"] for t in codebook["themes"]}


# ---------------------------------------------------------------- questionnaire -> segments
def segments(text: str, mode: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", str(text)).strip()
    if not text:
        return []
    if mode == "answer":
        return [text]
    return [s.strip() for s in SENTENCE_END.split(text) if s and s.strip()]


def open_question_columns(df: pd.DataFrame, cfg: dict) -> tuple[list[str], str]:
    """(columns, how they were chosen)."""
    wanted = cfg.get("open_question_columns") or []
    if wanted:
        cols = []
        for w in wanted:
            hits = [c for c in df.columns if str(w).lower() in str(c).lower()]
            if len(hits) != 1:
                raise SystemExit(f"open_question_columns entry {w!r} matches {len(hits)} columns: {hits}")
            cols.append(hits[0])
        return cols, "config"
    excluded = [e.lower() for e in cfg.get("exclude_columns", [])]
    cols = []
    for c in df.columns:
        if any(e in str(c).lower() for e in excluded):
            continue
        values = df[c].dropna()
        if values.empty or not all(isinstance(v, str) for v in values):
            continue
        if values.str.len().median() >= cfg.get("min_median_chars", 20):
            cols.append(c)
    if not cols:
        raise SystemExit("No open-question columns detected; list them in config/b3_codebook.yaml")
    return cols, "detected"


def build_rows(df: pd.DataFrame, cfg: dict) -> tuple[list[dict], list[str], str]:
    ids = cfg["respondent_ids"]
    if len(df) != len(ids):
        raise SystemExit(f"questionnaire has {len(df)} rows but respondent_ids lists {len(ids)}")
    cols, how = open_question_columns(df, cfg)
    rows = []
    for q_no, col in enumerate(cols, 1):
        for expert, answer in zip(ids, df[col]):
            if pd.isna(answer):
                continue
            for s_no, seg in enumerate(segments(answer, cfg.get("segment", "sentence")), 1):
                rows.append({KEY: f"{expert}-Q{q_no:02d}-S{s_no:02d}", "expert": expert,
                             "question_no": q_no, "question": str(col), "segment_no": s_no, "segment": seg})
    return rows, [str(c) for c in cols], how


def questionnaire_rows(path: Path, codebook: dict) -> tuple[list[dict], list[str], str]:
    cfg = codebook["questionnaire"]
    return build_rows(pd.read_excel(path, sheet_name=cfg.get("sheet", 0)), cfg)


# ---------------------------------------------------------------- coding sheets
def write_sheet(path: Path, rows: list[dict], codebook: dict, coder: str, questions: list[str], how: str,
                codes: dict[str, dict[str, int]] | None = None, readme: list[str] | None = None) -> None:
    """Write a coding sheet. `codes` ({segment_id: {theme: 0/1}}) pre-fills the theme cells with 1s."""
    themes = codebook["themes"]
    wb = Workbook()
    ws = wb.active
    ws.title = "coding"
    ws.append(FIXED_COLS + [t["id"] for t in themes] + ["notes"])
    for r in rows:
        filled = [(1 if (codes or {}).get(r[KEY], {}).get(t["id"]) else None) for t in themes]
        ws.append([r[c] for c in FIXED_COLS] + filled + [None])
    bold, grey = Font(bold=True), PatternFill("solid", fgColor="EEEEEE")
    for cell in ws[1]:
        cell.font = bold
    first_theme_col, last_theme_col = len(FIXED_COLS) + 1, len(FIXED_COLS) + len(themes)
    for row in ws.iter_rows(min_row=2, max_col=len(FIXED_COLS)):
        for cell in row:
            cell.fill = grey
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for col, w in {"A": 16, "B": 7, "C": 6, "D": 40, "E": 6, "F": 80}.items():
        ws.column_dimensions[col].width = w
    if rows:
        dv = DataValidation(type="whole", operator="between", formula1="0", formula2="1", allow_blank=True,
                            error="Enter 1 (theme present) or leave blank / 0 (absent)")
        start = ws.cell(row=2, column=first_theme_col).coordinate
        end = ws.cell(row=len(rows) + 1, column=last_theme_col).coordinate
        dv.add(f"{start}:{end}")
        ws.add_data_validation(dv)
    ws.freeze_panes = ws.cell(row=2, column=first_theme_col)

    cb = wb.create_sheet("codebook")
    cb.append(["id", "theme", "definition"])
    for t in themes:
        cb.append([t["id"], t["label"], t.get("definition", "")])
    cb.column_dimensions["B"].width = 45
    cb.column_dimensions["C"].width = 90

    info = wb.create_sheet("README")
    for line in readme or [
        f"B3 coding sheet, {coder}.",
        "For every segment, enter 1 in each theme column that applies; leave blank (= 0) otherwise.",
        "A segment may carry several themes or none. Use 'notes' for doubts; do not edit grey columns.",
        f"Codebook version {codebook.get('version')} (status: {codebook.get('status')}).",
        f"Open questions ({how}):",
        *[f"  Q{i:02d}: {q}" for i, q in enumerate(questions, 1)],
        "Private file: it contains the experts' answers. Keep it in data/private/b3/.",
    ]:
        info.append([line])
    info.column_dimensions["A"].width = 120
    wb.save(path)


def _code(value) -> int:
    """A theme cell as 0/1: blank or 0 = absent; 1 or an "x" mark = present; anything else is invalid (-1)."""
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return 0
    if isinstance(value, str):
        return 1 if value.strip().lower() == "x" else -1
    return int(value) if value in (0, 1) else -1


def read_sheet(path: Path, themes: list[str]) -> pd.DataFrame:
    """A coding sheet indexed by segment id, theme cells as 0/1 integers (blank = 0; "x" = 1)."""
    df = pd.read_excel(path, sheet_name="coding")
    missing = [c for c in [KEY, "expert", *themes] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if df[KEY].duplicated().any():
        raise SystemExit(f"{path}: duplicate segment ids")
    codes = df[themes].map(_code)
    bad = [(r, c, df.at[r, c]) for c in themes for r, v in codes[c].items() if v == -1]
    if bad:
        raise SystemExit(f"{path}: values other than 0, 1, x or blank, e.g. " +
                         ", ".join(f"{df.at[r, KEY]}/{c}={v!r}" for r, c, v in bad[:5]))
    df[themes] = codes.astype(int)
    return df.set_index(KEY)


def expert_level(df: pd.DataFrame, themes: list[str]) -> pd.DataFrame:
    """Theme × expert presence (1 if any of the expert's segments is coded 1)."""
    return df.groupby("expert")[themes].max().T


def prevalence_matrix(df: pd.DataFrame, codebook: dict) -> pd.DataFrame:
    """Theme × expert 0/1 matrix with labels and the number of experts per theme."""
    themes, labels, experts = theme_ids(codebook), theme_labels(codebook), codebook["questionnaire"]["respondent_ids"]
    m = expert_level(df, themes).reindex(columns=experts, fill_value=0)
    out = pd.DataFrame({"theme": themes, "label": [labels[t] for t in themes]})
    for e in experts:
        out[e] = [int(m.at[t, e]) for t in themes]
    out["n_experts"] = out[experts].sum(axis=1)
    return out


# ---------------------------------------------------------------- agreement statistics
def cohen_kappa(a, b) -> float:
    """Cohen's κ for two binary ratings; NaN when chance agreement is 1 (both constant and equal)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if np.isclose(pe, 1.0) else (po - pe) / (1 - pe)


def fleiss_kappa(ratings) -> float:
    """Fleiss' κ for binary ratings, shape (items, raters); NaN when every rating is the same."""
    r = np.asarray(ratings, int)
    n_items, n_raters = r.shape
    ones = r.sum(axis=1)
    counts = np.stack([n_raters - ones, ones], axis=1)
    p_item = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_cat = counts.sum(axis=0) / (n_items * n_raters)
    pe = float((p_cat ** 2).sum())
    return float("nan") if np.isclose(pe, 1.0) else float((p_item.mean() - pe) / (1 - pe))
