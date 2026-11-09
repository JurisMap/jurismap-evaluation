import importlib.util
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


b2 = _load("b2_compare", "scripts/30_b2_compare.py")
sheet = _load("b3_sheet", "scripts/31_b3_coding_sheet.py")
hook = _load("hook", "scripts/hooks/block_private_files.py")
CODEBOOK = sheet.load_codebook()
THEMES = [t["id"] for t in CODEBOOK["themes"]]
# The codebook lists the real questionnaire's headers; the synthetic one below relies on detection
DETECT = {**CODEBOOK["questionnaire"], "open_question_columns": []}


def test_codebook_has_the_twelve_plan_themes():
    assert THEMES == [f"T{i:02d}" for i in range(1, 13)]
    assert CODEBOOK["themes"][11]["label"].startswith("Extend to other regulators")


def test_segments_split_sentences_not_lowercase_continuations():
    text = "Encontrar precedentes rapidamente. Economiza tempo; a filtragem é ótima.\nMelhorar a interface!"
    assert sheet.segments(text, "sentence") == [
        "Encontrar precedentes rapidamente.", "Economiza tempo; a filtragem é ótima.", "Melhorar a interface!"]
    assert len(sheet.segments(text, "answer")) == 1


def _questionnaire():
    return pd.DataFrame({
        "Nome": list("ABCDE"), "Nota": [5, 4, 4, 3, 5],
        "O que mais valorizou?": ["Precedentes rápidos. Poupa tempo.", "Chatbot útil para resumir casos.",
                                  "Foco no domínio do CRSNSP e não genérico.", None, "Ganho de tempo enorme nas pesquisas."],
    })


def test_open_questions_detected_and_rows_built():
    rows, questions, how = sheet.build_rows(_questionnaire(), DETECT)
    assert how == "detected" and questions == ["O que mais valorizou?"]
    assert [r["segment_id"] for r in rows][:2] == ["E1-Q01-S01", "E1-Q01-S02"]
    assert "E4" not in {r["expert"] for r in rows}  # blank answer: no segment


def test_configured_columns_must_each_match_one_header():
    assert len(CODEBOOK["questionnaire"]["open_question_columns"]) == 7
    cfg = {**DETECT, "open_question_columns": ["valorizou"]}
    assert sheet.build_rows(_questionnaire(), cfg)[2] == "config"
    with pytest.raises(SystemExit):   # the real headers are not in the synthetic questionnaire
        sheet.build_rows(_questionnaire(), CODEBOOK["questionnaire"])


def test_row_count_must_match_respondents():
    with pytest.raises(SystemExit):
        sheet.build_rows(_questionnaire().iloc[:4], DETECT)


def _wb(path, value, extra_sheet=False):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "stats"
    ws.append(["construct", "mean"])
    ws.append(["Value proposition", value])
    if extra_sheet:
        wb.create_sheet("figure_data").append([1])
    wb.save(path)
    return path


def test_b2_compare(tmp_path):
    ref = _wb(tmp_path / "ref.xlsx", 4.55, extra_sheet=True)
    assert not b2.compare(ref, _wb(tmp_path / "a.xlsx", 4.55))["identical"]            # missing sheet
    assert b2.compare(ref, _wb(tmp_path / "b.xlsx", 4.55 + 1e-13, True))["identical"]  # float noise
    bad = b2.compare(ref, _wb(tmp_path / "c.xlsx", 4.56, True))
    assert not bad["identical"] and bad["mismatches"][0]["cell"] == "B2"


def test_hook_blocks_expert_answer_files():
    for p in ["Avaliação do JurisMap_5resp_20250920.xlsx", "x/b3_coding_sheet_coder1.xlsx", "b3_disagreements_2.xlsx"]:
        assert hook.is_blocked(p), p
    assert not hook.is_blocked("results/b3_agreement.csv")
