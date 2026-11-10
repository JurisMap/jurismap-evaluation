"""Phase 11: the v2.6/v2.7 comparison, the paper's assets, and the records of numbers and quotations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from jm_eval import paper as P
from jm_eval.drafting import compare as V

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


records = _load("results_and_quotes", "98_results_and_quotes.py")
assets = _load("paper_assets", "99_build_paper_assets.py")


# ------------------------------------------------------------------------------------------- formats

def test_rounding_is_half_up_on_the_written_value():
    assert P.f(6.25, 1) == "6.3"            # binary 6.25 is exact; banker's rounding would give 6.2
    assert P.f(0.0405, 3) == "0.041"        # 0.0405 is below the half in binary; the written value is not
    assert P.signed(-0.0185, 3) == "−0.019"
    assert P.f(-0.75, 2) == "−0.75"
    assert P.interval(-0.253, -0.1666, 3) == "−0.253 to −0.167"
    assert P.pval(0.00009) == "< 0.001" and P.pval(0.0414) == "0.041"
    assert P.pct(0.4849, 1) == "48.5%" and P.pct(0.691, 0) == "69%"
    assert P.usd(0.18276, 2) == "$0.18"


def test_family_base_and_stale_pins(tmp_path):
    for name in ("x_2.csv", "x_3.csv"):
        (tmp_path / name).write_text("a\n1\n")
    assert P.family_base(tmp_path / "x_2.csv") == tmp_path / "x.csv"
    s = P.Sources({"sources": {"old": str(tmp_path / "x_2.csv"), "new": str(tmp_path / "x_3.csv"),
                               "gone": str(tmp_path / "y.csv")}})
    problems = s.check()
    assert any(p.startswith("old:") and "newer" in p for p in problems)
    assert any(p.startswith("gone:") and "does not exist" in p for p in problems)
    assert not any(p.startswith("new:") for p in problems)


# --------------------------------------------------------------------------------- version comparison

def test_invented_per_draft_uses_the_audited_shares():
    confusion = pd.DataFrame({"invented": [7, 0, 0, 1], "assertion": [4, 13, 0, 5], "procedural": [0, 0, 12, 0],
                              "in_notice": [2, 0, 0, 6]}, index=V.FACT_KINDS)
    shares = V.invented_shares(confusion)
    assert shares["invented"] == pytest.approx(7 / 13) and shares["assertion"] == 0 and shares["in_notice"] == pytest.approx(1 / 12)
    est = V.invented_per_draft([{"notice_id": "N01", "mode": "full", "run": 1,
                                 "kinds": ["invented", "invented", "assertion", "in_notice"]}], shares)
    assert est["estimated_invented"].iloc[0] == pytest.approx(2 * 7 / 13 + 1 / 12)


def test_paired_by_notice_averages_per_notice_and_drops_lost_and_leaked():
    a = pd.DataFrame({"notice_id": ["N1", "N1", "N2", "N3", "N3"], "lost": [False, False, False, False, True],
                      "leaked": [False, True, False, False, False], "x": [1.0, 9.0, 2.0, 3.0, 99.0]})
    b = pd.DataFrame({"notice_id": ["N1", "N2", "N3"], "lost": [False] * 3, "leaked": [False] * 3, "x": [2.0, 4.0, 6.0]})
    out = V.paired_by_notice(V.written(a), V.written(b), {"x": "x"}).iloc[0]
    assert out["notices"] == 3 and out["mean_a"] == pytest.approx(2.0) and out["mean_b"] == pytest.approx(4.0)
    assert out["notices_higher_in_b"] == 3 and out["wilcoxon_p"] == pytest.approx(0.25)   # exact test, n = 3


# ------------------------------------------------------------------------------------------- numbers

def test_number_formatter_specs():
    F = records.FMT
    assert F.format("{0:.3f} ({1:.3f} to {2:.3f})", 0.403154, 0.382278, 0.424485) == "0.403 (0.382 to 0.424)"
    assert F.format("{0:+.3f}, p {1:p}", 0.0431, 0.0004) == "+0.043, p < 0.001"
    assert F.format("{0:%1} {1:usd} {2:d} {3:,.0f}", 0.6906, 0.0589, 23.6, 1463.0) == "69.1% $0.059 24 1,463"


def test_selectors_need_exactly_one_row_unless_aggregated(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"k": ["a", "a", "b"], "flag": [True, False, False], "v": [1.0, 2.0, 5.0], "r": [None, "x", "x"]}).to_csv(path, index=False)
    S = P.Sources({"sources": {"t": str(path)}})
    raw, shown, _ = records.evaluate({"id": "n1", "source": "t", "where": {"k": "b"}, "value": "v", "fmt": "{0:.1f}"}, S, {})
    assert shown == "5.0"
    raw, shown, _ = records.evaluate({"id": "n2", "source": "t", "where": {"r": None}, "value": "v", "fmt": "{0:.0f}"}, S, {})
    assert shown == "1"                                         # null matches an empty cell
    raw, shown, _ = records.evaluate({"id": "n3", "source": "t", "where": {"flag": False}, "agg": "sum", "value": "v", "fmt": "{0:d}"}, S, {})
    assert shown == "7"                                         # booleans match case-insensitively
    with pytest.raises(ValueError, match="2 rows"):
        records.evaluate({"id": "n4", "source": "t", "where": {"k": "a"}, "value": "v", "fmt": "{0}"}, S, {})
    raw, shown, _ = records.evaluate({"id": "n5", "expr": "v['n1'] * 2", "fmt": "{0:.0f}"}, S, {"n1": 5.0})
    assert shown == "10"


def test_every_number_of_the_config_resolves():
    """The real config against the real results (the B2 workbook is git-ignored: its section is skipped if absent)."""
    from jm_eval.io import load_config
    S = P.Sources()
    cfg = load_config("paper_numbers")
    if not S.paths["b2_workbook"].exists():
        cfg["sections"] = [s for s in cfg["sections"] if s["id"] != "b2"]
    text, n = records.build_results(S, cfg, {})
    assert n > 100 and "**" in text


# ------------------------------------------------------------------------------------------- quotes

def test_prompt_excerpts_skip_english_prompts(tmp_path):
    md = tmp_path / "p.md"
    md.write_text("### v1.1 Chat (`x.yaml`)\n- **Excerpt:** «Responda só com os casos.»\n  - EN: \"Answer only from the cases.\"\n"
                  "### v1.8 OCR (inline, EN; same text in v2)\n- **Excerpt:** «Please extract the text.»\n"
                  "### v1.4 Search\n- **Excerpt (EN, inline):** «Extract up to 3 queries»\n", encoding="utf-8")
    S = P.Sources({"sources": {"prompt_specs": str(md)}})
    q = records.quotes_prompt_specs(S)
    assert [x["quote_id"] for x in q] == ["PS-v1.1-1"] and q[0]["translation_en"] == "Answer only from the cases."
    assert q[0]["reviewed"] is False


def test_a4_block_and_inline_quotations(tmp_path):
    md = tmp_path / "a4.md"
    md.write_text("## 2. Other facts\n\n**N01, full, run 1, v2.6:**\n\n> Texto **em negrito** do rascunho.\n\n*Draft text **in bold**.*\n\n"
                  'A cause: "divergências de validação" (N14, full, run 2; *validation discrepancies*). Also "outro trecho do texto" (*another passage*).\n',
                  encoding="utf-8")
    S = P.Sources({"sources": {"a4_examples": str(md)}})
    q = records.quotes_a4(S)
    assert [x["quote_id"] for x in q] == ["A4-2-01", "A4-2-02", "A4-2-03"]
    assert q[0]["original_pt"] == "Texto **em negrito** do rascunho." and q[0]["translation_en"] == "Draft text **in bold**."
    assert q[1]["original_pt"] == "divergências de validação" and "N14, full, run 2" in q[1]["location"]
    assert q[2]["translation_en"] == "another passage"


def test_review_marks_apply_only_to_an_unchanged_original(tmp_path):
    quotes = [{"quote_id": "Q1", "original_pt": "um", "translation_en": "one", "translation_by": "draft", "reviewed": False},
              {"quote_id": "Q2", "original_pt": "dois (novo)", "translation_en": "two", "translation_by": "draft", "reviewed": False}]
    sheet = tmp_path / "quotes_review.xlsx"
    pd.DataFrame({"quote_id": ["Q1", "Q2"], "original_pt": ["um", "dois"], "reviewed": [1, 1],
                  "translation_en_reviewed": ["one (reviewed)", None]}).to_excel(sheet, index=False)
    assert records.apply_review(quotes, sheet) == 1
    assert quotes[0]["reviewed"] and quotes[0]["translation_en"] == "one (reviewed)"
    assert not quotes[1]["reviewed"]


# ------------------------------------------------------------------------------------------- assets

@pytest.mark.parametrize("name", sorted(n for n in assets.TABLES if not n.startswith("tab_b2")))
def test_every_table_builds_from_results(name):
    S = P.Sources()
    table = assets.TABLES[name][0](S, name)
    assert len(table) > 0 and not table.columns.duplicated().any()
    assert S.used[name]                                         # every table names its sources


@pytest.mark.parametrize("name", sorted(assets.FIGURES))
def test_every_figure_draws(name):
    import matplotlib.pyplot as plt
    fig = assets.FIGURES[name][0](P.Sources(), name)
    assert fig.axes
    plt.close(fig)


def test_an_absent_optional_source_skips_what_reads_it(tmp_path):
    """The public release has no B2 workbook: its table and RESULTS.md section are skipped, not fatal."""
    from jm_eval.io import load_config
    cfg = load_config("paper_assets")
    cfg["sources"]["b2_workbook"] = str(tmp_path / "absent.xlsx")
    S = P.Sources(cfg)
    assert not any(p.startswith("b2_workbook") for p in S.check())
    assert "b2_workbook" in S.missing_optional()
    with pytest.raises(P.MissingOptionalSource):
        assets.TABLES["tab_b2_constructs"][0](S, "tab_b2_constructs")
    numbers = load_config("paper_numbers")
    numbers["sections"] = [s for s in numbers["sections"] if s["id"] in ("b2", "b3")]
    text, n = records.build_results(S, numbers, {})
    assert "Not rebuilt in this copy" in text and n > 0           # B3's numbers are still there
