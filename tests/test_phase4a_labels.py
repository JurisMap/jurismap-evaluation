import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jm_eval.io import latest_artefact_path, new_artefact_path
from jm_eval.llm import GeminiJudge
from jm_eval.retrieval import labels as L
from jm_eval.retrieval.provisions import build_canon, parse, provision_ids
from jm_eval.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


silver = _load("silver_labels", "scripts/40_silver_labels.py")
category = _load("infraction_category", "scripts/41_infraction_category.py")
audit = _load("labels_audit", "scripts/42_labels_audit.py")


# ---------------------------------------------------------------- citation parser
@pytest.mark.parametrize("text, expected", [
    ("art. 113 do Decreto-Lei nº 73/1966", ["DL73/1966:art113"]),                       # the plan's example
    ("parágrafo único do artigo 757 do Código Civil, artigos 24 e 113 do Decreto-Lei n° 73/66",
     ["CC:art757", "DL73/1966:art24", "DL73/1966:art113"]),
    ("Art. 9º da Resolução CNSP nº 98, de 30 de setembro de 2002, c.c. art. 5º, inc. II, alínea \"n\" da "
     "Resolução CNSP nº 60/2001.", ["RES-CNSP98/2002:art9", "RES-CNSP60/2001:art5"]),
    ("Art. 88 Decreto-Lei nº 73/1966 c/c art. 773 da Lei nº 10.406/2002.", ["DL73/1966:art88", "CC:art773"]),
    ("Art. 8º da Lei 7492 c/c artigo 4º, inciso IV da Circular SUSEP nº 320/2006.",
     ["LEI7492/1986:art8", "CIRC-SUSEP320/2006:art4"]),
    ("Art. 5º, XXXIX, CF/88", ["CF:art5"]),
    ("Sanção: Artigo 35-B da Resolução CNSP nº 243/2011", ["RES-CNSP243/2011:art35B"]),
])
def test_citations_are_normalised(text, expected):
    assert provision_ids(text) == expected


def test_paragraphs_incisos_items_and_annexes_are_not_articles():
    text = ("Art. 1º, Anexo XXVII; art. 2º, §1º e §2º, Anexo XXIX; art. 3º, II e III; e art. 4º, § 2º, "
            "da Resolução CNSP nº 321, de 2015 (itens 1, 2 e 3)")
    assert provision_ids(text) == [f"RES-CNSP321/2015:art{n}" for n in (1, 2, 3, 4)]
    assert provision_ids("Arts 3º e 108, § 1º da Circular SUSEP nº 302/2005") == [
        "CIRC-SUSEP302/2005:art3", "CIRC-SUSEP302/2005:art108"]
    assert provision_ids("arts. 485, VI e 932, III, do CPC") == ["CPC:art485", "CPC:art932"]


def test_articles_take_the_norm_before_them_or_a_reference_back():
    assert provision_ids("Art. 88 do Decreto-Lei 73/66 c/c Circular SUSEP 364/08, art. 2º") == [
        "DL73/1966:art88", "CIRC-SUSEP364/2008:art2"]
    assert provision_ids("Circular SUSEP n.º 430/2012, anexo 1, art. 8°, §6° e §7°, e art. 12") == [
        "CIRC-SUSEP430/2012:art8", "CIRC-SUSEP430/2012:art12"]
    assert provision_ids("com base no art. 19 da Resolução CNSP nº 243/2011 c/c arts. 9º, 10 e 11, III da citada norma") == [
        f"RES-CNSP243/2011:art{n}" for n in (19, 9, 10, 11)]
    # a norm already taken by the article before it is not taken again by the next article
    assert provision_ids("Art. 88 do Decreto-Lei nº 73/1966, Art. 1º da Circular SUSEP nº 251/2004") == [
        "DL73/1966:art88", "CIRC-SUSEP251/2004:art1"]


def test_amending_acts_and_articles_without_a_norm_are_left_out():
    text = ("Art. 88 do Decreto-Lei nº 73/1966, com redação dada pela Lei Complementar nº 126/2007, c/c os "
            "artigos 167 e 169 da Circular SUSEP nº 517/2015, com as alterações promovidas pela Circular SUSEP nº 544/2016")
    assert provision_ids(text) == ["DL73/1966:art88", "CIRC-SUSEP517/2015:art167", "CIRC-SUSEP517/2015:art169"]
    assert provision_ids("art. 59 do regulamento do plano") == []
    assert [p.id for p in parse("art. 59 do regulamento do plano")] == ["?:art59"]


def test_norm_without_article_and_ocr_forms():
    assert provision_ids("Tabela PREMIT, constante do Anexo VIII da Circular SUSEP nº 360/2008.") == ["CIRC-SUSEP360/2008"]
    assert provision_ids("Art. 2º da Circular SUSEP 364/08 c/c Circular SUSEP nº 364/2008") == ["CIRC-SUSEP364/2008:art2"]
    assert provision_ids("Art. 88 do Decreto-Lei n9 73/66") == ["DL73/1966:art88"]
    assert provision_ids("Artigo 6° da Lei Cornplernentar no 109/2001") == ["LC109/2001:art6"]
    assert provision_ids("Art. 72 do Anexo I da Resolucâo CNSP n2 86/02") == ["RES-CNSP86/2002:art72"]
    assert provision_ids("Art. 26, §5º do Regimento Interno do CRSNSP") == ["RICRSNSP:art26"]
    assert provision_ids(None) == [] and provision_ids(float("nan")) == []


def test_slips_found_in_the_first_audit_draw():
    assert provision_ids("art. 72, § 1° e 3º da Circular SUSEP nº 302/2005") == ["CIRC-SUSEP302/2005:art72"]
    assert provision_ids("art. 26 da Resolucão CNSP 0 118/2004") == ["RES-CNSP118/2004:art26"]
    assert provision_ids("anexo VIII, da Circular Susep n.º 360/2008; art. 2º, caput e § 4º, da Circular SUSEP nº 251/2004") == [
        "CIRC-SUSEP360/2008", "CIRC-SUSEP251/2004:art2"]
    assert provision_ids("Art. 1º do Regulamento anexo à Resolução nº 3.305/2005") == ["RES-CMN3305/2005:art1"]
    assert provision_ids("itens 58 e 59 do CPC n° 38") == ["PRON-CPC38"]            # accounting pronouncement
    assert provision_ids("Art. 1.022 do Código de Processo Civil") == ["CPC:art1022"]
    assert provision_ids("Art. 88 do Decreto-Lei nº 73/1966, sob a redação da Lei Complementar nº 126/2007") == ["DL73/1966:art88"]


def test_slips_found_by_the_audit():
    # corrected after the audit; the audited accuracy refers to the labels before these corrections
    assert provision_ids("Artigo 9° c/c 10° c/c 12 c/c 13 do Anexo I da Circular SUSEP n° 508/2015") == [
        f"CIRC-SUSEP508/2015:art{n}" for n in (9, 10, 12, 13)]
    assert provision_ids("Arts. 33, 36, 37, § 3º, 38, 40 , I e II da Resolução CNSP n° 168/2007.") == [
        f"RES-CNSP168/2007:art{n}" for n in (33, 36, 37, 38, 40)]
    assert provision_ids("art. 139, §§ 1º, 2º e 13, da Resolução CNSP nº 393/2020") == ["RES-CNSP393/2020:art139"]
    assert provision_ids("Art. 88 do Decreto-Lei 73/66 c/c art. 2º da Circular SUSEP 364/08") == [
        "DL73/1966:art88", "CIRC-SUSEP364/2008:art2"]                              # "c/c" before a new "art." still splits


def test_successor_provision_is_left_out():
    # The author, 2026-10-09 (review row A16): a provision named only as the substitute of the cited one is not cited
    text = 'Art. 104, alínea "f", Decreto nº 81402/1978 (vigente na contratação do plano), substituído pelo art. 6º da Lei Complementar nº 109/2001.'
    assert provision_ids(text) == ["DEC81402/1978:art104"]


def test_slip_found_by_the_second_reviewer():
    assert provision_ids("c/c do Art. 7, § 3º, Anexo I Circ SUSEP 365/2008 (itens 3, 4 e 5); Art. 88 do Decreto-Lei nº 73/1966") == [
        "CIRC-SUSEP365/2008:art7", "DL73/1966:art88"]
    assert provision_ids("Art. 2º da Circ. SUSEP nº 364/2008") == ["CIRC-SUSEP364/2008:art2"]
    assert provision_ids("o circo 12/2008, art. 3º do Decreto-Lei nº 73/1966") == ["DL73/1966:art3"]


def test_canon_completes_year_and_issuer_from_the_corpus():
    canon = build_canon(["Resolução CNSP nº 321/2015", "Resolução CNSP 321, de 2015", "Circular SUSEP nº 517/2015", None])
    assert canon[("RES", 321)] == ("CNSP", 2015)
    assert provision_ids("art. 112 da Resolução nº 321", canon) == ["RES-CNSP321/2015:art112"]
    assert provision_ids("art. 110 da Circular 517", canon) == ["CIRC-SUSEP517/2015:art110"]


# ---------------------------------------------------------------- labels
def test_process_keys_and_groups():
    assert L.process_keys("Processo 15414.629444/2019-55 - Apenso Processo Susep 15414.629448/2019-32") == {
        "15414629444201955", "15414629448201932"}
    assert L.process_keys("Recurso 7.392") == {"R7392"} != L.process_keys("Recurso 7.402")
    assert L.process_keys("Processo sem número", "https://x/15414-621879-2017-90/view") == {"15414621879201790"}
    groups = L.process_groups([{"a"}, {"b"}, {"a", "c"}, {"c"}, {"d"}])
    assert groups[0] == groups[2] == groups[3] and len({groups[0], groups[1], groups[4]}) == 3


def test_penalty_types_outcome_and_specific_key():
    assert L.penalty_types("Item 1 - multa no valor de R$ 10.000,00; Item 2 - advertência") == ["multa", "advertencia"]
    assert L.penalty_types("R$3.000.000,00") == ["multa"] and L.penalty_types(None) == []
    assert L.penalty_types("Cancelamento do Registro") == ["cancelamento_registro"]
    assert L.outcome("Não provido") == "nao_provido" and L.outcome(None) is None
    ids = ["DL73/1966:art88", "CIRC-SUSEP364/2008:art2", "RES-CNSP243/2011:art37"]
    spec = L.specific(ids, ["DL73/1966:art88"], ["RES-CNSP243/2011"])
    assert spec == ["CIRC-SUSEP364/2008:art2"] and L.key_of(spec) == "CIRC-SUSEP364/2008:art2"
    assert L.key_of([]) is None and L.key_of(["b", "a", "b"]) == "a|b"


def test_infringed_provisions_stop_where_the_text_turns_to_the_penalty():
    text = ("A representada enviou o FIP fora do prazo, infringindo o art. 2º da Circular SUSEP nº 364/08 c/c arts. 36 e 88 "
            "do Decreto-Lei nº 73/66. A penalidade aplicável se encontra no art. 37 da Resolução CNSP nº 243/11. "
            "Julgada subsistente por infração ao art. 2º da Circular SUSEP nº 364/2008, com penalidade prevista no "
            "art. 37 da Resolução CNSP nº 243/2011.")
    found = L.infringed_provisions(text)
    assert found[0] == "CIRC-SUSEP364/2008:art2"                      # cited twice as infringed
    assert set(found) == {"CIRC-SUSEP364/2008:art2", "DL73/1966:art36", "DL73/1966:art88"}
    assert "RES-CNSP243/2011:art37" in L.cited_provisions(text)
    assert L.infringed_provisions("O recurso é tempestivo.") == [] and L.infringed_provisions(None) == []


def test_relevance_matrix_strict_relaxed_and_same_process():
    sets = [{"a", "b"}, {"b", "a"}, {"a"}, set(), set(), {"a", "b"}]
    groups = [0, 1, 2, 3, 4, 0]                                     # decisions 0 and 5: same process
    strict = L.relevance_matrix(sets, groups)
    assert strict[0, 1] and strict[1, 0] and strict[1, 5]
    assert not strict[0, 5] and not strict[0, 0]                   # same process, self
    assert not strict[0, 2] and not strict[3, 4]                   # different sets; empty sets never match
    relaxed = L.relevance_matrix(sets, groups, kind="jaccard", threshold=0.5)
    assert relaxed[0, 2] and relaxed[2, 1] and not relaxed[3, 4] and not relaxed[0, 5]
    assert L.jaccard_matrix([{"a", "b"}, {"a"}])[0, 1] == pytest.approx(0.5)
    summary = L.relevance_summary(strict)
    assert summary["decisions"] == 6 and summary["queries_with_relevant"] == 3
    with pytest.raises(ValueError):
        L.relevance_matrix(sets, groups, kind="other")


def _corpus():
    base = "Art. 88 do Decreto-Lei nº 73/1966 c/c art. 2º da Circular SUSEP nº 364/2008"
    report = "Envio do FIP fora do prazo, infringindo o art. 2º da Circular SUSEP nº 364/2008. Penalidade: art. 37."
    rows = [
        ("h1", "Processo 15414.600001/2019-11", base, "Advertência", "Não provido", report),
        ("h2", "Processo 15414.600002/2019-22", "Art. 2º da Circular SUSEP 364/08", "Multa de R$ 9.000,00", "Provido", report),
        ("h3", "Processo 15414.600001/2019-11 (Retificação)", base, "Advertência", "Não provido", report),
        ("h4", "Recurso 7001", "Art. 88 do Decreto-Lei nº 73/1966", None, "Extinto", "Sem citação."),
        ("h5", "Recurso 7002", None, "Multa", "Parcialmente provido", "Infração ao art. 113 do Decreto-Lei nº 73/1966."),
    ]
    df = pd.DataFrame(rows, columns=["acordao_hash", "acordao_titulo", "acordao_base", "acordao_penalidade",
                                     "acordao_resultado", "acordao_relatorio"])
    df["acordao_url"], df["acordao_voto_relator"] = None, None
    return df


CFG = {"generic_provisions": ["DL73/1966:art88"], "sanction_norms": ["RES-CNSP243/2011"],
       "development": {"ids": ["h5"]}, "relevance": {"jaccard_threshold": 0.5}}


def test_build_labels_coverage_and_relevance_on_a_small_corpus():
    corpus = _corpus()
    labels = silver.build_labels(corpus, CFG, shared={"h1", "h2", "h3"})
    lab = labels.set_index("acordao_hash")
    assert lab.at["h1", "base_key"] == lab.at["h2", "base_key"] == "CIRC-SUSEP364/2008:art2"
    assert pd.isna(lab.at["h4", "base_key"]) and list(lab.at["h4", "base_provisions"]) == ["DL73/1966:art88"]
    assert pd.isna(lab.at["h5", "base_key"]) and lab.at["h5", "text_primary"] == "DL73/1966:art113"
    assert lab.at["h1", "process_group"] == lab.at["h3", "process_group"] != lab.at["h2", "process_group"]
    assert lab.at["h2", "penalty_key"] == "multa" and pd.isna(lab.at["h4", "penalty_key"])
    assert lab.at["h5", "in_development_sample"] and not lab.at["h5", "in_comparison_set"]

    cov = silver.coverage(labels, corpus).set_index(["scope", "label"])
    assert cov.at[("v2", "R-base key (a specific provision in the basis)"), "with_label"] == 3
    assert cov.at[("v2", "R-base key or infringed provision"), "with_label"] == 4
    assert cov.at[("comparison_set", "outcome"), "share"] == 1.0

    definitions = silver.relevance_definitions(labels, 0.5)
    strict = definitions["v2"]["R-base strict (identical specific provisions)"]
    assert strict["queries_with_relevant"] == 3                     # h1-h2, h3-h2; h1-h3 are one process
    agreement = silver.source_agreement(labels)["other_decisions"]
    assert agreement["decisions_with_base_key"] == 3 and agreement["primary_provision_is_in_base"] == 1.0
    assert silver.distributions(labels)["v2"]["process_groups"]["groups_with_several_decisions"] == 1


# ---------------------------------------------------------------- category labelling
TAXONOMY = category.load_taxonomy()


def test_taxonomy_is_closed_and_the_prompt_holds_it():
    ids = category.category_ids(TAXONOMY)
    assert len(ids) == len(set(ids)) and 10 <= len(ids) <= 15 and ids[-1] == "C99"
    assert category.response_schema(TAXONOMY)["properties"]["primary"]["enum"] == ids
    prompt = category.build_prompt(TAXONOMY, "EMENTA X", "relatório " * 5000)
    assert all(f"{i}: " in prompt for i in ids) and "EMENTA X" in prompt
    assert len(prompt) < TAXONOMY["input"]["max_report_chars"] + 12000      # the report is cut


def test_validate_and_verbatim_evidence():
    ids = category.category_ids(TAXONOMY)
    label, problem = category.validate({"primary": "C05", "secondary": ["C05", "C09", "C07", "C08"], "evidence": "x"}, ids)
    assert problem is None and label["secondary"] == ["C09", "C07"]
    assert category.validate({"primary": "C77"}, ids)[0] is None
    assert category.validate({"primary": "C05", "secondary": ["nope"]}, ids)[0] is None
    assert category.validate("C05", ids)[0] is None
    assert category.is_verbatim("envio do FIP, fora do prazo", "Trata-se de envio do FIP fora do prazo.")
    assert not category.is_verbatim("atraso inventado", "Trata-se de envio do FIP fora do prazo.")


class FakeModel:
    """Returns the queued answers in order and records the cache extras it was called with."""

    def __init__(self, answers):
        self.answers, self.extras = list(answers), []

    def complete(self, prompt, *, system=None, json_output=False, cache_extra=None):
        self.extras.append(cache_extra)
        answer = self.answers.pop(0)
        if answer is None:
            raise ValueError("no JSON")
        return SimpleNamespace(parsed=answer, cached=False, input_tokens=100, output_tokens=10, cost_usd=0.001,
                               latency_s=0.1, credentials="primary")


def test_label_one_asks_again_after_an_unusable_answer():
    row = {"acordao_hash": "h1", "acordao_ementa": "Envio do FIP fora do prazo.", "acordao_relatorio": "Relatório."}
    model = FakeModel([None, {"primary": "C77"}, {"primary": "C05", "secondary": [], "evidence": "Envio do FIP fora do prazo"}])
    record = category.label_one(model, TAXONOMY, row)
    assert record["primary"] == "C05" and record["format_retries"] == 2 and record["evidence_verbatim"]
    assert [e.get("format_retry") for e in model.extras] == [None, 1, 2]     # each retry has its own cache entry
    assert len(record["calls"]) == 2                                         # the non-JSON answer has no usage
    failed = category.label_one(FakeModel([{"primary": "C77"}] * 3), TAXONOMY, row)
    assert failed["primary"] is None and failed["error"]

    dist = category.distribution(pd.DataFrame([record, failed]), TAXONOMY).set_index("category")
    assert dist.at["C05", "primary"] == 1 and dist.at["C05", "share_primary"] == 1.0


def test_corpus_is_not_labelled_with_a_draft_taxonomy(monkeypatch):
    monkeypatch.setattr(category, "load_taxonomy", lambda: {**TAXONOMY, "status": "draft"})
    monkeypatch.setattr(sys, "argv", ["41_infraction_category.py"])
    with pytest.raises(SystemExit, match="approved"):
        category.main()


def test_labeler_parameters_do_not_change_the_judges_cache_keys():
    judge = GeminiJudge("judge_gemini", {"model": "m", "thinking_level": "high"}, cache=object())
    assert judge.params() == {"thinking_level": "high"}
    labeler = GeminiJudge("labeler", {"model": "m", "thinking_level": "low", "temperature": 0.0,
                                      "response_schema": {"type": "OBJECT"}}, cache=object())
    assert labeler.params() == {"thinking_level": "low", "temperature": 0.0, "response_schema": {"type": "OBJECT"}}


# ---------------------------------------------------------------- audit
def test_audit_sample_sheet_and_accuracy(tmp_path):
    corpus = _corpus()
    corpus["acordao_ementa"] = "Ementa."
    labels = silver.build_labels(corpus, CFG, shared=set())
    ids = audit.sample_ids(labels, 3, seed=1)
    assert ids == audit.sample_ids(labels, 3, seed=1) and "h5" not in ids and len(ids) == 3
    assert "h1" not in audit.sample_ids(labels, 4, seed=1, excluded={"h1"})
    categories = pd.DataFrame({"acordao_hash": ids, "primary": ["C05"] * 3, "secondary": [["C09"], [], []],
                               "evidence": ["FIP fora do prazo"] * 3})
    table = audit.audit_table(labels, corpus, categories, TAXONOMY, ids)
    assert table["label_category"].str.startswith("C05: Periodic information").all()
    h1 = table.set_index("acordao_hash").loc["h1"] if "h1" in ids else None
    if h1 is not None:
        assert h1["label_base"] == "[DL73/1966:art88]; CIRC-SUSEP364/2008:art2"      # generic in brackets
    path = tmp_path / "labels_audit.xlsx"
    audit.write_workbook(path, table, TAXONOMY)

    filled = pd.read_excel(path, sheet_name="audit")
    assert list(pd.read_excel(path, sheet_name="taxonomy")["id"]) == category.category_ids(TAXONOMY)
    with pytest.raises(SystemExit, match="not filled"):
        audit.accuracy(filled)
    filled = filled.astype({"correct_category": object})
    for ok_column, label_column in audit.CHECKS.values():
        filled[ok_column] = filled[label_column].notna().map({True: 1.0, False: np.nan})
    filled.loc[0, "ok_category"], filled.loc[0, "correct_category"] = 0, "C06"
    result = audit.accuracy(filled)
    assert result["infraction_category"]["correct"] == 2 and result["infraction_category"]["judged"] == 3
    assert result["infraction_category"]["corrections"] == [{"labelled": "C05", "correct": "C06"}]
    assert result["legal_basis"]["accuracy"] == 1.0
    assert result["penalty_type"]["judged"] == result["penalty_type"]["with_label"]    # empty labels are not judged

    # a later sheet for the same decisions keeps the marks whose label did not change
    first = audit.audit_table(labels, corpus, None, None, ids)
    assert "ok_category" not in first.columns
    first["ok_base"], first["notes"] = [1.0, 0.0, np.nan], ["checked", None, None]
    changed = table.copy()
    changed.loc[1, "label_base"] = "something else"
    carried, copied = audit.carry_marks(changed, first)
    assert copied == 1 and carried.loc[0, "ok_base"] == 1.0 and pd.isna(carried.loc[1, "ok_base"])
    assert carried.loc[0, "notes"] == "checked" and carried["ok_category"].isna().all()

    filled.loc[1, "ok_base"] = 2
    with pytest.raises(SystemExit, match="other than 0 or 1"):
        audit.accuracy(filled)


def test_latest_artefact_path(tmp_path):
    base = tmp_path / "labels.parquet"
    assert latest_artefact_path(base) == base
    for name in ("labels.parquet", "labels_2.parquet", "labels_10.parquet", "labels_pilot.parquet"):
        (tmp_path / name).touch()
    assert latest_artefact_path(base).name == "labels_10.parquet"
    assert new_artefact_path(base).name == "labels_11.parquet"
    (tmp_path / "labels.parquet").unlink()                         # a deleted first version is not reused
    assert new_artefact_path(base).name == "labels_11.parquet"
    assert new_artefact_path(tmp_path / "other.csv").name == "other.csv"


def test_wilson_interval():
    low, high = wilson_interval(54, 60)
    assert low == pytest.approx(0.7985, abs=1e-3) and high == pytest.approx(0.9534, abs=1e-3)
    assert wilson_interval(60, 60)[1] == 1.0 and all(np.isnan(wilson_interval(0, 0)))
