"""Phase 8 (A4), task 4: template coverage and precedent citations of the drafts."""

from __future__ import annotations

import pandas as pd

from jm_eval.drafting import metrics as M

SECTIONS = {"III – Dos Fatos": ["fatos", "fatica"], "VI – Do Direito": ["direito", "fundamentacao juridica"],
            "VIII – Dos Pedidos": ["pedidos", "requerimentos"]}


def test_coverage_reads_headings_only_and_accepts_the_section_names():
    draft = ("# DEFESA\n\n**IV – SÍNTESE FÁTICA E DO PROCESSO**\n\nO direito da recorrente...\n\n"
             "V – DO MÉRITO E DA FUNDAMENTAÇÃO JURÍDICA\n\ntexto\n\n### VIII – Dos Pedidos e Requerimentos\n")
    out = M.section_coverage(draft, SECTIONS)
    assert out["sections_found"] == {"III – Dos Fatos": True, "VI – Do Direito": True, "VIII – Dos Pedidos": True}
    merged = "**V – DO MÉRITO**\n\nO direito aplicável...\n\n**VIII – DOS PEDIDOS**"
    assert M.section_coverage(merged, SECTIONS)["sections_found"]["VI – Do Direito"] is False   # "direito" in the body only
    assert abs(M.section_coverage(merged, SECTIONS)["coverage"] - 1 / 3) < 1e-9


def toy_index():
    corpus = pd.DataFrame({"acordao_hash": ["a", "b", "src"],
                           "acordao_acordao": ["ACÓRDÃO CRSNSP 7038/2021 Vistos", "ACÓRDÃO CRSNSP 6.213 RETIFICAÇÃO", "Vistos..."]})
    labels = pd.DataFrame({"acordao_hash": ["a", "b", "src"],
                           "process_keys": [["15414611111201801"], ["15414622222201902"], ["15414633333202003"]]})
    return M.CorpusIndex(corpus, labels)


def test_citations_are_resolved_to_the_corpus():
    index = toy_index()
    draft = ("Conforme o Acórdão CRSNSP nº 7038/2021 (Processo nº 15414.611111/2018-01) e os Acórdãos 6213/2019 e 9999/2022, "
             "bem como o Processo SUSEP nº 15414.699999/2020-11 e o Proc. 15414.633333/2020-03.")
    out = M.classify_citations(draft, index, supplied=["a"], source="src")
    status = {(c["kind"], c["key"]): c["status"] for c in out["citations"]}
    assert status[("ruling", "7038")] == "supplied" and status[("process", "15414611111201801")] == "supplied"
    assert status[("ruling", "6213")] == "corpus" and status[("ruling", "9999")] == "not_found"
    assert status[("process", "15414699999202011")] == "not_found" and status[("process", "15414633333202003")] == "source"
    assert out["cited_supplied"] == 1 and out["cited_corpus_not_supplied"] == 1 and out["cited_source"]
    assert out["process_not_in_corpus"] == 1 and out["ruling_unresolved"] == 1 and out["supplied_used_share"] == 1.0


TABLE = {"DL73/1966": {"articles": ["24", "36", "88", "113"], "last_article": 153},
         "RES-CNSP243/2011": {"articles": ["9", "10", "12", "37"], "last_article": 153},
         "RES-CNSP388/2020": {"articles": ["1", "14"], "last_article": 14},
         "L10406/2002": {"articles": ["265", "757"], "last_article": 2046}}


def test_statute_citations_tie_articles_to_acts():
    draft = ("nos arts. 36 e 88 do DL 73/66; no Decreto-Lei nº 73/1966 (arts. 113 e 24), na Lei nº 9.784/1999; "
             "arts. 9º, 10 e 12 da Res. CNSP nº 243, de 7 de dezembro de 2011; art. 133 da Resolução CNSP nº 388/2020; "
             "art. 150 da Resolução CNSP nº 243/2011; art. 2º do Decreto nº 10.016/2019; o artigo 12 do mesmo normativo. "
             'Código Civil: > *"Art. 265. A solidariedade não se presume"* e o Decreto-Lei nº 73/1966.')
    out = M.classify_statutes(draft, TABLE)
    got = {(r["act"], r["article"]): r["status"] for r in out["statutes"]}
    assert got[("DL73/1966", "36")] == got[("DL73/1966", "88")] == "found"
    assert got[("DL73/1966", "113")] == got[("DL73/1966", "24")] == "found"           # the act named just before, in parentheses
    assert got[("RES-CNSP243/2011", "9")] == got[("RES-CNSP243/2011", "12")] == "found"
    assert got[("RES-CNSP388/2020", "133")] == "beyond" and got[("RES-CNSP243/2011", "150")] == "in_range"
    assert got[("DEC10016/2019", "2")] == "unknown_act" and got[("L10406/2002", "265")] == "found"
    assert got[(None, "12")] == "no_act"
    assert out["statute_beyond"] == 1 and out["statute_invalid_share"] == 1 / (out["statute_found"] + 2)


def test_judge_answers_and_added_facts():
    from jm_eval.drafting import judging as DJ
    ok = {"scores": dict.fromkeys(DJ.CRITERIA, 3), "added_facts": ["intimação em 12/03/2021", "CPF [A PREENCHER: CPF]"], "reason": "x"}
    assert DJ.validate_rubric(ok)[0]["scores"]["structure"] == 3
    assert DJ.validate_rubric({**ok, "scores": {**ok["scores"], "structure": 6}})[0] is None
    assert DJ.added_facts(ok) == ["intimação em 12/03/2021"]
    assert DJ.added_facts({"added_facts": ["Nome do Advogado – OAB/UF nº XXXXX", "multa de R$ 9.000,00"]}) == ["multa de R$ 9.000,00"]
    guard = {"findings": [{"kind": "date", "text": "12/03/2021", "action": "flagged"}]}
    assert DJ.missed_by_guard(["intimação em 12/03/2021", "reunião com a SUSEP"], guard) == ["reunião com a SUSEP"]
    assert DJ.validate_arguments({"arguments": [{"id": 2, "label": "absent"}, {"id": 1, "label": "raised"}]}, 2)[0] == {"labels": ["raised", "absent"]}
    assert DJ.validate_arguments({"arguments": [{"id": 1, "label": "raised"}]}, 2)[0] is None


def test_ruling_and_process_cited_together_must_be_the_same_decision(tmp_path):
    corpus = pd.DataFrame({"acordao_hash": ["a", "p"], "acordao_acordao": ["ACÓRDÃO CRSNSP 7038/2021 Vistos", "Vistos..."]})
    labels = pd.DataFrame({"acordao_hash": ["a", "p"], "process_keys": [["15414611111201801"], ["15414626623201779"]]})
    (tmp_path / "p.txt").write_text("... ACÓRDÃO CRSNSP 7357/2021 Vistos ...", encoding="utf-8")
    index = M.CorpusIndex(corpus, labels, tmp_path)
    assert index.resolve("ruling", "7357") == {"p"}                          # read from the full text
    draft = ("No **Acórdão CRSNSP nº 286/2021** (Processo SUSEP nº 15414.626623/2017-79), e no Acórdão CRSNSP 7357/2021 "
             "(Processo nº 15414.626623/2017-79). Item 18.4.b do Proc. SUSEP 15414.100326/2004-83.")
    out = M.classify_citations(draft, index, supplied=["p"], source="src",
                               notice="condições do Proc. SUSEP 15414.100326/2004-83")
    assert out["ruling_process_mismatch"] == 1 and out["mismatches"][0]["ruling"] == "286/2021"
    status = {(c["kind"], c["key"]): c["status"] for c in out["citations"]}
    assert status[("process", "15414100326200483")] == "from_notice" and out["process_not_in_corpus"] == 0
    assert status[("ruling", "286")] == "not_found" and status[("ruling", "7357")] == "supplied"
    listed = ("no Acórdão CRSNSP 7038/2021 (Processo nº 15414.626623/2017-79) e no Acórdão CRSNSP 7357/2021 "
              "(Processo nº 15414.611111/2018-01)")                      # a list: each ruling with the process after it
    assert M.classify_citations(listed, index, supplied=["p"], source="src")["ruling_process_mismatch"] == 2
    swapped = "conforme o Processo nº 15414.626623/2017-79) e o Acórdão CRSNSP 7357/2021 - Processo nº 15414.626623/2017-79"
    assert M.classify_citations(swapped, index, supplied=["p"], source="src")["ruling_process_mismatch"] == 0
