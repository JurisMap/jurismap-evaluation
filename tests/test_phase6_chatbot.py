"""Phase 6 (A3 chatbot evaluation): case sets, questions, deterministic scoring, evidence retrieval,
metrics and the checks on judge calls. No model is called and no private file is read."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jm_eval.chatbot import casesets
from jm_eval.chatbot import controls as K
from jm_eval.chatbot import evidence as E
from jm_eval.chatbot import judging as J
from jm_eval.chatbot import metrics as M
from jm_eval.chatbot import nli as N
from jm_eval.chatbot import questions as Q
from jm_eval.chatbot import scoring as S
from jm_eval.io import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config("a3_chatbot")
PATTERNS = CFG["scoring"]["abstention_patterns"]


# ---------------------------------------------------------------- case sets
def _labels_and_vectors(n=120, seed=0):
    rng = np.random.default_rng(seed)
    keys = [f"K{i % 4}" for i in range(n)]                       # four strata of 30
    labels = pd.DataFrame({"acordao_hash": [f"h{i:03d}" for i in range(n)], "base_key": keys,
                           "process_group": [i // 2 for i in range(n)]})     # two decisions per process
    vectors = rng.normal(size=(n, 8))
    return labels, vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_sets_are_disjoint_exclude_the_seed_process_and_are_reproducible():
    labels, vectors = _labels_and_vectors()
    sets = casesets.build_sets(labels, vectors, n_sets=3, size=10, seed=7, development_sets=1)
    assert [s["set_id"] for s in sets] == ["S1", "S2", "S3", "D1"] and sets[3]["development"]
    members = [h for s in sets for h in s["case_hashes"]]
    assert len(members) == 40 == len(set(members))
    group = dict(zip(labels["acordao_hash"], labels["process_group"]))
    stratum = dict(zip(labels["acordao_hash"], labels["base_key"]))
    for s in sets:
        assert stratum[s["seed_hash"]] == s["stratum"] and s["seed_hash"] not in members
        assert group[s["seed_hash"]] not in {group[h] for h in s["case_hashes"]}
        assert s["similarity"] == sorted(s["similarity"], reverse=True)
    assert len({s["stratum"] for s in sets}) == 4
    assert sets == casesets.build_sets(labels, vectors, n_sets=3, size=10, seed=7, development_sets=1)


def test_strata_are_the_most_frequent_keys():
    labels = pd.DataFrame({"base_key": ["b", "b", "a", "a", "c", None]})
    assert casesets.strata(labels, 2) == ["a", "b"]


# ---------------------------------------------------------------- questions
def _corpus(n=14):
    rows = []
    for i in range(n):
        rows.append({
            "acordao_hash": f"h{i:02d}", "acordao_titulo": f"Processo 15414.{600000 + i}/2019-{10 + i}\t",
            "ano_sessoes": 2020 + i % 2, "sessao_nr_int": 280 + i, "sessao_datas": "25 e 26 de agosto",
            "sessao_acordao": "https://example.org/sessao", "acordao_url": f"https://example.org/{i}",
            "acordao_recorrente": np.array([f"Seguradora {i} S.A."]), "acordao_recorrido": np.array(["SUSEP"]),
            "acordao_relator": np.array(["Maria da Silva" if i % 3 else "João Souza"]),
            "acordao_advogado": np.array([f"Advogado {i}"]), "acordao_ementa": f"Ementa {i}",
            "acordao_penalidade": "Multa de R$ 8.000,00 (oito mil reais)",
            "acordao_base": "art. 2º da Circular SUSEP nº 364/2008 c/c art. 88 do Decreto-Lei nº 73/66",
            "acordao_acordao": "Acórdão", "acordao_participantes": "Participaram os Conselheiros",
            "acordao_relatorio": f"Relatório do caso {i}." + (" CNPJ 12.345.678/0001-90" if i == 0 else ""),
            "acordao_voto_relator": "Voto", "acordao_resultado": "Não provido" if i % 2 else "Provido"})
    return pd.DataFrame(rows)


def test_question_set_of_one_case_set():
    corpus = _corpus()
    case_set = {"set_id": "S2", "case_hashes": [f"h{i:02d}" for i in range(10)]}
    qs = Q.build_questions(case_set, 1, corpus, CFG, None, {f"h{i:02d}" for i in range(10)}, seed=3)
    assert [q["type"] for q in qs] == ["templated"] * 4 + ["unanswerable"] + ["open"] * 2
    assert [q["subtype"] for q in qs[:4]] == CFG["questions"]["templated_by_set"][1]
    assert len({q["question_id"] for q in qs}) == 7 and all(q["case_hashes"] == case_set["case_hashes"] for q in qs)
    by = {q["subtype"]: q for q in qs}
    assert by["session_date"]["gold"] == {"days": [25, 26], "month": 8, "year": by["session_date"]["gold"]["year"]}
    assert by["penalty"]["gold"] == {"types": ["multa"], "amounts": [8000.0]}
    assert by["appellant"]["gold"][0].startswith("Seguradora") and "\t" not in by["appellant"]["question"]
    assert len({q["target_hash"] for q in qs[:4]}) == 4                      # a different decision per question
    cnpj = by["appellant_cnpj"]
    assert cnpj["target_hash"] != "h00"                                       # h00's report holds a CNPJ
    assert qs == Q.build_questions(case_set, 1, corpus, CFG, None, {f"h{i:02d}" for i in range(10)}, seed=3)


def test_unanswerable_outside_process_is_outside_every_set_and_every_text():
    corpus = _corpus()
    corpus.loc[3, "acordao_voto_relator"] = "cita o Processo 15414.600012/2019-22 como precedente"   # h12's number
    case_set = {"set_id": "S1", "case_hashes": [f"h{i:02d}" for i in range(10)]}
    excluded = {f"h{i:02d}" for i in range(11)}                               # h10 is a seed
    qs = Q.build_questions(case_set, 0, corpus, CFG, None, excluded, seed=1)
    outside = next(q for q in qs if q["type"] == "unanswerable")
    assert outside["subtype"] == "outside_process" and outside["outside_hash"] in {"h11", "h13"}
    count = next(q for q in qs if q["subtype"] == "count")
    assert count["gold"] == 5 and "Não provido" in count["question"] or "Provido" in count["question"]


def test_every_type_is_asked_of_the_development_set():
    qs = Q.build_questions({"set_id": "D1", "case_hashes": [f"h{i:02d}" for i in range(10)]}, 0, _corpus(), CFG, None,
                           set(), seed=1, every_type=True)
    subtypes = [q["subtype"] for q in qs]
    assert set(CFG["questions"]["templated"]) <= set(subtypes) and subtypes.count("count") == 3
    assert {"outside_process", "appellant_cnpj"} <= set(subtypes) and sum(q["type"] == "open" for q in qs) == 4


def test_spot_check_sample_covers_every_type():
    qs = [{"question_id": f"q{i}", "type": t} for i, t in enumerate(["templated"] * 24 + ["unanswerable"] * 6 + ["open"] * 12)]
    sample = Q.spot_check_sample(qs, 0.2, 1)
    types = [next(q["type"] for q in qs if q["question_id"] == i) for i in sample]
    assert len(sample) == 9 and types.count("templated") == 5 and types.count("open") == 3 and types.count("unanswerable") == 1


# ---------------------------------------------------------------- deterministic scoring
@pytest.mark.parametrize("answer,stated", [
    ("decidiram conhecer do recurso e negar-lhe provimento", ["nao_provido"]),
    ("O recurso não foi provido.", ["nao_provido"]),
    ("Resultado: Parcialmente provido", ["parcialmente_provido"]),
    ("deu-se parcial provimento ao recurso", ["parcialmente_provido"]),
    ("o Conselho deu provimento ao recurso", ["provido"]),
    ("o recurso não foi conhecido", ["nao_conhecido"]),
    ("o resultado não consta dos casos", []),
])
def test_outcomes_stated(answer, stated):
    assert S.outcomes_stated(answer) == stated


def test_outcome_is_correct_only_when_it_is_the_one_stated():
    assert S.score_outcome("O recurso foi provido.", "provido")["correct"]
    assert not S.score_outcome("O recurso não foi provido.", "provido")["correct"]
    mixed = S.score_outcome("Foi parcialmente provido; quanto à multa, negou-se provimento.", "parcialmente_provido")
    assert not mixed["correct"] and mixed["ambiguous"]


def test_names_match_whatever_the_accents_case_and_company_suffix():
    assert S.score_names("A relatora foi BEATRIZ DE MOURA CAMPOS MELLO ALMADA.", ["Beatriz de Moura Campos Mello Almada"])["correct"]
    assert S.score_names("Recorrente: Chubb Seguros Brasil S/A", ["Chubb Seguros Brasil S.A."])["correct"]
    partial = S.score_names("Recorrentes: Chubb Seguros Brasil SA e outra.", ["Chubb Seguros Brasil S.A.", "Ace Seguradora S.A."])
    assert partial["score"] == 0.5 and partial["missing"] == ["Ace Seguradora S.A."]
    assert not S.score_names("A relatora foi Beatriz Almada.", ["Beatriz de Moura Campos Mello Almada"])["correct"]


def test_session_dates():
    gold = S.parse_session_date("26 e 27\xa0 de \xa0agosto", 2021)
    assert gold == {"days": [26, 27], "month": 8, "year": 2021}
    assert S.parse_session_date("16 de julho à 03 de agosto - ", 2020) is None
    assert S.score_session_date("A sessão ocorreu nos dias 26 e 27 de agosto de 2021.", gold)["correct"]
    assert S.score_session_date("Sessão de 2021, realizada em 26 e 27 de agosto.", gold)["correct"]
    assert S.score_session_date("Em 26/08/2021 e 27/08/2021.", gold)["correct"]
    assert S.score_session_date("Em 26 de agosto de 2021.", gold)["score"] == 0.5
    assert S.score_session_date("Em 26 e 27 de agosto de 2020.", gold)["score"] == 0.0
    assert S.score_session_date("Em 26 e 27 de julho de 2021.", gold)["score"] == 0.0


def test_penalty_types_and_amounts_within_tolerance():
    gold = S.parse_penalty("Multa de R$8.000,00 (oito mil reais)")
    assert gold == {"types": ["multa"], "amounts": [8000.0]}
    assert S.score_penalty("Foi aplicada multa de R$ 8.000,00.", gold)["correct"]
    assert S.score_penalty("Foi aplicada multa de 8.000,00 reais.", gold)["correct"]
    assert S.score_penalty("Foi aplicada multa de R$ 9.000,00.", gold)["score"] == 0.5
    assert S.score_penalty("Foi aplicada advertência.", gold)["score"] == 0.0
    assert S.score_penalty("A penalidade foi de advertência.", {"types": ["advertencia"], "amounts": []})["correct"]


def test_legal_basis_by_provisions():
    gold = ["CIRC-SUSEP364/2008:art2", "DL73/1966:art88"]
    answer = "A base legal é o artigo 2º da Circular SUSEP nº 364/2008 combinado com o artigo 88 do Decreto-Lei nº 73/66."
    assert S.score_provisions(answer, gold)["correct"]
    spelled = "É o Artigo 88 do Decreto-Lei número 73/1966, combinado com o Artigo 2º da Circular SUSEP número 364/2008."
    assert S.score_provisions(spelled, gold)["correct"]
    half = S.score_provisions("A base legal é o art. 2º da Circular SUSEP nº 364/2008.", gold)
    assert half["score"] == 0.5 and half["missing"] == ["DL73/1966:art88"]


@pytest.mark.parametrize("answer,gold,correct", [
    ("Entre os dez casos apresentados, seis recursos tiveram o resultado não provido: Processo 15414.004937 2015-54 e Recurso 7354.", 6, True),
    ("Dos 10 recursos, 7 tiveram esse resultado.", 7, True),
    ("Foram cinco recursos julgados em 2020, na sessão de 29 e 30 de julho de 2020.", 5, True),
    ("Todos os recursos apresentados tiveram esse relator.", 10, True),
    ("Nenhum dos recursos teve esse resultado.", 0, True),
    ("Três recursos tiveram esse resultado.", 4, False),
    ("Uma análise dos casos mostra que quatro recursos tiveram esse resultado.", 4, True),
    ("Foram 4 recursos não providos e 6 providos.", 4, False),
])
def test_counts(answer, gold, correct):
    assert S.score_count(answer, gold)["correct"] is correct


def test_a_count_next_to_another_number_is_marked_ambiguous():
    result = S.score_count("Foram 4 recursos não providos e 6 providos.", 4)
    assert result["ambiguous"] and result["stated"] == [4, 6]


@pytest.mark.parametrize("answer", [
    "Informa-se que não constam informações sobre o Processo 15414.600337/2020-89 nos casos apresentados.",
    "Não há informação disponível sobre o CNPJ da recorrente.",
    "O número do CNPJ não é mencionado nos documentos fornecidos.",
    "Não foi possível localizar esse dado.",
    "Nenhum dos casos apresentados corresponde a esse processo.",
])
def test_abstention_patterns(answer):
    assert S.score_unanswerable(answer, "outside_process", PATTERNS)["status"] == "abstained"


def test_unanswerable_fabrication_and_unclassified():
    fabricated = S.score_unanswerable("O CNPJ é 12.345.678/0001-90, embora não conste do relatório.", "appellant_cnpj", PATTERNS)
    assert fabricated["status"] == "fabricated" and fabricated["correct"] is False
    unclear = S.score_unanswerable("O recurso foi provido por unanimidade.", "outside_process", PATTERNS)
    assert unclear["status"] == "unclassified" and unclear["correct"] is None


def test_citation_check_tells_supplied_from_unknown_numbers():
    supplied = {"154146067642016" + "94", "R7354"}
    answer = ("No Processo 15414.606764/2016-94 e no Recurso 7354 houve advertência; o Processo 15414.600337/2020-89 "
              "foi citado, assim como o Processo 15414.111111/2011-11 e o Processo 15414.222222/2012-22.")
    check = S.citation_check(answer, supplied, "E o Processo 15414.600337/2020-89?", "precedente: Processo 15414.111111/2011-11")
    assert (check["supplied"], check["in_question"], check["in_context"], check["unknown"]) == (2, 1, 1, 1)
    assert check["unknown_numbers"] == ["15414222222201222"] and not check["all_supplied"] and not check["none_unknown"]
    assert S.citation_check("Sem números.", supplied, "", "")["all_supplied"]


# ---------------------------------------------------------------- evidence
def test_passages_cover_every_field_with_title_and_field():
    rows = _corpus(2)
    rows.loc[0, "acordao_relatorio"] = " ".join(f"w{i}" for i in range(400))
    passages = E.build_passages(rows, size=150, overlap=30)
    first = passages[passages["acordao_hash"] == "h00"]
    assert first["field"].tolist()[0] == "dados do caso" and "Conselheiro Relator: João Souza" in first["body"].iloc[0]
    report = first[first["field"] == "relatório"]["body"].tolist()
    assert len(report) == 4 and report[0].split()[-30:] == report[1].split()[:30] and report[-1].split()[-1] == "w399"
    assert first["text"].iloc[1].startswith("[Processo 15414.600000/2019-10 | ementa] ")
    assert E.split_words("a b c", 150, 30) == ["a b c"] and E.split_words("", 150, 30) == []


def test_fusion_and_the_passages_of_named_decisions():
    passages = pd.DataFrame({"acordao_hash": ["a", "a", "b", "b", "c", "c", "c"]})
    dense = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.5])
    bm25 = np.array([7.0, 0.0, 6.0, 1.0, 5.0, 4.0, 3.0])
    assert np.argsort(-E.rrf([dense, bm25])).tolist() == [0, 2, 4, 5, 6, 3, 1]
    assert E.select_evidence(dense, bm25, passages, [], top=2) == [0, 2]                  # no decision named: whole set
    assert E.select_evidence(dense, bm25, passages, ["c"], named={1: 2}) == [4, 5]        # only the named decision
    assert E.select_evidence(dense, bm25, passages, ["b", "c"], named={2: 2}) == [2, 3, 4, 5]
    assert E.select_evidence(dense, bm25, passages, ["a", "b", "c"], named={3: 2}, limit=4) == [0, 2, 4, 5]
    assert E.select_evidence(dense, bm25, passages, ["a", "b", "c"], named={1: 4}, beyond=1) == [0, 2, 4]
    assert E.most_likely_decision([2, 0], passages, []) == "b" and E.most_likely_decision([2, 0], passages, ["c"]) == "c"


def test_claims_name_decisions_by_their_numbers():
    rows = _corpus(3)
    assert E.cited_hashes(["Processo 15414.600001/2019-11"], "a multa foi mantida", rows) == ["h01"]
    assert E.cited_hashes([], "No Processo 15414.600002 2019-12 houve advertência", rows) == ["h02"]
    assert E.cited_hashes([], "sem número", rows) == []


def test_nli_claim_label():
    assert N.claim_label(np.array([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1]])) == "supported"
    assert N.claim_label(np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1]])) == "contradicted"
    assert N.claim_label(np.array([[0.4, 0.4, 0.2]])) == "not_found" and N.claim_label(np.zeros((0, 3))) == "not_found"


# ---------------------------------------------------------------- metrics
def _claims():
    return pd.DataFrame({
        "claim_id": [f"c{i}" for i in range(8)],
        "answer_id": ["a1"] * 3 + ["a2"] * 3 + ["a3"] * 2,
        "judge": ["supported", "supported", "not_found", "supported", "contradicted", "not_found", "supported", "supported"],
    })


def test_faithfulness_and_hallucination_rate():
    f = M.faithfulness(_claims(), "judge")
    assert (f["supported"], f["contradicted"], f["not_found"]) == (5, 1, 2)
    assert f["faithfulness"] == pytest.approx(5 / 8) and f["contradiction_rate"] == pytest.approx(1 / 8)
    assert f["hallucination_rate"] == pytest.approx(2 / 3)


def test_correction_with_the_retrieval_miss_control():
    control = pd.DataFrame({"claim_id": ["c2"], "full_text_label": ["supported"]})       # the only re-judged claim flips
    assert M.miss_rate(control)["miss_rate"] == 1.0
    out = M.corrected(_claims(), "judge", control, rate=0.5)
    # c2 becomes supported; c5 (not re-judged) counts 0.5; a2 still has a contradicted claim
    assert out["faithfulness_corrected"] == pytest.approx((5 + 1 + 0.5) / 8)
    assert out["hallucination_rate_corrected"] == pytest.approx((0 + 1 + 0) / 3)
    none = M.corrected(_claims(), "judge", pd.DataFrame({"claim_id": [], "full_text_label": []}), rate=0.0)
    assert none["hallucination_rate_corrected"] == pytest.approx(2 / 3) and none["faithfulness_corrected"] == pytest.approx(5 / 8)


def test_majority_needs_more_than_half():
    assert M.majority(["supported", "supported", "not_found"]) == "supported"
    assert M.majority(["supported", "contradicted", "not_found"]) is None
    assert M.majority(["supported", None, "supported"]) == "supported" and M.majority([None, None]) is None
    assert M.majority(["supported", "not_found"]) is None


def test_agreement_statistics():
    a = pd.Series(["supported", "supported", "not_found", "contradicted", None])
    same = M.nominal_alpha(a, a, M.LABELS)
    assert same["n"] == 4 and same["agreement"] == 1.0 and same["alpha"] == pytest.approx(1.0)
    b = pd.Series(["supported", "not_found", "not_found", "contradicted", "supported"])
    assert M.nominal_alpha(a, b, M.LABELS)["agreement"] == 0.75
    assert M.coverage_score(["covered", "partly", "missing", "covered"]) == {"points": 4, "coverage": 0.625, "coverage_strict": 0.5}


# ---------------------------------------------------------------- judge calls
class FakeModel:
    model = "fake"

    def __init__(self, texts):
        self.texts, self.keys = list(texts), []

    def complete(self, prompt, *, system=None, cache_extra=None):
        self.keys.append(cache_extra)
        return SimpleNamespace(text=self.texts.pop(0), cached=False, input_tokens=100, output_tokens=10,
                               reasoning_tokens=None, cost_usd=0.001, latency_s=0.1, credentials="primary")


def test_an_unusable_answer_is_asked_again_and_every_call_is_logged():
    model = FakeModel(["not json", '{"label": "maybe"}', '```json\n{"label": "supported", "quote": "x"}\n```'])
    asker = J.Asker("judge_a", "miss_control", model=model)
    assert asker.ask("prompt", J.validate_label, item="c1") == {"label": "supported", "quote": "x"}
    assert [c["problem"] is None for c in asker.calls] == [False, False, True]
    assert [k.get("format_retry") for k in model.keys] == [None, 1, 2] and model.keys[0]["step"] == "miss_control"
    summary = J.calls_summary(asker.calls)
    assert summary["calls"] == 3 and summary["unusable_answers"] == 2 and summary["max_input_tokens"] == 100
    assert summary["served_by"] == {"primary": 3} and summary["cost_usd"] == pytest.approx(0.003)
    assert J.Asker("judge_a", "x", model=FakeModel(["no"] * 3)).ask("p", J.validate_label, item="c") is None


def test_a_prompt_above_the_token_limit_is_refused_before_any_call():
    model = FakeModel([])
    asker = J.Asker("judge_b", "verification", model=model)
    with pytest.raises(J.PromptTooLong):
        asker.ask("x" * int(CFG["max_prompt_tokens"] * CFG["chars_per_token"] + 10), J.validate_label, item="big")
    assert model.keys == [] and asker.calls == []


def test_answer_checks():
    assert J.validate_verdicts({"verdicts": [{"id": 2, "label": "not_found"}, {"id": 1, "label": "supported", "passage": 3}]}, 2)[0] == \
        {"verdicts": [{"label": "supported", "passage": 3}, {"label": "not_found", "passage": None}]}
    assert J.validate_verdicts({"verdicts": [{"id": 1, "label": "supported"}]}, 2)[0] is None
    assert J.validate_verdicts({"verdicts": [{"id": 1, "label": "true"}]}, 1)[0] is None
    assert J.validate_claims({"claims": []})[0] == {"claims": []}
    assert J.validate_claims({"claims": [{"claim": " x ", "processes": None}]})[0] == {"claims": [{"claim": "x", "processes": []}]}
    assert J.validate_claims({"claims": [{"claim": ""}]})[0] is None
    assert J.validate_key_points({"key_points": ["a", "b"]}, 3, 6)[0] is None
    assert J.validate_key_points({"key_points": ["a", "b", "c"]}, 3, 6)[0] == {"key_points": ["a", "b", "c"]}
    assert J.validate_coverage({"points": [{"id": 1, "coverage": "partly"}]}, 1)[0] == {"coverage": ["partly"]}
    assert J.validate_coverage({"points": [{"id": 1, "coverage": "yes"}]}, 1)[0] is None
    assert J.validate_similarity({"similarity": 3, "reason": "r"})[0]["similarity"] == 3
    assert J.validate_similarity({"similarity": 6})[0] is None and J.validate_similarity({"similarity": 2.5})[0] is None


def test_long_texts_are_cut_in_the_middle_and_prompts_stay_under_the_limit():
    assert J.cut_middle("a" * 10, 4, 4) == "aaaa\n[...]\naaaa" and J.cut_middle("abc", 4, 4) == "abc"
    row = {**_corpus(1).iloc[0].to_dict(), "acordao_relatorio": "r" * 200_000, "acordao_voto_relator": float("nan")}
    spec = CFG["material"]
    text = J.decision_text(row, spec["report_chars"], spec["vote_chars"])
    prompt = J.KEY_POINTS_PROMPT.format(decision=text, **spec["key_points"])
    assert J.estimated_tokens(prompt) < CFG["max_prompt_tokens"] and "VOTO DO RELATOR:\n" in text
    part = J.FULL_TEXT_PROMPT.format(claim="c" * 600, decision="d" * CFG["faithfulness"]["miss_control"]["part_chars"],
                                     part_note="", scope_note=J.SEVERAL_DECISIONS_NOTE)
    assert J.estimated_tokens(part) < CFG["max_prompt_tokens"]


# ---------------------------------------------------------------- private data stays local
def test_a_process_number_is_not_a_cnpj():
    assert not S.has_cnpj("No Processo 15414.600000/2019-10 não consta o dado.")
    assert S.has_cnpj("CNPJ 12.345.678/0001-90") and S.has_cnpj("inscrita sob o nº 12345678000190".replace("0001", "/0001-"))
    answer = "No Processo 15414.629600/2019-88 não consta o número de inscrição no CNPJ da recorrente."
    assert S.score_unanswerable(answer, "appellant_cnpj", PATTERNS)["status"] == "abstained"


def test_control_verdict_of_a_claim_about_several_decisions():
    import importlib.util
    spec = importlib.util.spec_from_file_location("a3_faithfulness", ROOT / "scripts/64_a3_faithfulness.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.combine_control(["supported", "supported"]) == "supported"
    assert script.combine_control(["supported", "not_found"]) == "not_found"
    assert script.combine_control(["supported", "contradicted", "not_found"]) == "contradicted"
    assert script.combine_control(["supported", None]) is None
    claims = pd.DataFrame({"answer_id": ["a"] * 7, "claim_id": [f"c{i}" for i in range(7)], "claim": ["x" * 400] * 7,
                           "evidence": [["p" * 950] * 4] * 7})
    groups = script.batches(claims, 5)
    assert [len(g) for g in groups] == [3, 3, 1]                              # four claims would pass the token budget
    assert all(J.estimated_tokens(J.VERIFY_PROMPT.format(items="\n\n".join(
        script.item_text(i, c["claim"], c["evidence"]) for i, c in enumerate(g, 1)))) <= script.BATCH_TOKEN_BUDGET for g in groups)


def test_titles_in_questions():
    assert Q.clean_title("Processo 15414.003279/2014-01.\t") == "Processo 15414.003279/2014-01"
    rows = _corpus(4)
    rows.loc[0, "acordao_titulo"] = "RETIFICAÇÃO 15414.620669 2017-84"
    rows.loc[1, "acordao_titulo"] = "Recurso 7220"
    assert Q._unique_titles(rows)["acordao_hash"].tolist() == ["h01", "h02", "h03"]


# ---------------------------------------------------------------- planted controls
def test_field_controls_alternate_true_and_false_values():
    rows = _corpus(10)
    made = K.field_controls(rows, np.random.default_rng(1), 12)
    assert len(made) == 12 and [c["truth"] for c in made] == [True, False] * 6
    by_hash = rows.set_index("acordao_hash")
    for c in made:
        row, kind = by_hash.loc[c["acordao_hash"]], c["kind"].split("/")[1]
        own = {"rapporteur": row["acordao_relator"][0], "appellant": row["acordao_recorrente"][0],
               "outcome": f'"{row["acordao_resultado"]}"', "session_date": f"25 e 26 de agosto de {row['ano_sessoes']}"}[kind]
        assert (own in c["claim"]) is c["truth"] and "\t" not in c["claim"]


def test_sentence_controls_copy_or_change_one_figure():
    assert K.alter_figure("A multa foi fixada em R$ 12.000,00 pela autoridade.") == "A multa foi fixada em R$ 36.000,00 pela autoridade."
    assert K.alter_figure("A majoração por reincidência foi de 50% no caso.") == "A majoração por reincidência foi de 75% no caso."
    assert K.alter_figure("O prazo era de 30 dias para a entrega.") == "O prazo era de 45 dias para a entrega."
    assert K.alter_figure("Demonstrações de 31 de dezembro de 2011 enviadas com atraso.") == "Demonstrações de 31 de dezembro de 2008 enviadas com atraso."
    assert K.alter_figure("Multa de R$ 1.000,00 e de R$ 2.000,00.") is None and K.alter_figure("Sem números.") is None
    long = "A recorrente alegou que o envio dos quadros ocorreu dentro do prazo e que a multa de R$ 9.000,00 seria indevida no caso."
    rows = _corpus(2)
    rows.loc[0, "acordao_relatorio"] = f"Relatório curto. {long} Fim."
    made = K.sentence_controls(rows, np.random.default_rng(0), 1, 1)
    false = next(c for c in made if not c["truth"])
    assert "R$ 27.000,00" in false["claim"] and false["source"] == long and false["claim"].startswith("No Processo 15414.600000/2019-10: ")
    title = "Processo 15414.600000/2019-10"
    assert K.evidence_holds_source(false, [f"[{title} | relatório] Relatório curto. {long} Fim."], title)
    assert not K.evidence_holds_source(false, [f"[{title} | voto do relator] Voto"], title)
    field = {"source": "dados do caso"}
    assert K.evidence_holds_source(field, [f"[{title} | dados do caso] Resultado: Provido"], title)
    assert not K.evidence_holds_source(field, ["[Recurso 1 | dados do caso] Resultado: Provido"], title)


def test_control_scores():
    table = pd.DataFrame({"truth": [True, True, True, False, False, False],
                          "j": ["supported", "not_found", "supported", "contradicted", "supported", "not_found"]})
    assert K.score(table, "j") == {"true_controls": 3, "true_accepted": 2, "true_called_not_found": 1, "true_called_contradicted": 0,
                                   "false_controls": 3, "false_rejected": 2, "false_called_contradicted": 1, "false_accepted": 1}


# ---------------------------------------------------------------- Phase 6.1: candidate selection
def _selection():
    import importlib.util
    spec = importlib.util.spec_from_file_location("a3_selection", ROOT / "scripts/70_a3_candidate_selection.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def _candidate_rows():
    base = {"coverage": 0.70, "templated_correct": 29, "abstained": 6, "cost_usd_per_open_answer": 0.03}
    return pd.DataFrame([
        {"candidate": "baseline", "unsupported_rate": 0.20, **base},
        {"candidate": "grounded", "unsupported_rate": 0.08, **base, "coverage": 0.66},
        {"candidate": "quoted", "unsupported_rate": 0.075, **base, "coverage": 0.64, "cost_usd_per_open_answer": 0.05},
        {"candidate": "terse", "unsupported_rate": 0.01, **base, "coverage": 0.40},          # says too little
        {"candidate": "sloppy", "unsupported_rate": 0.02, **base, "templated_correct": 26},
        {"candidate": "talkative", "unsupported_rate": 0.02, **base, "abstained": 5},
    ])


def test_rule_selects_the_lowest_unsupported_rate_among_the_eligible():
    script = _selection()
    rule = load_config("a3_candidates")["rule"]
    differences = {"grounded": {"difference": -0.12, "ci_low": -0.2, "ci_high": -0.05}}
    outcome = script.apply_rule(_candidate_rows(), differences, rule)
    assert set(outcome["not_eligible"]) == {"terse", "sloppy", "talkative"}
    assert outcome["tied"] == ["grounded", "quoted"] and outcome["selected"] == "grounded"    # the tie goes to the cheaper
    assert outcome["replaces_the_baseline"] is True
    unsure = script.apply_rule(_candidate_rows(), {"grounded": {"difference": -0.12, "ci_low": -0.3, "ci_high": 0.02}}, rule)
    assert unsure["selected"] == "grounded" and unsure["replaces_the_baseline"] is False
    alone = script.apply_rule(_candidate_rows().iloc[[0, 3]], {}, rule)
    assert alone["selected"] == "baseline" and alone["replaces_the_baseline"] is False


def test_paired_difference_and_expected_unsupported_claims():
    script = _selection()
    index = pd.Index(["q1", "q2", "q3"], name="question_id")
    baseline = pd.DataFrame({"claims": [10.0, 10.0, 10.0], "unsupported": [3.0, 2.0, 4.0]}, index=index)
    candidate = pd.DataFrame({"claims": [8.0, 10.0, 6.0], "unsupported": [0.0, 1.0, 0.0]}, index=index)
    out = script.paired_difference(candidate, baseline, 2000, 1)
    assert out["difference"] == pytest.approx(1 / 24 - 9 / 30) and out["ci_high"] < 0 and out["questions"] == 3
    claims = pd.DataFrame({"claim_id": ["a", "b", "c", "d"], "question_id": ["q1", "q1", "q2", "q2"],
                           "judge_a": ["supported", "not_found", "contradicted", "not_found"],
                           "judge_b": ["supported", "supported", "contradicted", "not_found"],
                           "full_text_label": [None, "supported", None, None]})
    table = script.per_question(claims, ["judge_a", "judge_b"])
    # miss rate 1.0 (the one re-judged claim is supported): every other not-found claim counts as supported
    assert table.loc["q1", "unsupported"] == 0.0 and table.loc["q2", "unsupported"] == 1.0
    assert table.loc["q2", "contradicted"] == 1.0 and table.loc["q1", "raw_unsupported"] == 0.5


def test_candidate_tables_are_kept_apart():
    from jm_eval.chatbot import store
    try:
        store.use_candidate("quoted")
        assert store.work_dir(True).parts[-2:] == ("candidates", "quoted") and store.root_dir(True).name == "a3_dev"
        assert store.work_dir(False).name == "a3"                      # the evaluated set has no candidates
        with pytest.raises(ValueError):
            store.use_candidate("unknown")
    finally:
        store.use_candidate(None)
    assert store.work_dir(True).name == "a3_dev"


# ---------------------------------------------------------------- blind audit of the judges
def _audit():
    import importlib.util
    spec = importlib.util.spec_from_file_location("a3_audit", ROOT / "scripts/71_a3_claim_audit.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def _audit_claims(n=60):
    labels_a = ["supported"] * 40 + ["not_found"] * 20
    labels_b = ["supported"] * 30 + ["not_found"] * 10 + ["contradicted"] * 5 + ["supported"] * 15
    return pd.DataFrame({"claim_id": [f"c{i:02d}" for i in range(n)], "question_id": "q", "claim": "x",
                         "evidence": [["p"]] * n, "judge_a": labels_a, "judge_b": labels_b,
                         "judge_gemini": ["supported"] * 10 + [None] * 50})


def test_audit_sample_covers_every_pattern_of_the_judges_and_shows_no_label():
    script, claims = _audit(), _audit_claims()
    raters = ["judge_a", "judge_b", "judge_gemini"]
    picked = script.sample(claims, raters, 20, 1)
    assert len(picked) == 20 and picked["audit_id"].is_unique and picked["claim_id"].is_unique
    counts = picked["stratum"].value_counts()
    assert set(counts.index) == set(script.strata(claims, raters)) and counts.max() - counts.min() <= 1
    assert script.sample(claims, raters, 20, 1)["claim_id"].tolist() == picked["claim_id"].tolist()
    assert len(script.sample(claims, raters, 500, 1)) == 60                    # never more than there are


def test_audit_scores_each_verifier_against_the_author():
    script, claims = _audit(), _audit_claims()
    raters = ["judge_a", "judge_b"]
    key = script.sample(claims, raters, 12, 1)[["audit_id", "claim_id", "stratum"]]
    truth = key.merge(claims, on="claim_id")
    marked = pd.DataFrame({"audit_id": truth["audit_id"], "hugo_label": truth["judge_a"].map({"supported": "S", "not_found": "n"})})
    table, summary = script.analyse(marked, key, claims, raters)
    a, b = table.set_index("rater").loc["judge_a"], table.set_index("rater").loc["judge_b"]
    assert a["same_label_share"] == 1.0 and a["weighted_to_all_claims"] == 1.0 and a["supported_by_hugo_rejected_by_rater"] == 0
    assert b["same_supported_or_not_share"] < 1.0 and summary["marked"] == 12
    assert b["supported_by_hugo_rejected_by_rater"] + b["not_supported_by_hugo_accepted_by_rater"] == 12 - b["same_supported_or_not"]


def test_a_verifier_that_judged_only_some_answers_is_measured_on_those():
    claims = _claims().assign(first_runs=["supported", "not_found", "supported", None, None, None, None, None])
    f = M.faithfulness(claims, "first_runs")
    assert f["answers"] == 1 and f["claims_judged"] == 3 and f["hallucination_rate"] == 1.0
    none = M.corrected(claims, "first_runs", pd.DataFrame({"claim_id": [], "full_text_label": []}), rate=0.0)
    assert none["hallucination_rate_corrected"] == 1.0 and none["faithfulness_corrected"] == pytest.approx(2 / 3)


def test_a_decision_is_also_named_by_its_acordao_number():
    rows = _corpus(3)
    rows["acordao_acordao"] = ["ACÓRDÃO CRSNSP 7053/2021\nVistos...", "ACÓRDÃO CRSNSP 602/2024 Vistos, cita o Acórdão 7053/2021", "Acórdão"]
    assert S.acordao_numbers("No julgamento do ACÓRDÃO CRSNSP 7053/2021 e no Acórdão nº 602/2024") == ["A7053/2021", "A602/2024"]
    assert S.own_acordao_number(rows.loc[1, "acordao_acordao"]) == "A602/2024" and S.own_acordao_number("Acórdão") is None
    assert E.cited_hashes([], "No ACÓRDÃO CRSNSP 7053/2021, a multa foi mantida", rows) == ["h00"]
    assert E.cited_hashes(["602/2024"], "o Conselho manteve a multa", rows) == ["h01"]            # as the claim splitter lists it
    assert E.cited_hashes(["Processo 15414.600002/2019-12"], "No Acórdão CRSNSP 7053/2021 ...", rows) == ["h00", "h02"]
    assert E.cited_hashes([], "a Resolução CNSP nº 243/2011 foi aplicada", rows) == []
    check = S.citation_check("No ACÓRDÃO CRSNSP 7053/2021 e no Acórdão 9999/2020 houve multa.", {"A7053/2021"}, "", "")
    assert (check["supplied"], check["unknown"]) == (1, 1) and check["unknown_numbers"] == ["A9999/2020"]


def test_audit_weights_use_the_strata_of_the_sample_when_more_raters_are_scored():
    script, claims = _audit(), _audit_claims().assign(nli="supported")
    key = script.sample(claims, ["judge_a", "judge_b"], 12, 1)[["audit_id", "claim_id", "stratum"]]
    marked = pd.DataFrame({"audit_id": key["audit_id"], "hugo_label": "s"})
    table, _ = script.analyse(marked, key, claims, ["nli", "judge_a", "judge_b"], ["judge_a", "judge_b"])
    assert table.set_index("rater").loc["nli", "weighted_to_all_claims"] == 1.0
