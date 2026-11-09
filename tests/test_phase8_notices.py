"""Phase 8 (A4), task 1: what may be a test notice, outvoted rapporteurs, the prompts and the leakage measures."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from jm_eval.drafting import notices as N
from jm_eval.io import resolve


@pytest.mark.parametrize("text", [
    "Vencidos os Conselheiros Neival Rodrigues Freitas (Relator), José Antônio Maia Piñeiro, que votaram para dar parcial provimento.",
    "Vencidos a Relatora e os Conselheiros Washington Luis Bezerra da Silva, que votaram pelo provimento do recurso.",
    "por maioria. Vencida a Conselheira Carmen Diva Beltrão Monteiro (Relatora), que votou pela manutenção.",
    "Também vencidos o Relator e os Conselheiros Neival Rodrigues Freitas e Juliana Paes.",
    "Vencida a fundamentação do voto da Relatora, no sentido da atipicidade da conduta.",
])
def test_outvoted_rapporteur_found(text):
    assert N.rapporteur_outvoted("ACÓRDÃO CRSNSP 1/2021 ... nos termos do voto do Conselheiro X. " + text)


@pytest.mark.parametrize("text", [
    "por maioria, negar-lhe provimento, nos termos do voto da Relatora. Vencido o Conselheiro José Carlos Gomes Mota, que votou para dar parcial provimento.",
    "Vencidos os Conselheiros Thompson da Gama Moret Santos e Carmen Diva Beltrão Monteiro, que divergiram do Relator unicamente para manter atenuantes.",
    "por unanimidade, dar-lhe parcial provimento, nos termos do Voto do Relator.",
    None,
])
def test_outvoted_rapporteur_not_found(text):
    assert not N.rapporteur_outvoted(text)


def test_excluded_decisions_cover_phases_6_and_7():
    excluded = N.excluded_decisions()
    assert len(excluded) == 225
    a5 = json.loads(sorted(resolve("data/v2/a5").glob("case_sets*.json"))[-1].read_text(encoding="utf-8"))["sets"]
    assert {h for s in a5 for h in s["ranked_hashes"]} <= excluded
    a3 = json.loads(resolve("data/v2/a3/case_sets_2.json").read_text(encoding="utf-8"))["sets"]
    assert {h for s in a3 if s["development"] for h in s["case_hashes"]} <= excluded     # D1-D3 too


def toy():
    rows = []
    for i, (key, outcome, group, vote) in enumerate([
        ("K1", "Provido", "g1", "nos termos do voto do Relator."),
        ("K1", "Provido", "g2", "Vencido o Relator, que votou por negar."),      # outvoted
        ("K1", "Não provido", "g1", "nos termos do voto do Relator."),           # same process as h00
        ("K1", "Não provido", "g3", "nos termos do voto do Relator."),
        ("K2", "Provido", "g4", "nos termos do voto do Relator."),
        ("K2", "Não provido", "g5", "nos termos do voto do Relator."),
        ("K2", "Não provido", "g6", "nos termos do voto do Relator."),           # excluded below
    ]):
        rows.append({"acordao_hash": f"h{i:02d}", "acordao_titulo": f"T{i}", "acordao_resultado": outcome,
                     "acordao_relatorio": "x" * 20, "acordao_voto_relator": "y" * 20, "acordao_acordao": vote,
                     "base_key": key, "process_group": group})
    df = pd.DataFrame(rows)
    spec = {"selection": {"n_strata": 2, "outcomes": ["Provido", "Não provido"], "min_report_chars": 10,
                          "min_vote_chars": 10, "seed": 1},
            "outvoted": {"exclude_from_test_notices": True}}
    return df, df[["acordao_hash", "base_key", "process_group"]], spec


def test_selection_skips_excluded_outvoted_and_used_processes():
    corpus, labels, spec = toy()
    for seed in range(20):
        spec["selection"]["seed"] = seed
        chosen = N.select(corpus, labels, {"h06"}, spec)
        assert len(chosen) == 4
        assert not {"h01", "h06"} & set(chosen["acordao_hash"])
        assert chosen["process_group"].is_unique
        assert set(chosen["acordao_hash"]) == {"h00", "h03", "h04", "h05"}


def test_a_replaced_decision_is_redrawn_in_its_cell_and_nothing_else_moves():
    corpus, labels, spec = toy()
    corpus.loc[7] = {**corpus.loc[3].to_dict(), "acordao_hash": "h07", "process_group": "g7"}   # a second choice in K1 / Não provido
    labels = corpus[["acordao_hash", "base_key", "process_group"]]
    before = N.select(corpus, labels, set(), spec)
    drawn = before.loc[1, "acordao_hash"]
    spec["selection"]["replaced"] = {drawn: "unusable"}
    after = N.select(corpus, labels, set(), spec)
    assert after.loc[1, "acordao_hash"] != drawn and after.loc[1, "replaces"] == drawn
    assert after.drop(index=1)["acordao_hash"].tolist() == before.drop(index=1)["acordao_hash"].tolist()


def test_selection_fails_on_an_empty_cell():
    corpus, labels, spec = toy()
    with pytest.raises(ValueError, match="no eligible decision"):
        N.select(corpus, labels, {"h04"}, spec)


ROW = {"acordao_titulo": "Processo 1", "acordao_resultado": "Parcialmente provido", "acordao_base": "Art. 5 da Circular 74",
       "acordao_relatorio": "R" * 30000, "acordao_voto_relator": "V" * 30000,
       "acordao_acordao": "por maioria, reduzir a multa. Vencido o Relator, que votou por negar provimento."}
SPEC = {"material": {"arguments_with_vote": {"report_chars": {"head": 4500, "tail": 3500}, "vote_chars": {"head": 4000, "tail": 3000}},
                     "notice_report_chars": {"head": 8000, "tail": 5000}, "arguments_report_chars": {"head": 7000, "tail": 5000},
                     "accepted_vote_chars": {"head": 5000, "tail": 5000}, "dispositive_chars": 3000,
                     "arguments": {"min": 2, "max": 10}}}


def test_prompts_that_read_the_decision_get_the_dispositive():
    arguments = N.arguments_prompt(ROW, SPEC)
    accepted = N.accepted_prompt(ROW, ["A recorrente alega que a multa é excessiva.", "Alega prescrição."], SPEC)
    for prompt in (arguments, accepted):
        assert ROW["acordao_acordao"] in prompt and "OUTCOME: Parcialmente provido" in prompt
    assert "prevails over the vote" in accepted and "1. A recorrente alega" in accepted
    notice = N.notice_prompt(ROW, SPEC)
    assert ROW["acordao_acordao"] not in notice and "V" * 50 not in notice     # the notice reads the report only
    assert "\n[...]\n" in notice and notice.count("R") < 13100      # the report is cut to head + tail


def test_notice_prompt_names_arguments_to_leave_out():
    prompt = N.notice_prompt(ROW, SPEC, avoid=["A recorrente alega boa-fé {sic}."])
    assert "Leave them out entirely" in prompt and "boa-fé {sic}" in prompt
    assert "Leave them out" not in N.notice_prompt(ROW, SPEC, avoid=[])


def test_answer_checks():
    assert N.validate_accepted({"accepted": [2, 1], "granted": "multa reduzida"}, 3) == ({"accepted": [1, 2], "granted": "multa reduzida"}, None)
    assert N.validate_accepted({"accepted": [4], "granted": "x"}, 3)[0] is None
    assert N.validate_accepted({"accepted": [], "granted": "x"}, 3)[0] == {"accepted": [], "granted": "x"}
    assert N.validate_arguments({"arguments": ["a", "b"]}, 2, 10)[0] == {"arguments": ["a", "b"]}
    assert N.validate_arguments({"arguments": ["a"]}, 2, 10)[0] is None
    assert N.validate_notice({"notice": "curto"})[0] is None
    assert N.validate_notice({"notice": "Dos Fatos Apurados\n" + "x " * 200 + "\nDos Dispositivos Legais"})[0] is None
    assert N.validate_notice({"notice": "Dos Fatos Apurados\n" + "x " * 200 + "\nDos Dispositivos Legais\nDa Decisão da SUSEP"})[0] is not None


def test_leakage_measures():
    notice = "Dos Fatos Apurados. A seguradora deixou de apresentar a proposta de seguro no prazo fixado pela SUSEP."
    assert N.ngram_containment("A recorrente alega que deixou de apresentar a proposta de seguro no prazo", notice, 5) > 0.5
    assert N.ngram_containment("A recorrente invoca a prescrição quinquenal da pretensão punitiva", notice, 5) == 0
    assert N.ngram_containment("curto", notice, 5) == 0
    terms = ["provid[oa]s?", "recurso", "relator[a]?", "aleg\\w+", "acordao"]
    assert N.forbidden_terms("O Recurso foi PROVIDO, nos termos do Acórdão; a sociedade alegou boa-fé.", terms) == \
        ["acordao", "alegou", "provido", "recurso"]
    assert N.forbidden_terms("Os recursos financeiros foram aplicados.", terms) == []
    dispositive = "dar-lhe parcial provimento, para reduzir a multa para R$25.720,00 (vinte e cinco mil)"
    report = "a SUSEP aplicou multa de R$ 38.000,00 ao final"
    assert N.outcome_amounts("Multa aplicada: R$ 38.000,00.", dispositive, report) == []
    assert N.outcome_amounts("Multa de R$ 38.000,00, reduzida a R$ 25.720,00.", dispositive, report) == ["25720.00"]
    assert N.amounts("R$ 1.500 e R$10,50") == {"1500.00", "10.50"}
    assert N.sentences("Primeira frase com palavras. Curta.\n• Item da lista com quatro palavras") == \
        ["Primeira frase com palavras.", "Item da lista com quatro palavras"]


def test_penalty_field_amount():
    assert N.penalty_field_amount("Penalidade Original: Multa no valor de R$34.000,00. Penalidade final: R$15.000,00") == "34000.00"
    assert N.penalty_field_amount("Advertência") is None
    assert "final first-instance decision" in N.notice_prompt(ROW, SPEC)


def test_amount_rule_version_2_catches_an_outcome_figure_whatever_its_source():
    dispositive = "dar-lhe parcial provimento, para reduzir o valor da pena de multa para R$ 20.000,00 (vinte mil reais)"
    report = "a multa base majorada ao dobro, totalizando R$ 20.000,00 ... valor final de R$ 50.000,00"
    notice = "Penalidade de R$ 50.000,00. A multa base foi majorada ao dobro, totalizando R$ 20.000,00."
    assert N.outcome_amounts(notice, dispositive, report) == ["20000.00"]          # version 1 missed it (in the report)
    kept = "negar-lhe provimento, para manter a pena de multa aplicada pela Autarquia, no valor de R$ 42.900,00"
    assert N.council_amounts(kept) == set()
    assert N.outcome_amounts("Multa de R$ 42.900,00.", kept, "multa de R$ 42.900,00") == []
    assert N.council_amounts("mantendo a multa de R$ 10.000,00 e reduzindo a outra para R$ 5.000,00") == {"5000.00"}


def test_review_prompts_carry_the_comment_and_the_rejected_version():
    prompt = N.notice_prompt(ROW, SPEC, [], {"previous": "Dos Fatos\nconstatou-se a pendência {x}", "comment": "Tirar a versão da seguradora."})
    assert "REVIEWER'S COMMENT: Tirar a versão da seguradora." in prompt and "    constatou-se a pendência {x}" in prompt
    args = N.arguments_prompt(ROW, SPEC, {"previous": ["Alega {a}.", "Alega b."], "comment": "Falta a prescrição {sic}."})
    assert "Falta a prescrição {sic}." in args and "1. Alega {a}." in args
    assert args.index("REJECTED LIST") < args.index("Return JSON only") < args.index("OUTCOME:")


def test_review_marks():
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("s81", resolve("scripts/81_a4_notice_rewrite.py"))
    s81 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(s81)
    review = N.settings()["review"]
    assert s81.mark("1", review) is True and s81.mark(1.0, review) is True and s81.mark("Sim", review) is True
    assert s81.mark("0", review) is False and s81.mark("não", review) is False
    assert s81.mark("partly", review) is False                  # rewritten from the comment
    assert s81.clean_comment("[LLM reviewer] Fix the notice.", review) == "Fix the notice."
    assert s81.mark("", review) is None and s81.mark(float("nan"), review) is None
    with pytest.raises(ValueError):
        s81.mark("maybe", review)


def test_arguments_prompt_with_vote():
    plain = N.arguments_prompt(ROW, SPEC)
    voted = N.arguments_prompt(ROW, SPEC, {"previous": ["Alega a."], "comment": "Falta {x}."}, with_vote=True)
    assert "VOTE:" not in plain and "V" * 50 not in plain
    assert voted.rstrip().endswith("V" * 20) and "\nVOTE:\n" in voted and "prevails over the vote" in voted
    assert voted.count("R") < plain.count("R")                       # the report is cut shorter to make room
    assert ROW["acordao_acordao"] in voted and "Falta {x}." in voted


def test_hand_edits_and_the_frozen_notices():
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("s82", resolve("scripts/82_a4_freeze_notices.py"))
    s82 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(s82)
    notices = [{"notice_id": "N01", "notice": "A multa. B."}, {"notice_id": "N02", "notice": "C."}]
    out = s82.apply_edits(notices, [{"notice": "N01", "old": "A multa.", "new": "A multa de R$ 1,00."}])
    assert out[0]["notice"] == "A multa de R$ 1,00. B." and out[0]["hand_edits"] and out[1] == notices[1]
    with pytest.raises(ValueError, match="not once"):
        s82.apply_edits(notices, [{"notice": "N02", "old": "X", "new": "Y"}])
    frozen = N.load_frozen()
    assert len(frozen) == 15 and all(r["accepted_by_hugo"] for r in frozen)
    by_id = {r["notice_id"]: r for r in frozen}
    assert "inscrição do veículo na BIN" in by_id["N05"]["notice"] and "pena base foi fixada em R$ 10.000,00" in by_id["N14"]["notice"]
    assert "totalizando R$ 20.000,00" not in by_id["N14"]["notice"]
    assert not {r["acordao_hash"] for r in frozen} & N.excluded_decisions()
    corpus = pd.read_parquet(resolve(N.settings()["corpus"]), columns=["acordao_hash", "acordao_acordao"]).set_index("acordao_hash")
    assert not any(N.rapporteur_outvoted(corpus.loc[r["acordao_hash"], "acordao_acordao"]) for r in frozen)


def _s83():
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("s83", resolve("scripts/83_a4_drafts.py"))
    s83 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(s83)
    return s83


def test_precedents_drop_the_source_and_its_process():
    s83 = _s83()
    notice = {"acordao_hash": "src"}
    groups = {"src": 1, "same": 1, "a": 2, "b": 3}
    out = s83.precedents_for(notice, {"filtered_data_ids": ["src", "a", "same", "b"], "similarity": [0.9, 0.8, 0.7, 0.6]}, groups)
    assert out["precedents"] == ["a", "b"] and out["removed"] == ["src", "same"] and out["source_rank"] == 1
    assert s83.precedents_for(notice, {"filtered_data_ids": ["a", "b"]}, groups)["source_rank"] is None


def test_pilot_takes_one_notice_per_legal_basis_and_costs_from_the_record():
    s83 = _s83()
    frozen = N.load_frozen()
    pilot = s83.pilot_notices(frozen, 20261018)
    assert len(pilot) == 5 and len({r["base_key"] for r in frozen if r["notice_id"] in pilot}) == 5
    record = {"llm_calls": [{"model": "gemini-3.7-flash", "prompt_token_count": 1_000_000, "candidates_token_count": 100_000,
                             "thoughts_token_count": 100_000}, {"cache_hit": True, "model": "gemini-3.7-flash"}]}
    assert abs(s83.record_cost(record) - (0.75 + 0.2 * 3.75)) < 1e-9


def test_hand_edits_replace_arguments_and_skip_edits_already_applied():
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("s82b", resolve("scripts/82_a4_freeze_notices.py"))
    s82 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(s82)
    notices = [{"notice_id": "N10", "notice": "Autuada: X. Multa de R$ 1,00.", "arguments": ["a", "b"], "accepted": [2]}]
    edits = [{"notice": "N10", "old": "Autuada: X.", "new": "Autuados: X e Y."},
             {"notice": "N10", "arguments": ["a", "b", "c"], "accepted": [1, 3]}]
    once = s82.apply_edits(notices, edits)
    assert once[0]["notice"] == "Autuados: X e Y. Multa de R$ 1,00." and once[0]["arguments"] == ["a", "b", "c"] and once[0]["accepted"] == [1, 3]
    twice = s82.apply_edits(once, edits)                         # a later freeze carries the edits over
    assert twice[0]["notice"] == once[0]["notice"] and len(twice[0]["hand_edits"]) == 2
    with pytest.raises(ValueError, match="outside"):
        s82.apply_edits(notices, [{"notice": "N10", "accepted": [3]}])


def test_network_failures_are_told_apart_from_system_failures():
    s83 = _s83()
    assert s83.network_failure(["Erro durante a geração: HTTPSConnectionPool(host='oauth2.googleapis.com', port=443): Max retries "
                                "exceeded with url: /token (Caused by NameResolutionError(...))"])
    assert not s83.network_failure(["A defesa final não foi gerada pelo modelo (motivo: RECITATION). Tente novamente."])
    assert not s83.network_failure(["Gemini returned no text (finish reason: FinishReason.RECITATION)"]) and not s83.network_failure(None)


def test_an_expired_fallback_login_is_an_environment_failure():
    s83 = _s83()
    assert s83.network_failure(["Erro durante a geração: Reauthentication is needed. Please run `gcloud auth application-default login`"])
