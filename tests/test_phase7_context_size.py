"""Phase 7 (A5): contexts of other sizes and the questions asked with them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jm_eval.chatbot import context_size as CS


def toy(n: int = 12):
    """Decisions h00..h11 on a circle: h00 is the seed, similarity falls with the index."""
    angles = np.linspace(0, 1.2, n)
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    hashes = [f"h{i:02d}" for i in range(n)]
    labels = pd.DataFrame({"acordao_hash": hashes, "process_group": [f"p{i}" for i in range(n)]})
    labels.loc[1, "process_group"] = "p0"                      # h01 is the seed's own process
    return labels, vectors, hashes


def test_extension_keeps_members_and_skips_process_and_excluded():
    labels, vectors, hashes = toy()
    sets = [{"set_id": "S1", "seed_hash": "h00", "case_hashes": ["h02", "h03"]}]
    out = CS.extend_sets(sets, labels, vectors, size=5, excluded={"h00", "h02", "h03", "h05"})
    ranked = out[0]["ranked_hashes"]
    assert ranked[:2] == ["h02", "h03"]                        # Phase 6's members first, in their order
    assert ranked[2:] == ["h04", "h06", "h07"]                 # not h01 (seed's process), not h05 (excluded)
    assert out[0]["similarity"] == sorted(out[0]["similarity"], reverse=True)


def test_extension_keeps_sets_apart():
    labels, vectors, _ = toy()
    sets = [{"set_id": "S1", "seed_hash": "h00", "case_hashes": ["h02"]},
            {"set_id": "S2", "seed_hash": "h00", "case_hashes": ["h03"]}]
    out = CS.extend_sets(sets, labels, vectors, size=3, excluded={"h00", "h02", "h03"})
    assert not set(out[0]["ranked_hashes"]) & set(out[1]["ranked_hashes"])


def test_extension_refuses_an_unordered_ranking():
    labels, vectors, _ = toy()
    sets = [{"set_id": "S1", "seed_hash": "h00", "case_hashes": ["h08"]}]   # a member less similar than free decisions
    with pytest.raises(ValueError):
        CS.extend_sets(sets, labels, vectors, size=3, excluded={"h00", "h08"})


def corpus():
    return pd.DataFrame({
        "acordao_hash": ["a", "b", "c", "d"],
        "acordao_titulo": ["Processo 15414.000001/2010-01", "Processo 15414.000002/2010-02",
                           "Processo 15414.000003/2010-03", "Processo 15414.000003/2010-03"],
        "acordao_resultado": ["Provido", "Negado provimento", "Negado provimento", "Negado provimento"],
        "acordao_relator": [["X"], ["Y"], ["X"], ["X"]],
        "ano_sessoes": [2020, 2021, 2021, 2022]})


QUESTIONS = [
    {"question_id": "S1-T1", "set_id": "S1", "type": "templated", "subtype": "outcome", "target_hash": "b",
     "case_hashes": ["a", "b"], "gold": "Negado provimento"},
    {"question_id": "S1-T2", "set_id": "S1", "type": "templated", "subtype": "rapporteur", "target_hash": "c",
     "case_hashes": ["a", "b"], "gold": ["X"]},
    {"question_id": "S1-T3", "set_id": "S1", "type": "templated", "subtype": "count", "variant": "outcome",
     "value": "Negado provimento", "target_hash": None, "case_hashes": ["a", "b"], "gold": 1},
    {"question_id": "S1-U1", "set_id": "S1", "type": "unanswerable", "subtype": "outside_process", "case_hashes": ["a", "b"]},
    {"question_id": "S1-O1", "set_id": "S1", "type": "open", "subtype": "compare_outcomes", "case_hashes": ["a", "b"],
     "reference_points": ["p"]},
]


def test_questions_of_a_smaller_context():
    asked, dropped = CS.setting_questions(QUESTIONS, {"S1": ["a", "b", "c", "d"]}, 1, corpus(), ["templated", "open"])
    assert {d["question_id"] for d in dropped} == {"S1-T1", "S1-T2", "S1-O1"}   # targets outside; one group only
    count = next(q for q in asked if q["question_id"] == "S1-T3")
    assert count["gold"] == 0 and count["phase6_gold"] == 1
    assert all(q["type"] != "unanswerable" for q in asked)


def test_questions_of_phase6_context_are_unchanged():
    asked, dropped = CS.setting_questions(QUESTIONS, {"S1": ["a", "b", "c", "d"]}, 2, corpus(), ["templated", "open"])
    assert {d["question_id"] for d in dropped} == {"S1-T2"}
    open_q = next(q for q in asked if q["type"] == "open")
    assert open_q["reference_points"] == ["p"] and open_q["case_hashes"] == ["a", "b"]


def test_questions_of_a_larger_context():
    asked, dropped = CS.setting_questions(QUESTIONS, {"S1": ["a", "b", "c", "d"]}, 4, corpus(), ["templated", "open"])
    # c and d share a title, so the rapporteur question cannot name one decision any more
    assert dropped == [{"question_id": "S1-T2", "reason": "the title asked about does not identify one decision of the context"}]
    assert next(q for q in asked if q["question_id"] == "S1-T3")["gold"] == 3
    assert "reference_points" not in next(q for q in asked if q["type"] == "open")


# Counts above ten and amounts in words (found on the Phase 7 answers; the Phase 6 scores are unchanged)
@pytest.mark.parametrize("answer, gold, set_size, correct", [
    ("Entre os casos apresentados, quatorze recursos tiveram o resultado de não provimento.", 14, 30, True),
    ("Entre os casos apresentados, vinte recursos foram julgados em sessões do ano de 2021.", 20, 30, True),
    ("Foram vinte e dois recursos não providos.", 22, 30, True),
    ("Foram vinte e dois recursos não providos.", 20, 30, False),
    ("Entre os cinco casos apresentados, dois recursos tiveram o resultado de não provimento. "
     "No outro, a multa de R$ 10.000,00 (dez mil reais) foi convolada em advertência.", 2, 5, True),
    ("Dois foram providos, e a multa caiu para dez mil reais.", 2, 5, True),
    ("Catorze recursos, dos trinta apresentados, não foram providos.", 14, 30, True),
])
def test_counts_above_ten_and_amounts_in_words(answer, gold, set_size, correct):
    from jm_eval.chatbot import scoring as S
    assert S.score_count(answer, gold, set_size)["correct"] is correct
