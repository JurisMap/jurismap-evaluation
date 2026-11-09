import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from jm_eval.retrieval import review as R

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("label_review", ROOT / "scripts/49_label_review.py")
script = importlib.util.module_from_spec(spec)
sys.modules["label_review"] = script
spec.loader.exec_module(script)
labeller = script._labeller()
TAXONOMY = labeller.load_taxonomy()


def test_prompts_hold_the_fields_and_no_category_label():
    prompt = R.base_prompt("Art. 88 do Decreto-Lei nº 73/1966", ["DL73/1966:art88"], "Multa de R$ 10.000,00")
    assert "DL73/1966:art88" in prompt and "Multa de R$ 10.000,00" in prompt and '"advertência"' in prompt
    assert "(none)" in R.base_prompt(None, [], None)
    long = R.infringed_prompt("CIRC-SUSEP364/2008:art2", "ementa", "r" * 50000, "v" * 50000)
    assert "CIRC-SUSEP364/2008:art2" in long and len(long) < 12000 + 6000 + 3000          # texts are cut


def test_answers_are_validated():
    ok, problem = R.validate_base({"base_verdict": "wrong", "base_missing": ["art. 10"], "base_spurious": [],
                                   "penalty_types": ["Advertência", "multa", "multa"]})
    assert problem is None and ok["penalty_types"] == ["multa", "advertencia"] and ok["base_missing"] == ["art. 10"]
    assert R.validate_base({"base_verdict": "maybe", "penalty_types": []})[0] is None
    assert R.validate_base({"base_verdict": "correct", "penalty_types": ["prisão"]})[0] is None
    assert R.validate_infringed({"verdict": "yes", "evidence": "x"})[0]["verdict"] == "yes"
    assert R.validate_infringed({"verdict": "perhaps"})[0] is None and R.validate_infringed("yes")[0] is None


def test_passages_that_cite_the_act():
    text = "x" * 2000 + " infringindo o artigo 2º da Resolução CNSP nº 102/2004, " + "y" * 2000 + " Circular 102/2010 "
    passages = R.act_passages("RES-CNSP102/2004:art2", "sem citação", text, window=100)
    assert len(passages) == 1 and "102/2004" in passages[0] and len(passages[0]) < 260
    assert R.act_passages("RES-CMN3308/2005:art1", "Resolução CMN n° 3.308/05 e 3308, de 2005") != []
    assert R.act_passages("CC:art757", "art. 757 do Código Civil") != [] and R.act_passages("DL73/1966:art84", "artigo 84") == []
    assert "[1] ..." in R.passages_prompt("CC:art757", ["art. 757 do Código Civil"])
    assert R.validate_passages({"verdict": "not_in_text"})[0] is None


def _labels(n=40):
    rows = []
    for i in range(n):
        base = [f"CIRC-SUSEP{300 + i}/2008:art{i + 1}", "DL73/1966:art88"]
        rows.append({"acordao_hash": f"h{i:02d}", "base_provisions": base, "text_primary": base[0] if i % 2 == 0 else None,
                     "text_cited": [base[0]], "penalty_types": ["multa"]})
    return pd.DataFrame(rows)


def test_sample_keeps_the_audited_decisions_and_avoids_development_material():
    labels = _labels()
    audited, excluded = ["h03", "h01"], {"h00", "h02"}
    sample = R.review_sample(labels, audited, excluded, n=10, seed=1)
    assert sample[:2] == audited and len(sample) == len(set(sample)) == 10 and not excluded & set(sample)
    assert sample == R.review_sample(labels, audited, excluded, n=10, seed=1)
    assert R.review_sample(labels, audited, excluded, n=1, seed=1) == audited                # never fewer than the audit


def test_controls_carry_exactly_one_planted_error():
    labels = _labels()
    sample = [f"h{i:02d}" for i in range(10)]
    controls = R.controls(labels, sample, {"h10"}, n_base=5, n_infringed=5, seed=3)
    assert len(controls) == 10 and not set(controls["acordao_hash"]) & (set(sample) | {"h10"})
    assert controls["acordao_hash"].is_unique
    lab = labels.set_index("acordao_hash")
    for row in controls.itertuples():
        if row.kind == "base":
            original = list(lab.at[row.acordao_hash, "base_provisions"])
            assert list(row.label) != original and len(set(original) ^ set(row.label)) in (1, 2)
        else:
            assert row.label != lab.at[row.acordao_hash, "text_primary"]
            assert row.label.split(":art")[0] not in {p.split(":art")[0] for p in lab.at[row.acordao_hash, "text_cited"]}
    rng = np.random.default_rng(0)
    assert R.corrupt_base(["DL73/1966:art88"], rng) == (["DL73/1966:art98"], "changed DL73/1966:art88 to DL73/1966:art98")


def test_agreement_and_kappa():
    a = pd.Series(["C01", "C02", "C02", "C05", None])
    b = pd.Series(["C01", "C02", "C03", "C05", "C01"])
    out = R.agreement(a, b)
    assert out["n"] == 4 and out["agreement"] == 0.75 and 0 < out["kappa"] < 1
    assert R.agreement(pd.Series([None]), pd.Series(["C01"])) == {"n": 0, "agreement": None, "kappa": None}


class FakeReviewer:
    """Says every list is correct unless it holds an article above 50 or a provision of act 999."""

    def complete(self, prompt, *, system=None, json_output=False, cache_extra=None):
        task = (cache_extra or {}).get("task")
        if task == "base":
            listed = prompt.split("PARSED PROVISIONS:\n")[1].split("\n")[0]
            wrong = "DL73/1966:art88" not in listed or "art9" in listed
            parsed = {"base_verdict": "wrong" if wrong else "correct", "base_missing": ["art. 88"] if wrong else [],
                      "base_spurious": [], "base_comment": "", "penalty_types": ["multa"]}
        elif task == "infringed":
            parsed = {"verdict": "no" if "CIRC-SUSEP3" not in prompt.split("\n")[4] else "yes", "evidence": "", "comment": ""}
        else:
            parsed = {"primary": "C05", "secondary": [], "evidence": "FIP"}
        return SimpleNamespace(parsed=parsed, cached=False, input_tokens=10, output_tokens=5, cost_usd=0.001,
                               latency_s=0.1, credentials="x")


def test_review_runs_end_to_end_with_controls_and_hugos_marks():
    labels = _labels(12)
    corpus = pd.DataFrame({"acordao_hash": labels["acordao_hash"], "acordao_base": "Art. 88 do Decreto-Lei nº 73/1966",
                           "acordao_penalidade": "Multa", "acordao_ementa": "Envio do FIP fora do prazo.",
                           "acordao_relatorio": "Relatório.", "acordao_voto_relator": "Voto.", "acordao_titulo": "t",
                           "acordao_url": "u"})
    sample = ["h00", "h01", "h02", "h03"]
    controls = pd.DataFrame([{"acordao_hash": "h05", "kind": "base", "label": ["CIRC-SUSEP305/2008:art6"], "planted": "dropped"},
                             {"acordao_hash": "h06", "kind": "infringed", "label": "RES-CNSP999/2020:art1", "planted": "replaced"}])
    verdicts, control_verdicts, calls = script.run_review(FakeReviewer(), labeller, TAXONOMY, labels, corpus, sample, controls, workers=1)
    assert len(verdicts) == 4 and verdicts["base_verdict"].tolist() == ["correct"] * 4
    assert verdicts["penalty_agree"].all() and verdicts["reviewer_category"].tolist() == ["C05"] * 4
    assert verdicts["infringed_verdict"].tolist()[0] == "yes" and pd.isna(verdicts["infringed_verdict"].tolist()[1])   # no label, no question
    assert control_verdicts["caught"].tolist() == [True, True]
    assert {c["task"] for c in calls} == {"base", "infringed", "category"}        # the act is not in the text: no second look

    verdicts["category_gemini"] = ["C05", "C05", "C02", "C05"]
    sheet = pd.DataFrame({"acordao_hash": sample[:3], "audit_id": ["A01", "A02", "A03"], "ok_base": [1, 0, 1],
                          "ok_penalty": [1, 1, 1], "ok_infringed": [1, np.nan, 1], "ok_category": [1, 0, np.nan],
                          "correct_category": [None, "C02", None], "label_category": ["C05: x", "C05: x", "C02: y"]})
    old = labels.copy()
    old.at[2, "base_provisions"] = ["something else"]                                        # the label changed after the audit
    hugo = script.hugo_marks(sheet, old, labels)
    assert hugo["label_unchanged"].tolist() == [True, True, False] and hugo["hugo_category"].tolist()[:2] == ["C05", "C02"]
    summary = script.summarise(verdicts, control_verdicts, hugo)
    assert summary["legal_basis"]["correct"] == 4 and summary["penalty_type"]["share"] == 1.0
    assert summary["controls"] == {"base": {"planted": 1, "caught": 1}, "infringed": {"planted": 1, "caught": 1}}
    assert summary["agreement_with_hugo"]["legal_basis"] == {"n": 2, "same": 1, "hugo_correct_reviewer_wrong": 0,
                                                             "hugo_wrong_reviewer_correct": 1}
    assert summary["category"]["reviewer vs gemini"]["agreement"] == 0.75
    assert summary["category"]["reviewer vs hugo"]["n"] == 2

    table = script.disputes(verdicts)
    assert table["label_type"].tolist() == ["infraction category"] and table["acordao_hash"].tolist() == ["h02"]
