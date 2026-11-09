"""Tests for jm_eval.latency (Phase 9, A6): the round plan and the summaries."""

import itertools

import numpy as np
import pandas as pd

from jm_eval import latency as L


def test_arm_orders_cycle_through_every_permutation():
    orders = L.arm_orders(["a", "b", "c"], 30)
    assert len(orders) == 30
    counts = {tuple(p): sum(o == list(p) for o in orders) for p in itertools.permutations("abc")}
    assert set(counts.values()) == {5}
    for arm in "abc":                                    # every arm is first, second and third equally often
        assert [sum(o[i] == arm for o in orders) for i in range(3)] == [10, 10, 10]


def test_sample_questions_draws_by_type_without_repeats():
    qs = [{"question_id": f"{t}{i}", "type": t} for t, n in (("templated", 24), ("open", 12), ("unanswerable", 6)) for i in range(n)]
    sample = L.sample_questions(qs, {"templated": 17, "unanswerable": 4, "open": 9}, 20260926)
    assert len(sample) == 30 and len({q["question_id"] for q in sample}) == 30
    assert sum(q["type"] == "open" for q in sample) == 9
    assert sample == L.sample_questions(qs, {"templated": 17, "unanswerable": 4, "open": 9}, 20260926)


def test_round_plan_goes_through_the_notices_twice():
    cfg = L.settings()
    notices = [{"notice_id": f"N{i:02d}"} for i in range(1, 16)]
    questions = [{"question_id": f"Q{i}"} for i in range(30)]
    plan = L.round_plan(cfg, notices, questions)
    assert [p["notice_id"] for p in plan[:2]] == ["N01", "N02"] and plan[15]["notice_id"] == "N01"
    assert plan[15]["notice_pass"] == 2
    assert set(plan[0]["arms"]) == set(cfg["endpoints"]["defense"]["arms"])


def calls_frame():
    rows = []
    for r in range(1, 11):
        for unit, wall in (("v2.7_full", 100 + r), ("v2.6_full", 90 + r), ("chat", 30.0)):
            rows.append({"round": r, "warmup": False, "unit": unit, "endpoint": "chat" if unit == "chat" else "defense",
                         "ok": True, "wall_s": float(wall), "server_s": wall - 0.5, "first_event_s": None,
                         "llm_calls": 5, "prompt_tokens": 1000, "output_tokens": 200, "thoughts_tokens": 150,
                         "cost_usd": 0.2, "fallback_calls": 0, "started_utc": f"2026-10-26T00:{r:02d}:00+00:00"})
    rows.append({**rows[0], "round": -1, "warmup": True, "wall_s": 999.0})
    rows.append({**rows[0], "wall_s": 120.0})            # sent again: the last record counts
    return pd.DataFrame(rows)


def test_measured_drops_warmups_and_keeps_the_last_record():
    m = L.measured(calls_frame())
    assert (m["round"] > 0).all()
    assert len(m) == 30
    assert m[(m["round"] == 1) & (m["unit"] == "v2.7_full")]["wall_s"].item() == 120.0


def test_summarise_percentiles_and_costs():
    cfg = {**L.settings(), "analysis": {**L.settings()["analysis"], "bootstrap": {"resamples": 500, "ci": 0.95}}}
    s = L.summarise(L.measured(calls_frame()), cfg).set_index("unit")
    chat = s.loc["chat"]
    assert chat["latency_p50_s"] == 30.0 and chat["latency_p95_s"] == 30.0
    assert chat["usd_total"] == 2.0 and chat["success_rate"] == 1.0
    full = s.loc["v2.6_full"]
    assert full["latency_p50_s"] == np.percentile(np.arange(91, 101), 50).round(1)


def test_paired_differences_by_round():
    cfg = {**L.settings(), "analysis": {**L.settings()["analysis"], "paired": [["v2.7_full", "v2.6_full"]]}}
    p = L.paired(L.measured(calls_frame()), cfg).iloc[0]
    assert p["rounds"] == 10 and p["median_difference_s"] == 10.0
    assert p["share_of_rounds_slower"] == 1.0 and p["wilcoxon_p"] < 0.01
