"""A6 latency and cost (Phase 9): the plan of rounds and the summaries of the measured calls.

A round is one map view, one case upload, one chat question and one draft of each defense arm, sent
one at a time. The defense arms' order cycles through every permutation, so no arm is always first.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jm_eval.io import load_config
from jm_eval.stats import holm

CONFIG = "a6_latency_cost"
ENDPOINTS = ("map_view", "case_upload", "chat")


def settings() -> dict:
    return load_config(CONFIG)


def arm_orders(arms: list[str], rounds: int) -> list[list[str]]:
    """The arms' order in each round: the permutations of `arms` in turn."""
    perms = [list(p) for p in itertools.permutations(arms)]
    return [perms[i % len(perms)] for i in range(rounds)]


def sample_questions(questions: list[dict], sample: dict[str, int], seed: int) -> list[dict]:
    """`sample[type]` questions of each type, drawn without replacement, then shuffled into one order."""
    rng = np.random.default_rng(seed)
    chosen = []
    for qtype, n in sample.items():
        pool = sorted((q for q in questions if q["type"] == qtype), key=lambda q: q["question_id"])
        if len(pool) < n:
            raise ValueError(f"{n} {qtype} questions asked, {len(pool)} available")
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    return [chosen[i] for i in rng.permutation(len(chosen))]


def round_plan(cfg: dict, notices: list[dict], questions: list[dict]) -> list[dict]:
    """The measured rounds: the notice (in turn), the chat question and the defense arms' order of each."""
    n = cfg["rounds"]
    if len(questions) != n:
        raise ValueError(f"{len(questions)} questions for {n} rounds")
    orders = arm_orders(list(cfg["endpoints"]["defense"]["arms"]), n)
    return [{"round": r + 1, "notice_id": notices[r % len(notices)]["notice_id"], "notice_pass": r // len(notices) + 1,
             "question_id": questions[r]["question_id"], "arms": orders[r]} for r in range(n)]


def read_calls(path: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.DataFrame(rows)


def measured(calls: pd.DataFrame) -> pd.DataFrame:
    """The calls that count: not warm-ups, and the last record of each (round, unit) if one was sent again."""
    m = calls[~calls["warmup"]].copy()
    return m.drop_duplicates(["round", "unit"], keep="last")


def percentile(values, q: float) -> float:
    x = np.asarray(values, dtype=float)
    return float(np.percentile(x, q)) if len(x) else float("nan")


def bootstrap_median_ci(values, *, resamples: int, level: float, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.median(x[rng.integers(0, len(x), size=(resamples, len(x)))], axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(medians, [tail, 1 - tail])
    return float(low), float(high)


def summarise(calls: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per unit (endpoint, or defense arm): latency percentiles, success, tokens and USD per call."""
    a = cfg["analysis"]
    rows = []
    for unit, g in calls.groupby("unit", sort=False):
        wall = g["wall_s"].to_numpy(float)
        server = g["server_s"].dropna().to_numpy(float)
        low, high = bootstrap_median_ci(wall, resamples=a["bootstrap"]["resamples"], level=a["bootstrap"]["ci"], seed=cfg["seed"])
        row = {"unit": unit, "endpoint": g["endpoint"].iloc[0], "calls": len(g), "ok": int(g["ok"].sum()),
               "success_rate": round(float(g["ok"].mean()), 3)}
        for q in a["percentiles"]:
            row[f"latency_p{q}_s"] = round(percentile(wall, q), 1)
        row |= {"latency_p50_ci_low_s": round(low, 1), "latency_p50_ci_high_s": round(high, 1),
                "latency_max_s": round(float(wall.max()), 1),
                "server_latency_p50_s": round(percentile(server, 50), 1), "server_latency_p95_s": round(percentile(server, 95), 1),
                "first_event_p50_s": round(percentile(g["first_event_s"].dropna(), 50), 1) if "first_event_s" in g else None,
                "model_calls_mean": round(float(g["llm_calls"].mean()), 2),
                "prompt_tokens_mean": round(float(g["prompt_tokens"].mean())),
                "prompt_tokens_p50": round(percentile(g["prompt_tokens"], 50)),
                "output_tokens_mean": round(float(g["output_tokens"].mean())),          # thinking included
                "thoughts_tokens_mean": round(float(g["thoughts_tokens"].mean())),
                "usd_mean": round(float(g["cost_usd"].mean()), 4), "usd_p50": round(percentile(g["cost_usd"], 50), 4),
                "usd_p95": round(percentile(g["cost_usd"], 95), 4), "usd_total": round(float(g["cost_usd"].sum()), 4),
                "calls_not_served_by_primary_key": int(g["fallback_calls"].sum()),
                "first_utc": g["started_utc"].min(), "last_utc": g["started_utc"].max()}
        rows.append(row)
    return pd.DataFrame(rows)


def paired(calls: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Wilcoxon signed-rank tests of wall-clock latency between defense arms, paired by round; Holm across them."""
    from scipy.stats import wilcoxon

    wide = calls[calls["endpoint"] == "defense"].pivot(index="round", columns="unit", values="wall_s")
    rows = []
    for a, b in cfg["analysis"]["paired"]:
        pair = wide[[a, b]].dropna()
        d = pair[a] - pair[b]
        p = float(wilcoxon(pair[a], pair[b]).pvalue) if len(pair) else float("nan")
        rows.append({"arm": a, "against": b, "rounds": len(pair), "median_difference_s": round(float(d.median()), 1),
                     "mean_difference_s": round(float(d.mean()), 1), "share_of_rounds_slower": round(float((d > 0).mean()), 3),
                     "median_ratio": round(float((pair[a] / pair[b]).median()), 3), "wilcoxon_p": p})
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].to_numpy()) if len(out) else []
    return out
