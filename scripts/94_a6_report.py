"""Phase 9, A6: latency and cost per endpoint and defense arm, from the calls of scripts/93.

Latency is the client's wall-clock time on this machine (local Flask server plus the cloud APIs), not
production Cloud Run. Warm-up calls are left out; failed calls stay in, with their time. Costs are the
model calls the server logged, at config/prices.yaml (Standard PayGo; thinking counted as output). Not in
them: Google Search grounding of the legal search (per query; the server does not record the queries) and
the embedding calls (the server logs no token counts; at most 2 x 2,048 tokens per upload, $0.0006).

Outputs (never overwritten):
  results/a6_latency_cost.csv     one row per endpoint and defense arm: p50 (with a bootstrap CI) and p95
                                  latency, success, model calls, tokens and USD per call
  results/a6_latency_paired.csv   defense arms paired by round (Wilcoxon, Holm)
  results/a6_latency_by_round.csv the defense latencies of each round (for the reader to check the pairing)
  results/costs_phase9.csv        cost of every call sent in the phase, warm-ups included
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from jm_eval import latency as L
from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "94_a6_report"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = L.settings()
    calls_path = resolve(load_config("paths")["data"]["v2"]) / "a6" / "calls.jsonl"
    results = resolve(load_config("paths")["results"])
    with RunManifest(SCRIPT, results, params={"config": cfg}) as run:
        run.add_input(calls_path)
        calls = L.read_calls(calls_path)
        m = L.measured(calls)
        expected = cfg["rounds"] * (len(L.ENDPOINTS) + len(cfg["endpoints"]["defense"]["arms"]))
        order = [*L.ENDPOINTS, *cfg["endpoints"]["defense"]["arms"]]
        summary = L.summarise(m, cfg).set_index("unit").loc[[u for u in order if u in set(m["unit"])]].reset_index()
        paired = L.paired(m, cfg)
        by_round = (m[m["endpoint"] == "defense"].pivot(index="round", columns="unit", values="wall_s")
                    .join(m[m["unit"] == "case_upload"].set_index("round")["notice_id"]).reset_index())

        paths = {}
        for name, frame in (("a6_latency_cost.csv", summary), ("a6_latency_paired.csv", paired), ("a6_latency_by_round.csv", by_round)):
            paths[name] = new_artefact_path(results / name)
            frame.to_csv(paths[name], index=False)
            run.add_output(paths[name], f"A6: {name}")
        costs = calls.assign(phase="9", part=lambda d: d["warmup"].map({True: "warm-up", False: "measured"}))
        costs = costs[["phase", "part", "round", "unit", "endpoint", "started_utc", "llm_calls", "prompt_tokens",
                       "output_tokens", "cost_usd", "credentials", "server_dir"]]
        paths["costs"] = new_artefact_path(results / "costs_phase9.csv")
        costs.to_csv(paths["costs"], index=False)
        run.add_output(paths["costs"], "Phase 9: cost of every call")

        run.data["summary"] = {
            "measured_calls": len(m), "expected_calls": expected, "failed_calls": int((~m["ok"]).sum()),
            "calls_sent_again_or_superseded": int(len(calls[~calls["warmup"]]) - len(m)),
            "cost_usd_measured": round(float(m["cost_usd"].sum()), 2), "cost_usd_all": round(float(calls["cost_usd"].sum()), 2),
            "calls_not_served_by_primary_key": int(calls["fallback_calls"].sum()),
            "period_utc": [calls["started_utc"].min(), calls["started_utc"].max()],
            "server_starts": int(calls["server_dir"].nunique())}
        print(summary[["unit", "calls", "ok", "latency_p50_s", "latency_p50_ci_low_s", "latency_p50_ci_high_s",
                       "latency_p95_s", "prompt_tokens_mean", "output_tokens_mean", "usd_mean"]].to_string(index=False))
        print(paired.to_string(index=False))
        print(json.dumps(run.data["summary"], indent=1))
        if len(m) < expected:
            print(f"WARNING: {expected - len(m)} measured calls missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
