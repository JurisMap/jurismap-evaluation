"""Phase 4: cost of the model calls, from the run manifests and the embedding input table.

Outputs (never overwritten): results/costs_phase4.csv

- Label review (script 49): the cost logged per call.
- Category labels (script 41): the cost logged per call, each distinct prompt once per model (the
  final run reads from the cache what the pilot or an interrupted run had already paid for).
- Embeddings (script 43): every text was embedded once. Tokens and characters are those of the
  full inputs, so the cost is an upper bound where the model truncated the input.
- Compression rerun (script 45): tokens of the 50 compressions; their embeddings are counted too.
Prices: config/prices.yaml (entries marked `check` are unconfirmed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C

SCRIPT = "48_phase4_costs"


def manifests(results: Path, script: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(results.glob(f"{script}*.manifest.json"))]


def main() -> int:
    paths = load_config("paths")
    prices = load_config("prices")["prices"]
    results = resolve(paths["results"])
    rows = []
    with RunManifest(SCRIPT, results) as run:
        # Category labels: each distinct prompt was paid once, in whichever run first sent it (pilot,
        # interrupted or final). The final run of each model logs every prompt with its cost.
        final: dict[str, dict] = {}
        for m in manifests(results, "41_infraction_category"):
            if m.get("status") == "ok" and not m["params"].get("pilot") and m.get("calls"):
                final[m["models"]["labeler"]] = m
        for model, m in final.items():
            first = {}
            for c in m["calls"]:
                if c["attempt"] == 0 or not c["cached"]:
                    first.setdefault((c["prompt_sha256"], c["attempt"]), c)
            paid = list(first.values())
            rows.append({"step": "infraction category", "script": m["script"], "run_id": m["run_id"], "model": model,
                         "calls": len(paid), "input_tokens": sum(c["input_tokens"] or 0 for c in paid),
                         "output_tokens": sum(c["output_tokens"] or 0 for c in paid), "input_chars": None,
                         "cost_usd": round(sum(c["cost_usd"] or 0 for c in paid), 4),
                         "basis": "logged per call; distinct prompts, pilot and interrupted runs included"})

        inputs_path = latest_artefact_path(C.embeddings_dir() / "a1_embedding_inputs.parquet")
        if inputs_path.exists():
            run.add_input(inputs_path)
            inputs = pd.read_parquet(inputs_path)
            for (method, model), sub in inputs.groupby(["method", "model"], sort=True):
                price = prices[model]
                tokens, chars = int(sub["token_count"].sum()), int(sub["chars"].sum())
                cost = chars / 1e6 * price["input_per_million_chars"] if "input_per_million_chars" in price else tokens / 1e6 * price["input"]
                rows.append({"step": f"embeddings {method}", "script": "43_a1_cloud_embeddings", "run_id": None,
                             "model": model, "calls": len(sub), "input_tokens": tokens, "output_tokens": 0,
                             "input_chars": chars, "cost_usd": round(cost, 4),
                             "basis": "full input length (upper bound); price unconfirmed"})

        for m in manifests(results, "45_compression_stability"):
            tokens = m.get("summary", {}).get("tokens")
            if not tokens:
                continue
            model = m["models"]["compression"]["model"]
            price = prices[model]
            output = tokens.get("output_tokens", 0) + tokens.get("thoughts_tokens", 0)
            cost = tokens["prompt_tokens"] / 1e6 * price["input"] + output / 1e6 * price["output"]
            rows.append({"step": "compression rerun (50 reports)", "script": m["script"], "run_id": m["run_id"],
                         "model": model, "calls": m["summary"]["decisions"], "input_tokens": tokens["prompt_tokens"],
                         "output_tokens": output, "input_chars": None, "cost_usd": round(cost, 4),
                         "basis": "tokens of the rerun; its 50 embeddings add under 0.01"})

        for m in manifests(results, "49_label_review"):
            paid = [c for c in m.get("calls", []) if not c["cached"]]
            if paid:
                rows.append({"step": "label review by a second model", "script": m["script"], "run_id": m["run_id"],
                             "model": m["models"]["reviewer"], "calls": len(paid),
                             "input_tokens": sum(c["input_tokens"] or 0 for c in paid),
                             "output_tokens": sum(c["output_tokens"] or 0 for c in paid), "input_chars": None,
                             "cost_usd": round(sum(c["cost_usd"] or 0 for c in paid), 4),
                             "basis": "logged per call; price unconfirmed"})

        table = pd.DataFrame(rows)
        table.insert(0, "phase", "4")
        total = {"phase": "4", "step": "TOTAL", "cost_usd": round(float(table["cost_usd"].sum()), 4),
                 "basis": "plan line for Phase 4: 10 USD (8 Gemini + 2 embeddings); the second labelling "
                          "and the label review were added on the author's instruction (2026-10-07)"}
        table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
        path = new_artefact_path(results / "costs_phase4.csv")
        table.to_csv(path, index=False)
        run.add_output(path, "Phase 4: cost of model calls")
        run.data["summary"] = {"total_usd": total["cost_usd"]}
        print(table.drop(columns=["run_id", "basis"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
