"""Phase 4a, A1: infraction category of each decision (secondary silver label, R-cat).

A labelling model (config/models.yaml `labelers.infraction_category`: Gemini 3.7 Flash, temperature
0, JSON schema) reads the ementa and the report of a decision and assigns one category of the
closed taxonomy in config/a1_taxonomy.yaml, plus up to two secondary categories and the words of
the text that support the choice.

- `--labeler NAME` chooses another entry of `labelers` (e.g. infraction_category_mistral, a blind
  second labelling by a non-Gemini model); its files carry the model family in their names.
- The corpus is labelled only with an approved taxonomy (`status: approved`).
- `--pilot` labels the 30-document development sample (config/a1_labels.yaml) with whatever
  taxonomy is in the file, to show how it fits before approval. Its output is marked as a pilot.
- Every response is cached (data/cache/llm/); the manifest holds the call log with tokens and cost.
- Only public CRSNSP text is sent.

Outputs (never overwritten):
  data/labels/a1_infraction_category.parquet     (pilot: a1_infraction_category_pilot.parquet)
  results/a1_category_distribution.csv           (pilot: a1_category_pilot_distribution.csv)
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from jm_eval.cache import prompt_hash
from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.llm import get_model
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import labels as L

SCRIPT = "41_infraction_category"
DEFAULT_LABELER = "infraction_category"
MAX_FORMAT_RETRIES = 2
MAX_SECONDARY = 2

SYSTEM = ("You classify decisions of a Brazilian administrative appeals council for insurance (CRSNSP) by the "
          "kind of infraction the party was accused of. Apply the category definitions literally. "
          "Answer with JSON only.")

INSTRUCTIONS = """Below are the headnote (ementa) and the rapporteur's report of one CRSNSP decision, in Portuguese. Assign the category of the infraction.

Categories (id: label. Definition. Typical wording in Portuguese):
{categories}

Rules:
{rules}

Return JSON only:
{{"primary": "<category id>", "secondary": [<up to {max_secondary} other category ids, or none>], "evidence": "<at most 25 consecutive words copied exactly from the text that show the primary category>"}}

EMENTA:
{ementa}

RELATÓRIO:
{report}"""


def load_taxonomy() -> dict:
    return load_config("a1_taxonomy")


def category_ids(taxonomy: dict) -> list[str]:
    return [c["id"] for c in taxonomy["categories"]]


def response_schema(taxonomy: dict) -> dict:
    ids = category_ids(taxonomy)
    return {"type": "OBJECT", "required": ["primary", "secondary", "evidence"], "properties": {
        "primary": {"type": "STRING", "enum": ids},
        "secondary": {"type": "ARRAY", "items": {"type": "STRING", "enum": ids}},
        "evidence": {"type": "STRING"}}}


def build_prompt(taxonomy: dict, ementa: str | None, report: str | None) -> str:
    categories = "\n".join(
        f"{c['id']}: {c['label']}. {c['definition']}"
        + (" Typical wording: " + "; ".join(f'"{cue}"' for cue in c["cues_pt"]) + "." if c.get("cues_pt") else "")
        for c in taxonomy["categories"])
    rules = "\n".join(f"{n}. {rule}" for n, rule in enumerate(taxonomy["rules"], 1))
    clean = lambda t: re.sub(r"[ \t]+", " ", t if isinstance(t, str) else "").strip()  # noqa: E731
    return INSTRUCTIONS.format(categories=categories, rules=rules, max_secondary=MAX_SECONDARY,
                               ementa=clean(ementa) or "(sem ementa)",
                               report=clean(report)[:taxonomy["input"]["max_report_chars"]] or "(sem relatório)")


def validate(answer: object, ids: list[str]) -> tuple[dict | None, str | None]:
    """A cleaned label, or the reason the answer cannot be used."""
    if not isinstance(answer, dict) or answer.get("primary") not in ids:
        return None, "primary is not a category id"
    secondary = answer.get("secondary") or []
    if not isinstance(secondary, list) or any(s not in ids for s in secondary):
        return None, "secondary holds an unknown category id"
    secondary = [s for s in dict.fromkeys(secondary) if s != answer["primary"]][:MAX_SECONDARY]
    return {"primary": answer["primary"], "secondary": secondary, "evidence": str(answer.get("evidence") or "")}, None


def is_verbatim(evidence: str, *texts: str | None) -> bool:
    norm = lambda t: re.sub(r"\W+", " ", t if isinstance(t, str) else "").lower().strip()  # noqa: E731
    quote = norm(evidence)
    return bool(quote) and any(quote in norm(t) for t in texts)


def label_one(model, taxonomy: dict, row: dict) -> dict:
    """Label one decision; an unusable answer is asked again, up to `MAX_FORMAT_RETRIES` times."""
    ids = category_ids(taxonomy)
    prompt = build_prompt(taxonomy, row["acordao_ementa"], row["acordao_relatorio"])
    record = {"acordao_hash": row["acordao_hash"], "prompt_sha256": prompt_hash(prompt), "calls": []}
    for attempt in range(MAX_FORMAT_RETRIES + 1):
        extra = {"taxonomy_version": taxonomy["version"], **({"format_retry": attempt} if attempt else {})}
        problem = None
        try:
            result = model.complete(prompt, system=SYSTEM, json_output=True, cache_extra=extra)
            label, problem = validate(result.parsed, ids)
        except ValueError as e:                     # not JSON
            result, label, problem = None, None, f"not JSON: {e}"
        if result is not None:
            record["calls"].append({"attempt": attempt, "cached": result.cached, "input_tokens": result.input_tokens,
                                    "output_tokens": result.output_tokens, "cost_usd": result.cost_usd,
                                    "latency_s": result.latency_s, "credentials": result.credentials,
                                    "problem": problem})
        if label:
            return {**record, **label, "format_retries": attempt, "error": None,
                    "evidence_verbatim": is_verbatim(label["evidence"], row["acordao_ementa"], row["acordao_relatorio"])}
    return {**record, "primary": None, "secondary": [], "evidence": "", "format_retries": MAX_FORMAT_RETRIES,
            "error": problem, "evidence_verbatim": False}


def distribution(labels: pd.DataFrame, taxonomy: dict) -> pd.DataFrame:
    primary = Counter(labels["primary"].dropna())
    secondary = Counter(s for row in labels["secondary"] for s in row)
    n = int(labels["primary"].notna().sum())
    return pd.DataFrame([{"category": c["id"], "label": c["label"], "primary": primary.get(c["id"], 0),
                          "share_primary": round(primary.get(c["id"], 0) / n, 4) if n else None,
                          "secondary": secondary.get(c["id"], 0)} for c in taxonomy["categories"]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="label only the development sample (draft taxonomy allowed)")
    ap.add_argument("--labeler", default=DEFAULT_LABELER, help="entry of `labelers` in config/models.yaml")
    ap.add_argument("--limit", type=int, help="label only the first N decisions (by acordao_hash)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", help="write everything here instead of data/labels/ and results/ (tests)")
    args = ap.parse_args()

    taxonomy, cfg, paths = load_taxonomy(), L.settings(), load_config("paths")
    if not args.pilot and taxonomy.get("status") != "approved":
        raise SystemExit(f"config/a1_taxonomy.yaml has status '{taxonomy.get('status')}': the corpus is labelled only "
                         "with an approved taxonomy. Use --pilot for the development sample.")
    corpus_path = resolve(cfg["corpus"])
    labels_dir = Path(args.out_dir) if args.out_dir else resolve(paths["data"]["labels"])
    results = Path(args.out_dir) if args.out_dir else resolve(paths["results"])
    labels_dir.mkdir(parents=True, exist_ok=True)

    spec = load_config("models")["labelers"][args.labeler]
    overrides = {"response_schema": response_schema(taxonomy)} if spec["api"] == "genai" else {}
    model = get_model("labelers", args.labeler, **overrides)
    suffix = ("" if args.labeler == DEFAULT_LABELER else f"_{spec['family']}") + ("_pilot" if args.pilot else "")
    params = {"taxonomy_version": taxonomy["version"], "taxonomy_status": taxonomy["status"], "pilot": args.pilot,
              "labeler": args.labeler,
              "input": taxonomy["input"], "generation": model.params(), "workers": args.workers}
    with RunManifest(SCRIPT, results, models={"labeler": model.model}, params=params,
                     record_outputs=args.out_dir is None) as run:
        run.add_input(corpus_path)
        corpus = pd.read_parquet(corpus_path, columns=["acordao_hash", *taxonomy["input"]["fields"]])
        corpus = corpus.sort_values("acordao_hash")
        if args.pilot:
            corpus = corpus[corpus["acordao_hash"].isin(set(cfg["development"]["ids"]))]
        if args.limit:
            corpus = corpus.head(args.limit)
        rows = corpus.to_dict("records")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(lambda row: label_one(model, taxonomy, row), rows))

        labels = pd.DataFrame(records)
        calls = [c for r in records for c in r["calls"]]
        labels["taxonomy_version"] = taxonomy["version"]
        labels["taxonomy_status"] = taxonomy["status"]
        labels_path = new_artefact_path(labels_dir / f"a1_infraction_category{suffix}.parquet")
        labels.drop(columns=["calls"]).to_parquet(labels_path, index=False)
        run.add_output(labels_path, f"A1 infraction category per decision ({model.model})" + (" (pilot, development sample)" if args.pilot else ""))

        dist = distribution(labels, taxonomy)
        dist_path = new_artefact_path(results / f"a1_category{suffix}_distribution.csv")
        dist.to_csv(dist_path, index=False)
        run.add_output(dist_path, f"A1 infraction categories: distribution ({model.model})" + (" (pilot)" if args.pilot else ""))

        new_calls = [c for c in calls if not c["cached"]]
        cost = sum(c["cost_usd"] or 0 for c in calls)
        summary = {
            "decisions": len(labels), "labelled": int(labels["primary"].notna().sum()),
            "unusable_after_retries": int(labels["primary"].isna().sum()),
            "format_retries": int(labels["format_retries"].sum()),
            "evidence_verbatim_share": round(float(labels["evidence_verbatim"].mean()), 4) if len(labels) else None,
            "calls": len(calls), "calls_not_cached": len(new_calls),
            "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
            "cost_usd_all_calls": round(cost, 4),
            "cost_usd_per_decision": round(cost / len(labels), 5) if len(labels) else None,
            "credentials": dict(Counter(c["credentials"] for c in calls)),
        }
        run.data["summary"] = summary
        run.data["calls"] = [{"acordao_hash": r["acordao_hash"], "prompt_sha256": r["prompt_sha256"], **c}
                             for r in records for c in r["calls"]]
        print(dist.to_string(index=False))
        print(summary)
    return 0 if summary["unusable_after_retries"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
