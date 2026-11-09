"""Phase 4a, A1: silver labels of the CRSNSP corpus from fields it already has. No LLM calls.

Per decision of the frozen v2 corpus (config/a1_labels.yaml):
- legal basis: `acordao_base` parsed into provision IDs such as DL73/1966:art113; `base_key` is the
  set without the generic provisions and the sanction regulations (R-base);
- infringed provisions and all cited provisions read from the report and the vote (R-prov);
- penalty types (`acordao_penalidade`) and outcome (`acordao_resultado`);
- process group: decisions of the same process, excluded from each other's rankings.

Outputs (never overwritten):
  data/labels/a1_silver_labels.parquet
  results/a1_label_coverage.csv     coverage of every label on the comparison set and on v2
  results/a1_label_summary.json     relevance definitions compared, agreement between the two label
                                    sources (development sample and the rest), distributions
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import labels as L
from jm_eval.retrieval.provisions import build_canon, provision_ids

SCRIPT = "40_silver_labels"
COLUMNS = ["acordao_hash", "acordao_titulo", "acordao_url", "acordao_base", "acordao_penalidade",
           "acordao_resultado", "acordao_relatorio", "acordao_voto_relator"]


def build_labels(corpus: pd.DataFrame, cfg: dict, shared: set[str]) -> pd.DataFrame:
    generic, sanction = cfg["generic_provisions"], cfg["sanction_norms"]
    texts = (corpus["acordao_relatorio"].fillna("") + "\n" + corpus["acordao_voto_relator"].fillna("")).tolist()
    # Years and issuers that a citation omits are completed from the corpus's own citations.
    canon = build_canon(list(corpus["acordao_base"]) + texts)
    out = pd.DataFrame({"acordao_hash": corpus["acordao_hash"].to_numpy()})
    out["in_comparison_set"] = out["acordao_hash"].isin(shared)
    out["in_development_sample"] = out["acordao_hash"].isin(set(cfg["development"]["ids"]))
    keys = [L.process_keys(t, u) for t, u in zip(corpus["acordao_titulo"], corpus["acordao_url"])]
    out["process_keys"] = [sorted(k) for k in keys]
    # A decision without a readable number forms its own group.
    out["process_group"] = L.process_groups([k or {f"#{h}"} for k, h in zip(keys, out["acordao_hash"])])
    out["base_provisions"] = [provision_ids(b, canon) for b in corpus["acordao_base"]]
    out["base_specific"] = [L.specific(p, generic, sanction) for p in out["base_provisions"]]
    out["base_key"] = [L.key_of(p) for p in out["base_specific"]]
    out["text_infringed"] = [L.specific(L.infringed_provisions(t, canon), generic, sanction) for t in texts]
    out["text_primary"] = [p[0] if p else None for p in out["text_infringed"]]
    out["text_cited"] = [L.specific(L.cited_provisions(t, canon), generic, sanction) for t in texts]
    out["penalty_types"] = [L.penalty_types(p) for p in corpus["acordao_penalidade"]]
    out["penalty_key"] = ["+".join(p) or None for p in out["penalty_types"]]
    out["outcome"] = [L.outcome(r) for r in corpus["acordao_resultado"]]
    return out


def coverage(labels: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    has_text = lambda col: corpus[col].fillna("").str.strip().ne("").to_numpy()  # noqa: E731
    checks = {
        "acordao_base filled": has_text("acordao_base"),
        "legal basis parsed (at least one provision)": labels["base_provisions"].str.len().gt(0).to_numpy(),
        "R-base key (a specific provision in the basis)": labels["base_key"].notna().to_numpy(),
        "infringed provision found in report or vote": labels["text_primary"].notna().to_numpy(),
        "R-base key or infringed provision": (labels["base_key"].notna() | labels["text_primary"].notna()).to_numpy(),
        "a specific provision cited in report or vote": labels["text_cited"].str.len().gt(0).to_numpy(),
        "acordao_penalidade filled": has_text("acordao_penalidade"),
        "penalty type": labels["penalty_key"].notna().to_numpy(),
        "outcome": labels["outcome"].notna().to_numpy(),
    }
    rows = []
    for scope, mask in scopes(labels).items():
        for label, ok in checks.items():
            rows.append({"scope": scope, "label": label, "decisions": int(mask.sum()),
                         "with_label": int((ok & mask).sum()), "share": round(float(ok[mask].mean()), 4)})
    return pd.DataFrame(rows)


def scopes(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"comparison_set": labels["in_comparison_set"].to_numpy(),
            "v2": np.ones(len(labels), dtype=bool)}


def relevance_definitions(labels: pd.DataFrame, threshold: float) -> dict:
    """Coverage of each candidate relevance definition, per scope."""
    out = {}
    for scope, mask in scopes(labels).items():
        sub = labels[mask]
        groups = sub["process_group"].to_numpy()
        prov = L.build_relevance(sub, {"source": "text_cited", "kind": "jaccard", "or_same": "text_primary"}, threshold)
        definitions = {
            "R-base strict (identical specific provisions)":
                L.relevance_matrix(sub["base_specific"], groups),
            f"R-base relaxed (Jaccard >= {threshold} of the specific provisions)":
                L.relevance_matrix(sub["base_specific"], groups, kind="jaccard", threshold=threshold),
            "R-base as cited (identical provisions, generic and sanction articles kept)":
                L.relevance_matrix(sub["base_provisions"], groups),
            f"R-prov (same primary infringed provision, or Jaccard >= {threshold} of the cited provisions)": prov,
        }
        out[scope] = {name: L.relevance_summary(rel) for name, rel in definitions.items()}
    return out


def source_agreement(labels: pd.DataFrame) -> dict:
    """How far the provisions read from the report and vote agree with `acordao_base`."""
    out = {}
    parts = {"development_sample": labels["in_development_sample"], "other_decisions": ~labels["in_development_sample"]}
    for name, mask in parts.items():
        sub = labels[mask & labels["base_key"].notna()]
        found = sub[sub["text_primary"].notna()]
        in_base = [p in set(b) for p, b in zip(found["text_primary"], found["base_specific"])]
        overlap = [bool(set(t) & set(b)) for t, b in zip(found["text_infringed"], found["base_specific"])]
        out[name] = {
            "decisions_with_base_key": len(sub),
            "with_infringed_provision_in_text": len(found),
            "share_with_infringed_provision_in_text": round(len(found) / len(sub), 4) if len(sub) else None,
            "primary_provision_is_in_base": round(float(np.mean(in_base)), 4) if in_base else None,
            "any_infringed_provision_is_in_base": round(float(np.mean(overlap)), 4) if overlap else None,
        }
    return out


def distributions(labels: pd.DataFrame, top: int = 25) -> dict:
    out = {}
    for scope, mask in scopes(labels).items():
        sub = labels[mask]
        provisions = Counter(p for ps in sub["base_provisions"] for p in ps)
        keys = sub["base_key"].dropna().value_counts()
        out[scope] = {
            "distinct_base_provisions": len(provisions),
            "top_base_provisions": dict(provisions.most_common(top)),
            "distinct_base_keys": int(keys.size),
            "base_keys_held_by_one_decision": int((keys == 1).sum()),
            "top_base_keys": {k: int(v) for k, v in keys.head(top).items()},
            "penalty_key": {k: int(v) for k, v in sub["penalty_key"].fillna("(none)").value_counts().items()},
            "outcome": {k: int(v) for k, v in sub["outcome"].fillna("(none)").value_counts().items()},
            "process_groups": L.group_sizes(sub["process_group"].tolist()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="default: `corpus` in config/a1_labels.yaml")
    ap.add_argument("--out-dir", help="write everything here instead of data/labels/ and results/ (tests)")
    args = ap.parse_args()

    cfg = L.settings()
    paths = load_config("paths")
    corpus_path = resolve(args.corpus or cfg["corpus"])
    labels_dir = Path(args.out_dir) if args.out_dir else resolve(paths["data"]["labels"])
    results = Path(args.out_dir) if args.out_dir else resolve(paths["results"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    params = {"labels_version": cfg["version"], "labels_status": cfg["status"],
              "generic_provisions": cfg["generic_provisions"], "sanction_norms": cfg["sanction_norms"],
              "jaccard_threshold": cfg["relevance"]["jaccard_threshold"],
              "development": ("the patterns for the infringed provision were tuned only on the development sample "
                              "of config/a1_labels.yaml; the citation parser was also checked against the "
                              "legal-basis field of the whole corpus and corrected on a first audit draw, "
                              "which is excluded from the audit")}
    with RunManifest(SCRIPT, results, params=params, record_outputs=args.out_dir is None) as run:
        run.add_input(corpus_path)
        run.add_input(resolve(cfg["comparison_set"]))
        corpus = pd.read_parquet(corpus_path, columns=COLUMNS)
        shared = set(resolve(cfg["comparison_set"]).read_text().split())
        labels = build_labels(corpus, cfg, shared)

        labels_path = new_artefact_path(labels_dir / "a1_silver_labels.parquet")
        labels.to_parquet(labels_path, index=False)
        run.add_output(labels_path, "A1 silver labels (legal basis, provisions, penalty, outcome, process group)")

        cov = coverage(labels, corpus)
        cov_path = new_artefact_path(results / "a1_label_coverage.csv")
        cov.to_csv(cov_path, index=False)
        run.add_output(cov_path, "A1 silver labels: coverage on the comparison set and on v2")

        summary = {
            "labels_version": cfg["version"], "labels_status": cfg["status"],
            "decisions": {scope: int(mask.sum()) for scope, mask in scopes(labels).items()},
            "relevance_definitions": relevance_definitions(labels, cfg["relevance"]["jaccard_threshold"]),
            "agreement_text_vs_base": source_agreement(labels),
            "distributions": distributions(labels),
        }
        summary_path = new_artefact_path(results / "a1_label_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(summary_path, "A1 silver labels: relevance definitions, agreement, distributions")
        run.data["summary"] = {"decisions": summary["decisions"], "labels": labels_path.name}

        print(cov.to_string(index=False))
        for scope, definitions in summary["relevance_definitions"].items():
            print(f"\n{scope}")
            for name, s in definitions.items():
                print(f"  {s['queries_with_relevant']:5d} queries ({s['share_queries_with_relevant']:.1%}), "
                      f"median {s['relevant_per_query_median']} relevant  {name}")
        print("\nagreement:", json.dumps(summary["agreement_text_vs_base"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
