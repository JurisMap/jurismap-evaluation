"""Phase 4d, A1 (secondary): the SUSEP gold standard. LOCAL ONLY (ground rule 1).

The SUSEP infraction texts never leave this machine. Before any private file is opened, the script
refuses to run if a cloud client is loaded or a proxy is configured, puts the model libraries in
offline mode and blocks every network connection of the process (jm_eval.retrieval.guard). Only
aggregate figures are written.

Gold standard: 50 infractions, 1,225 pairs scored 0-5 by three experts. The mean of the three scores
is recomputed here from the experts' columns.

- Rater reliability: ICC(2,1), ICC(2,3), Krippendorff's alpha, pairwise Spearman correlations.
- Methods, all local, on the infraction texts: M1 TF-IDF, M2 BM25, M6 BGE-M3, M7 multilingual-E5-large.
- Metrics: mAR and Recall@5 with binary relevance (mean score > 0), as in the 2024 paper, and
  nDCG@10 with the mean score as the gain. Bootstrap 95% intervals over the 50 cases.
- Matched subset (illustrative, small n): the infractions whose process also has a public CRSNSP
  decision in version 2's corpus. For those pairs, M3 (version 1) and M4 (version 2) are scored
  with the vectors of the PUBLIC CRSNSP texts, next to the local methods on the SUSEP texts.

Outputs (never overwritten): results/a1_retrieval_susep.csv, results/a1_susep_reliability.json
"""

from __future__ import annotations

from jm_eval.retrieval import guard

guard.enforce_local_only()          # before anything else is imported or any private file is opened

import argparse  # noqa: E402
import json  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import krippendorff  # noqa: E402
import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from jm_eval import stats  # noqa: E402
from jm_eval.io import load_config, new_artefact_path, resolve  # noqa: E402
from jm_eval.manifest import RunManifest  # noqa: E402
from jm_eval.retrieval import corpus as C  # noqa: E402
from jm_eval.retrieval import dense_local as D  # noqa: E402
from jm_eval.retrieval import gold as G  # noqa: E402
from jm_eval.retrieval import labels as L  # noqa: E402
from jm_eval.retrieval import lexical, metrics  # noqa: E402

SCRIPT = "47_a1_susep"
EXPERTS = ["Exp 1", "Exp 2", "Exp 3"]
LOCAL_METHODS = ["M1", "M2", "M6", "M7"]
PUBLIC_TEXT_METHODS = ["M3", "M4"]
METRICS = ["mAR", "Recall@5", "nDCG@10"]


def load_gold(private: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(infractions, pair scores with the recomputed mean, what the stored mean column holds)."""
    infractions = pd.read_excel(private / "similarity_experts_evaluation.xlsm", sheet_name="InfracoesData")
    infractions = infractions[["Número", "Texto"]].rename(columns={"Número": "id", "Texto": "text"})
    scores_path = private / "experts_evaluations_mean.xlsx"
    scores = pd.read_excel(scores_path)[["Inf A", "Inf B", *EXPERTS]]
    scores["mean"] = scores[EXPERTS].mean(axis=1)
    stored = stored_mean_check(scores_path, scores)
    return infractions, scores, stored


def stored_mean_check(path: Path, scores: pd.DataFrame) -> dict:
    """What the file's own `Exp Mean` column holds, against the mean recomputed from the experts."""
    def column(data_only: bool) -> list:
        ws = openpyxl.load_workbook(path, data_only=data_only).worksheets[0]
        col = [c.value for c in ws[1]].index("Exp Mean") + 1
        return [ws.cell(row=r, column=col).value for r in range(2, ws.max_row + 1)]

    formulas, cached = column(False), column(True)
    is_average = [isinstance(f, str) and f.upper().startswith("=AVERAGE(D") and f.upper().endswith(f"F{r})")
                  for r, f in enumerate(formulas, 2)]
    values = pd.Series([v for v in cached if isinstance(v, (int, float))], dtype=float)
    out = {"rows": len(formulas), "cells_with_a_row_average_formula": int(sum(is_average)),
           "cells_with_a_stored_value": int(len(values))}
    if len(values) == len(scores):
        out["max_abs_difference_from_recomputed_mean"] = float((values.to_numpy() - scores["mean"].to_numpy()).__abs__().max())
    return out


def reliability(scores: pd.DataFrame) -> dict:
    ratings = scores[EXPERTS].to_numpy(dtype=float)
    spearman = {f"{a} vs {b}": round(float(sps.spearmanr(scores[a], scores[b]).statistic), 4)
                for a, b in combinations(EXPERTS, 2)}
    icc = G.icc2(ratings)
    return {
        "pairs": len(scores), "experts": len(EXPERTS),
        "icc_2_1": round(icc["icc_2_1"], 4), "icc_2_3": round(icc["icc_2_k"], 4),
        "krippendorff_alpha_interval": round(float(krippendorff.alpha(ratings.T, level_of_measurement="interval")), 4),
        "krippendorff_alpha_ordinal": round(float(krippendorff.alpha(ratings.T, level_of_measurement="ordinal")), 4),
        "pairwise_spearman": spearman,
        "pairs_with_mean_above_zero": int((scores["mean"] > 0).sum()),
        "share_pairs_with_mean_above_zero": round(float((scores["mean"] > 0).mean()), 4),
        "pairs_above_zero_per_expert": {e: int((scores[e] > 0).sum()) for e in EXPERTS},
    }


def local_similarities(texts: list[str], cfg: dict) -> dict[str, np.ndarray]:
    counts = lexical.count_matrix(texts)
    out = {"M1": lexical.tfidf_similarity_from_counts(counts),
           "M2": lexical.bm25_scores_from_counts(counts, cfg["methods"]["M2"]["k1"], cfg["methods"]["M2"]["b"])}
    out["M2"] = (out["M2"] + out["M2"].T) / 2          # pairs are unordered: score a pair by both directions
    for method in ("M6", "M7"):
        spec = cfg["methods"][method]
        model = D.load_model(spec["model"], spec["max_seq_length"], offline=True)
        vectors, _ = D.encode(model, texts, spec.get("prefix", ""))
        out[method] = metrics.cosine_similarity(vectors)
        del model
    return out


def score(similarity: np.ndarray, gold: np.ndarray, cfg: dict, seed: int) -> list[dict]:
    per_case = G.per_case_metrics(similarity, gold)
    rows = []
    for metric in METRICS:
        values = per_case[metric].dropna().to_numpy()
        low, high = stats.bootstrap_mean_ci(values, resamples=cfg["bootstrap_resamples"], seed=seed)
        rows.append({"metric": metric, "cases": len(values), "mean": float(values.mean()), "ci_low": low, "ci_high": high})
    rows.append({"metric": "Spearman (pairs)", "cases": int(np.isfinite(gold[np.triu_indices(len(gold), 1)]).sum()),
                 "mean": G.pair_spearman(similarity, gold), "ci_low": np.nan, "ci_high": np.nan})
    return rows


def match_public(private: Path, ids: list) -> dict:
    """SUSEP infraction id -> row of the public CRSNSP decision of the same process (version 2's corpus)."""
    data = pd.read_csv(private / "data.csv", usecols=["infracaoId", "processoNum"]).drop_duplicates("infracaoId")
    process = dict(zip(data["infracaoId"], data["processoNum"].astype(str)))
    labels = C.load_labels()
    owner: dict[str, list[int]] = {}
    for row, keys in enumerate(labels["process_keys"]):
        for key in keys:
            owner.setdefault(key, []).append(row)
    matched = {}
    for i in ids:
        rows = sorted({r for m in L.PROC_RE.findall(process.get(i, "")) for r in owner.get("".join(m), [])})
        if rows:
            matched[i] = rows[0]                        # a process with two decisions: the first in hash order
    return matched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg, paths = C.methods_config(), load_config("paths")
    seed = load_config("seeds")["bootstrap"]
    private, results = resolve(paths["data"]["private"]), resolve(paths["results"])
    labels = {m: cfg["methods"][m]["label"] for m in cfg["methods"]}
    guard_state = {"cloud_modules_loaded": guard.loaded_cloud_modules(), "proxy_variables": guard.proxy_settings(),
                   "network_blocked": True, "hf_offline": True}
    with RunManifest(SCRIPT, results, models={m: cfg["methods"][m] for m in LOCAL_METHODS + PUBLIC_TEXT_METHODS},
                     params={"guard": guard_state, "bootstrap_resamples": cfg["bootstrap_resamples"]},
                     seeds={"bootstrap": seed}) as run:
        for name in ("similarity_experts_evaluation.xlsm", "experts_evaluations_mean.xlsx", "data.csv"):
            run.add_input(private / name)
        infractions, scores, stored = load_gold(private)
        ids = infractions["id"].tolist()
        gold = G.pair_matrix(ids, scores["Inf A"].to_numpy(), scores["Inf B"].to_numpy(), scores["mean"].to_numpy())
        rel = reliability(scores)
        rel["stored_mean_column"] = stored
        rel["infractions"] = len(ids)
        rel["distinct_texts"] = int(infractions["text"].nunique())

        similarities = local_similarities(infractions["text"].fillna("").tolist(), cfg)
        rows = []
        for method in LOCAL_METHODS:
            for r in score(similarities[method], gold, cfg, seed):
                rows.append({"set": "gold standard (50 infractions)", "method": method, "label": labels[method],
                             "text": "SUSEP infraction text", **r})

        matched = match_public(private, ids)
        position = [ids.index(i) for i in matched]
        rel["matched_subset"] = {"infractions": len(matched), "pairs": len(matched) * (len(matched) - 1) // 2,
                                 "pairs_with_mean_above_zero": None}
        if len(matched) >= 3:
            sub_gold = gold[np.ix_(position, position)]
            upper = sub_gold[np.triu_indices(len(position), 1)]
            rel["matched_subset"]["pairs_with_mean_above_zero"] = int((upper > 0).sum())
            public_rows = np.array(list(matched.values()))
            name = f"matched subset ({len(matched)} infractions)"
            for method in LOCAL_METHODS:
                for r in score(similarities[method][np.ix_(position, position)], sub_gold, cfg, seed):
                    rows.append({"set": name, "method": method, "label": labels[method], "text": "SUSEP infraction text", **r})
            for method in PUBLIC_TEXT_METHODS:
                similarity = metrics.cosine_similarity(C.method_vectors(method)[public_rows])
                for r in score(similarity, sub_gold, cfg, seed):
                    rows.append({"set": name, "method": method, "label": labels[method],
                                 "text": "public CRSNSP decision of the same process", **r})

        table = pd.DataFrame(rows)
        table_path = new_artefact_path(results / "a1_retrieval_susep.csv")
        table.to_csv(table_path, index=False, float_format="%.6g")
        run.add_output(table_path, "A1 (SUSEP gold standard): aggregate metrics per method")
        rel_path = new_artefact_path(results / "a1_susep_reliability.json")
        rel_path.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(rel_path, "A1 (SUSEP gold standard): rater reliability and the stored mean column")
        run.data["summary"] = {"reliability": rel, "guard": guard_state}
        print(json.dumps(rel, indent=1))
        print(table.pivot_table(index=["set", "method"], columns="metric", values="mean", sort=False).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
