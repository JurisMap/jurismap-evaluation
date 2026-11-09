"""Metrics of the A3 chatbot evaluation (Phase 6, tasks 5.4 to 7).

Per verifier (the NLI model, each judge, their majority):
  faithfulness       supported claims / all claims
  contradiction rate contradicted claims / all claims
  hallucination rate answers with at least one contradicted or not-found claim / answers
The retrieval-miss control re-judges a sample of the not-found claims against the full decision;
the share that turns out supported is the miss rate, and the corrected figures use it.
"""

from __future__ import annotations

from collections import Counter

import krippendorff
import numpy as np
import pandas as pd

LABELS = ["supported", "contradicted", "not_found"]
COVERAGE_VALUE = {"covered": 1.0, "partly": 0.5, "missing": 0.0}


def majority(labels: list[str | None]) -> str | None:
    """The label most verifiers give; None without a strict majority among those that answered."""
    given = [label for label in labels if label in LABELS]
    if not given:
        return None
    label, n = Counter(given).most_common(1)[0]
    return label if n * 2 > len(given) else None


def faithfulness(claims: pd.DataFrame, column: str) -> dict:
    """Raw figures of one verifier. `claims` has answer_id and the verifier's label column."""
    judged = claims[claims[column].isin(LABELS)]
    answers = judged["answer_id"].nunique()          # a verifier may have judged only some answers (first runs)
    bad = judged[judged[column] != "supported"].groupby("answer_id").size()
    n = len(judged)
    return {
        "claims": int(len(claims)), "claims_judged": int(n), "answers": int(answers),
        "supported": int((judged[column] == "supported").sum()),
        "contradicted": int((judged[column] == "contradicted").sum()),
        "not_found": int((judged[column] == "not_found").sum()),
        "faithfulness": float((judged[column] == "supported").mean()) if n else float("nan"),
        "contradiction_rate": float((judged[column] == "contradicted").mean()) if n else float("nan"),
        "not_found_rate": float((judged[column] == "not_found").mean()) if n else float("nan"),
        "hallucination_rate": float(len(bad) / answers) if answers else float("nan"),
        "answers_with_a_contradicted_claim": int(judged[judged[column] == "contradicted"]["answer_id"].nunique()),
    }


def miss_rate(control: pd.DataFrame) -> dict:
    """Share of the re-judged not-found claims that the full decision supports."""
    judged = control[control["full_text_label"].isin(LABELS)]
    flipped = int((judged["full_text_label"] == "supported").sum())
    return {"sampled": int(len(control)), "judged": int(len(judged)), "flipped_to_supported": flipped,
            "flipped_to_contradicted": int((judged["full_text_label"] == "contradicted").sum()),
            "miss_rate": flipped / len(judged) if len(judged) else float("nan")}


def with_control(claims: pd.DataFrame, column: str, control: pd.DataFrame, rate: float) -> pd.DataFrame:
    """The claims one verifier judged, with `p_supported` and `p_contradicted` after the control.

    A not-found claim that was re-judged takes the full-text verdict; one that was not re-judged
    counts as supported with probability `rate`.
    """
    judged = claims[claims[column].isin(LABELS)].drop(columns=["full_text_label"], errors="ignore")
    judged = judged.merge(control[["claim_id", "full_text_label"]], on="claim_id", how="left")
    label = judged[column]
    rejudged = (label == "not_found") & judged["full_text_label"].isin(LABELS)
    p_supported = np.where(label == "supported", 1.0, 0.0)
    p_supported = np.where((label == "not_found") & ~rejudged, rate if rate == rate else 0.0, p_supported)
    p_supported = np.where(rejudged, (judged["full_text_label"] == "supported").astype(float), p_supported)
    p_contradicted = np.where(label == "contradicted", 1.0, 0.0)
    p_contradicted = np.where(rejudged, (judged["full_text_label"] == "contradicted").astype(float), p_contradicted)
    return judged.assign(p_supported=p_supported, p_contradicted=p_contradicted)


def corrected(claims: pd.DataFrame, column: str, control: pd.DataFrame, rate: float) -> dict:
    """Figures of one verifier after the retrieval-miss control.

    The hallucination rate is the mean, over the answers, of the probability that at least one
    claim is contradicted or truly not found.
    """
    judged = with_control(claims, column, control, rate)
    p_supported, p_contradicted = judged["p_supported"].to_numpy(), judged["p_contradicted"].to_numpy()
    clean = judged.groupby("answer_id")["p_supported"].prod()            # P(every claim is supported)
    answers = judged["answer_id"].nunique()
    n = len(judged)
    return {"faithfulness_corrected": float(p_supported.sum() / n) if n else float("nan"),
            "contradiction_rate_corrected": float(p_contradicted.sum() / n) if n else float("nan"),
            "hallucination_rate_corrected": float((1.0 - clean).sum() / answers) if answers else float("nan")}


def nominal_alpha(a: pd.Series, b: pd.Series, values: list[str]) -> dict:
    """Krippendorff's alpha (nominal), raw agreement and n over the items both raters labelled."""
    both = a.isin(values) & b.isin(values)
    n = int(both.sum())
    if n == 0:
        return {"n": 0, "agreement": float("nan"), "alpha": float("nan")}
    codes = {v: i for i, v in enumerate(values)}
    data = np.array([[codes[v] for v in a[both]], [codes[v] for v in b[both]]], dtype=float)
    distinct = len(set(data.ravel()))
    alpha = float(krippendorff.alpha(data, level_of_measurement="nominal")) if distinct > 1 else float("nan")
    return {"n": n, "agreement": float((a[both] == b[both]).mean()), "alpha": alpha}


def interval_alpha(a: pd.Series, b: pd.Series, level: str = "interval") -> dict:
    both = a.notna() & b.notna()
    n = int(both.sum())
    if n < 2 or len(set(a[both]) | set(b[both])) < 2:
        return {"n": n, "alpha": float("nan")}
    data = np.array([a[both].to_numpy(dtype=float), b[both].to_numpy(dtype=float)])
    return {"n": n, "alpha": float(krippendorff.alpha(data, level_of_measurement=level))}


def coverage_score(coverage: list[str]) -> dict:
    """Coverage of the reference points: covered = 1, partly = 0.5, missing = 0; and the strict share."""
    values = [COVERAGE_VALUE[c] for c in coverage]
    return {"points": len(values), "coverage": float(np.mean(values)) if values else float("nan"),
            "coverage_strict": float(np.mean([c == "covered" for c in coverage])) if values else float("nan")}
