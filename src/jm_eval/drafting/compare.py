"""Phase 8 (A4): two versions of the drafting pipeline compared on the same notices.

The unit is the test notice: each version's written drafts of a notice (every mode and run) are averaged,
and the two versions' notice means are compared with the Wilcoxon signed-rank test, as the ablation tests of
`scripts/88` are. Lost drafts are left out here (the comparison is of what each version writes; the lost
drafts are counted separately), and so is the draft that cites its own source decision (`leaked`), which is
left out of the rubric and of argument recall throughout Phase 8.

Invented case facts per draft are estimated as in `scripts/90`: the classifier's passages of each kind times
the share of that kind the version's blind audit found invented (the audit's confusion matrix).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FACT_KINDS = ["invented", "assertion", "procedural", "in_notice"]


def invented_shares(confusion: pd.DataFrame) -> dict[str, float]:
    """P(invented | classifier kind) from an audit's confusion matrix (rows: classifier kind; columns: audit)."""
    return {k: float(confusion.loc[k, "invented"] / confusion.loc[k].sum()) for k in FACT_KINDS}


def invented_per_draft(kind_rows: Iterable[dict], shares: dict[str, float]) -> pd.DataFrame:
    """One row per classified draft: notice_id, mode, run and its estimated invented case facts."""
    return pd.DataFrame([{"notice_id": r["notice_id"], "mode": r["mode"], "run": int(r["run"]),
                          "estimated_invented": sum((r.get("kinds") or []).count(k) * shares[k] for k in FACT_KINDS)}
                         for r in kind_rows])


def written(scores: pd.DataFrame) -> pd.DataFrame:
    """Drafts that were written and do not cite their own source decision."""
    return scores[~scores["lost"].astype(bool) & ~scores["leaked"].astype(bool)]


def paired_by_notice(a: pd.DataFrame, b: pd.DataFrame, metrics: dict[str, str]) -> pd.DataFrame:
    """Per metric: both versions' means over notices, the pooled draft means, and the Wilcoxon test.

    `metrics` maps a column of both score tables to its description. A notice enters a metric's test only if
    both versions have a value for it (e.g. the accepted argument exists only for granted notices).
    """
    rows = []
    for column, label in metrics.items():
        per_a = a.groupby("notice_id")[column].mean()
        per_b = b.groupby("notice_id")[column].mean()
        joined = pd.concat([per_a, per_b], axis=1, keys=["a", "b"]).dropna()
        differs = bool((joined["a"] != joined["b"]).any())
        test = wilcoxon(joined["a"], joined["b"]) if differs else None
        rows.append({
            "metric": column, "label": label, "notices": len(joined),
            "drafts_a": int(a[column].notna().sum()), "drafts_b": int(b[column].notna().sum()),
            "mean_a": joined["a"].mean(), "mean_b": joined["b"].mean(),
            "difference": joined["b"].mean() - joined["a"].mean(),
            "notices_higher_in_b": int((joined["b"] > joined["a"]).sum()),
            "notices_lower_in_b": int((joined["b"] < joined["a"]).sum()),
            "pooled_mean_a": a[column].mean(), "pooled_mean_b": b[column].mean(),
            "wilcoxon_statistic": float(test.statistic) if test else np.nan,
            "wilcoxon_p": float(test.pvalue) if test else 1.0,
        })
    return pd.DataFrame(rows)
