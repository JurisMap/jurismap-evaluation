"""Krippendorff's ordinal alpha (Phase 8, agreement between the rubric judges)."""

from __future__ import annotations

from jm_eval.stats import krippendorff_alpha_ordinal as alpha


def test_alpha_ordinal_limits_and_known_value():
    assert alpha([[1, 1], [2, 2], [3, 3], [5, 5]]) == 1.0
    assert alpha([[1, 5], [5, 1], [1, 5], [5, 1]]) < 0
    assert alpha([[3, 3], [3, 3]]) is None                       # no variation: undefined
    assert alpha([[1, None], [2, 2], [3, 3]]) == 1.0              # a unit with one value is left out
    # Two raters, values 1-2-3 (hand-computed: n_1=2, n_2=4, n_3=2, n=8; delta^2(1,2) = (6 - 3)^2 = 9,
    # delta^2(2,3) = (6 - 3)^2 = 9, delta^2(1,3) = (8 - 2)^2 = 36; observed sum o*d = 4 * 9 = 36; expected
    # sum n_c*n_k*d = 2 * (72 + 144 + 72) = 576): alpha = 1 - 7 * 36 / 576 = 0.5625.
    assert abs(alpha([[1, 1], [2, 2], [2, 3], [3, 2]]) - 0.5625) < 1e-9


def test_project_only_routing_calls_the_named_project_and_no_other(monkeypatch):
    import requests
    from jm_eval import llm

    seen = []

    class Response:
        def raise_for_status(self):
            raise requests.HTTPError("403")

    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: (seen.append((url, kw["headers"])), Response())[1])
    monkeypatch.setattr(llm, "_google_token", lambda: "user-token")
    judge = llm.RawPredictJudge("judge_b", {"model": "mistral-medium-3", "region": "europe-west4",
                                            "credentials": "project_only", "project": "your-mistral-gcp-project"}, cache=None)
    import pytest
    with pytest.raises(requests.HTTPError):
        judge._call("x", None)
    assert len(seen) == 1 and "/projects/your-mistral-gcp-project/" in seen[0][0]   # no fallback to the fallback project
    assert seen[0][1]["Authorization"] == "Bearer user-token"
