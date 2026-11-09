import numpy as np
import pandas as pd
import pytest
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from jm_eval import stats
from jm_eval.retrieval import labels as L
from jm_eval.retrieval import lexical, metrics


# ---------------------------------------------------------------- statistics
def test_bootstrap_interval_covers_the_mean_and_is_repeatable():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=(400, 2))
    low, high = stats.bootstrap_mean_ci(x, resamples=2000, seed=1)
    assert np.all(low < x.mean(axis=0)) and np.all(x.mean(axis=0) < high)
    assert np.allclose(high - low, 2 * 1.96 * 0.1 / 20, rtol=0.2)             # about 2 x 1.96 standard errors
    again = stats.bootstrap_mean_ci(x, resamples=2000, seed=1)
    assert np.array_equal(low, again[0]) and np.array_equal(high, again[1])
    one = stats.bootstrap_mean_ci(x[:, 0], resamples=2000, seed=1)
    assert one == (pytest.approx(low[0]), pytest.approx(high[0]))             # same draws for every column
    assert all(np.isnan(v) for v in stats.bootstrap_mean_ci(np.array([]), resamples=10))


def test_randomisation_test_and_holm():
    rng = np.random.default_rng(0)
    base = rng.random(300)
    diff, p = stats.paired_randomisation_test(base + 0.05, base, permutations=2000)
    assert diff == pytest.approx(0.05) and p == pytest.approx(1 / 2001)       # a constant gain is never matched
    _, p_null = stats.paired_randomisation_test(base, rng.permutation(base), permutations=2000)
    assert p_null > 0.05
    _, p_same = stats.paired_randomisation_test(base, base, permutations=200)
    assert p_same == 1.0
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------- ranking metrics
def test_per_query_metrics_on_a_hand_checked_ranking():
    # query 0 ranks 1, 2, 3, 4 in that order; 1 and 3 are relevant
    sim = np.zeros((5, 5))
    sim[0] = [9, 4, 3, 2, 1]
    rel = np.zeros((5, 5), dtype=bool)
    rel[0, [1, 3]] = True
    allowed = metrics.candidate_mask(np.arange(5))
    m = metrics.per_query_metrics(sim, rel, allowed)
    assert list(m.index) == [0]                                               # only queries with a relevant decision
    row = m.loc[0]
    assert row["P@5"] == pytest.approx(2 / 5) and row["P@10"] == pytest.approx(2 / 10)
    assert row["R@10"] == 1.0 and row["MRR"] == 1.0 and row["R@5"] == 1.0
    assert row["mAR"] == pytest.approx((0.5 + 0.5 + 1.0 + 1.0) / 4)             # Recall@k for k = 1..4 candidates
    assert row["MAP"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    dcg, ideal = 1 / np.log2(2) + 1 / np.log2(4), 1 / np.log2(2) + 1 / np.log2(3)
    assert row["nDCG@10"] == pytest.approx(dcg / ideal)


def test_same_process_decisions_are_never_ranked():
    sim = np.array([[1.0, 0.99, 0.2, 0.1], [0.99, 1.0, 0.2, 0.1], [0.2, 0.2, 1.0, 0.5], [0.1, 0.1, 0.5, 1.0]])
    groups = np.array([0, 0, 1, 2])                                           # 0 and 1: same process
    allowed = metrics.candidate_mask(groups)
    assert metrics.rank_candidates(sim, allowed)[0, :2].tolist() == [2, 3]
    rel = np.zeros((4, 4), dtype=bool)
    rel[0, 1] = rel[0, 3] = True                                              # 1 is relevant by label but excluded
    m = metrics.per_query_metrics(sim, rel, allowed)
    assert m.loc[0, "MRR"] == pytest.approx(1 / 2) and m.loc[0, "R@10"] == 1.0
    assert m.loc[0, "mAR"] == pytest.approx((0.0 + 1.0) / 2)                    # two candidates: 2, then 3


def test_neighbour_agreement_observed_and_expected():
    vectors = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1], [0.1, 0.9], [0.2, 0.8]], dtype=float)
    sim = metrics.cosine_similarity(vectors)
    classes = np.array(["a", "a", "a", "b", "b", None], dtype=object)
    out = metrics.neighbour_agreement(sim, metrics.candidate_mask(np.arange(6)), classes, k=2)
    assert 5 not in out.index and out.loc[0, "observed"] == 1.0               # unknown class: not a query
    assert out.loc[0, "expected"] == pytest.approx(2 / 4)                     # 2 of the 4 known candidates
    assert out.loc[4, "observed"] == 1.0                                      # neighbours 5 (unknown, left out) and 3


# ---------------------------------------------------------------- lexical baselines
DOCS = ["A seguradora enviou o FIP fora do prazo à SUSEP.",
        "Envio do FIP com atraso pela seguradora; multa aplicada.",
        "Negativa de pagamento de indenização do seguro de vida.",
        "A corretora não repassou os prêmios à seguradora.",
        "Atraso no pagamento da indenização; seguro de automóvel.",
        "Provisões técnicas constituídas de forma inadequada."]


def test_tokens_are_stemmed_without_stop_words():
    tokens = lexical.tokenize("As seguradoras enviaram os FIPs fora do prazo, art. 88")
    assert "as" not in tokens and "os" not in tokens and "88" in tokens
    assert lexical.tokenize("seguradoras")[0] == lexical.tokenize("seguradora")[0]
    assert lexical.tokenize(None) == []


def test_bm25_matches_rank_bm25():
    tokens = [lexical.tokenize(d) for d in DOCS]
    reference = BM25Okapi(tokens, k1=1.5, b=0.75)
    expected = np.vstack([reference.get_scores(q) for q in tokens])
    assert np.allclose(lexical.bm25_scores(DOCS), expected)
    # a subset of a larger count matrix scores as a fit on the subset alone
    counts = lexical.count_matrix(DOCS)
    subset = [0, 2, 3, 5]
    ref_subset = BM25Okapi([tokens[i] for i in subset], k1=1.5, b=0.75)
    assert np.allclose(lexical.bm25_scores_from_counts(counts[subset]),
                       np.vstack([ref_subset.get_scores(tokens[i]) for i in subset]))


def test_tfidf_matches_sklearn_and_ranks_the_similar_report_first():
    sim = lexical.tfidf_similarity(DOCS)
    matrix = TfidfVectorizer(analyzer=lexical.tokenize).fit_transform(DOCS)
    assert np.allclose(sim, (matrix @ matrix.T).toarray())
    order = metrics.rank_candidates(sim, metrics.candidate_mask(np.arange(len(DOCS))))
    assert order[0, 0] == 1 and order[2, 0] == 4


# ---------------------------------------------------------------- relevance definitions
def test_build_relevance_definitions():
    labels = pd.DataFrame({
        "process_group": [0, 1, 2, 3, 0],
        "base_specific": [["a", "b"], ["a", "b"], ["a"], [], ["a", "b"]],
        "text_cited": [["x"], ["y"], ["x", "z"], ["q"], ["x"]],
        "text_primary": ["p1", "p1", None, None, "p1"],
        "category": ["C05", "C02", "C05", None, "C05"],
    })
    strict = L.build_relevance(labels, {"source": "base_specific", "kind": "exact"}, 0.5)
    assert strict[0, 1] and not strict[0, 2] and not strict[0, 4]             # 4 is the same process as 0
    relaxed = L.build_relevance(labels, {"source": "base_specific", "kind": "jaccard"}, 0.5)
    assert relaxed[0, 2] and not relaxed[3, 2]
    prov = L.build_relevance(labels, {"source": "text_cited", "kind": "jaccard", "or_same": "text_primary"}, 0.5)
    assert prov[0, 2] and prov[0, 1] and not prov[1, 2] and not prov[2, 3]    # Jaccard 0.5; same primary; neither
    labels["categories"] = [["C05", "C09"], ["C02"], ["C05"], None, ["C09", "C02"]]
    multi = L.build_relevance(labels, {"source": "categories", "kind": "overlap"}, 0.5)
    assert multi[0, 2] and multi[1, 4] and not multi[0, 1] and not multi[0, 4] and not multi[3, 0]   # 0 and 4: same process
    cat = L.build_relevance(labels, {"source": "category", "kind": "same"}, 0.5)
    assert cat[0, 2] and not cat[0, 1] and not cat[3, 3] and not cat[0, 4] and not cat[3, 0]


# ---------------------------------------------------------------- scripts 43, 45, 46
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stability = _load("compression_stability", "scripts/45_compression_stability.py")
retrieval = _load("a1_retrieval", "scripts/46_a1_retrieval.py")


def test_input_length_table_counts_truncated_inputs():
    # script 43 imports the cloud client; only its table function is exercised here
    cloud = _load("a1_cloud_embeddings", "scripts/43_a1_cloud_embeddings.py")
    inputs = pd.DataFrame({"acordao_hash": ["a", "b", "c", "d"], "method": "M5", "model": "m", "text": "report",
                           "chars": [4000, 8000, 400, 12000], "token_count": [1000, 3000, 100, 4096],
                           "truncated": [False, True, False, True]})
    table = cloud.input_length_table(inputs, shared={"a", "b"}, limit=2048).set_index("scope")
    assert table.at["v2", "inputs"] == 4 and table.at["v2", "share_truncated"] == 0.5
    assert table.at["comparison_set", "truncated"] == 1
    assert table.at["v2", "share_of_tokens_read"] == pytest.approx((1000 + 2048 + 100 + 2048) / 8196, abs=1e-4)


def test_compression_rerun_comparison():
    rng = np.random.default_rng(0)
    frozen = rng.normal(size=(30, 8))
    frozen /= np.linalg.norm(frozen, axis=1, keepdims=True)
    same = stability.compare(frozen, frozen[[3, 7]] * 2.5, [3, 7], k=5)            # the same direction: nothing moves
    assert np.allclose(same["cosine"], 1.0) and same["neighbours_kept"].tolist() == [5, 5]
    assert same["first_neighbour_kept"].all()
    moved = stability.compare(frozen, -frozen[[3]], [3], k=5)                     # the opposite direction
    assert moved.loc[0, "cosine"] == pytest.approx(-1.0) and moved.loc[0, "neighbours_kept"] == 0


def test_summarise_gives_intervals_tests_and_contrasts():
    rng = np.random.default_rng(1)
    base = pd.DataFrame(rng.random((200, len(metrics.METRICS))) * 0.5, columns=metrics.METRICS)
    per_query = {"M2": base, "M3": base + 0.1, "M4": base + 0.1, "M5": base + rng.normal(0, 0.01, base.shape)}
    cfg = {"bootstrap_resamples": 500, "randomisation_permutations": 500}
    main, contrasts = retrieval.summarise(per_query, "M2", cfg, seed=1)
    assert len(main) == 4 * len(metrics.METRICS) and (main["ci_low"] <= main["mean"]).all() and (main["mean"] <= main["ci_high"]).all()
    m3 = main[(main["method"] == "M3") & (main["metric"] == "MAP")].iloc[0]
    assert m3["diff_vs_baseline"] == pytest.approx(0.1) and m3["p_vs_baseline"] == pytest.approx(1 / 501)
    assert m3["p_holm_vs_baseline"] >= m3["p_vs_baseline"]
    assert main[main["method"] == "M2"]["p_vs_baseline"].isna().all()               # the baseline is not tested
    v2_v1 = contrasts[(contrasts["contrast"] == "M4 - M3") & (contrasts["metric"] == "MAP")].iloc[0]
    assert v2_v1["diff"] == pytest.approx(0.0, abs=1e-12) and v2_v1["p"] == 1.0
    assert set(contrasts["contrast"]) == {"M4 - M3", "M4 - M5"}                     # M4b was not run

    sets = pd.Series([["a"]] * 120 + [["a", "b"]] * 70 + [None] * 10)              # one, several, unknown
    groups = retrieval.by_complexity(per_query, sets, "M2", cfg, 1, "v2", "R-base")
    assert [(t["infractions"].iloc[0], int(t["queries"].iloc[0])) for t, _ in groups] == [("one", 120), ("several", 70)]
    assert all(set(c["contrast"]) == {"M4 - M3", "M4 - M5"} for _, c in groups)
