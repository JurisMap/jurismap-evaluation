import numpy as np
import pytest

from jm_eval.io import load_config
from jm_eval.retrieval import candidates as K
from jm_eval.retrieval import corpus as C


# ---------------------------------------------------------------- chunks
def test_a_text_within_the_limit_is_one_unchanged_chunk():
    text = "Relatório.\n\nTrata-se de recurso.  "
    assert K.chunk_text(text, 100) == [text]                                  # the same string, so the same cached vector


def test_chunks_respect_the_limit_and_keep_every_word():
    paragraphs = [" ".join(f"palavra{i}_{j}." for j in range(40)) for i in range(12)]
    text = "\n\n".join(paragraphs) + "\n" + "x" * 950                         # the last paragraph has no place to cut
    chunks = K.chunk_text(text, 400)
    assert all(0 < len(c) <= 400 for c in chunks) and len(chunks) > 12
    assert "".join("".join(chunks).split()) == "".join(text.split())          # nothing lost, nothing repeated
    assert chunks[0].endswith(".")                                            # cut at a sentence end
    assert K.chunk_text(paragraphs[0] + "\n\n" + paragraphs[1], 1000) == [paragraphs[0], paragraphs[1]]   # at the paragraph end


def test_pool_and_combine_give_unit_vectors():
    chunks = np.array([[2.0, 0.0], [0.0, 5.0]])
    pooled = K.pool(chunks, np.array([3, 1]))                                 # chunk length as the weight, chunk norm ignored
    assert pooled == pytest.approx(np.array([3.0, 1.0]) / np.sqrt(10))
    parts = {"a": np.array([[1.0, 0.0], [0.0, 2.0]]), "b": np.array([[0.0, 3.0], [0.0, 1.0]])}
    fused = K.combine(parts, {"a": 0.5, "b": 0.5})
    assert fused == pytest.approx(np.array([[np.sqrt(0.5), np.sqrt(0.5)], [0.0, 1.0]]))
    assert K.combine(parts, {"a": 1.0}) == pytest.approx(K.unit_rows(parts["a"]))


# ---------------------------------------------------------------- split and score
def test_development_split_keeps_a_process_on_one_side():
    groups = np.repeat(np.arange(100), 3)
    mask = K.development_mask(groups, 0.5, seed=1)
    assert mask.sum() == 150 and all(len(set(mask[groups == g])) == 1 for g in range(100))
    assert np.array_equal(mask, K.development_mask(groups, 0.5, seed=1))
    assert not np.array_equal(mask, K.development_mask(groups, 0.5, seed=2))


def test_selection_score_averages_families_then_definitions():
    nan = np.nan
    values = {"base": np.array([[0.2, 0.4], [nan, nan], [0.4, 0.8]]),         # the second decision is not a query here
              "cat": np.array([[0.5, 0.5], [0.7, 0.9], [0.9, 0.7]]), "cat2": np.array([[0.1, 0.3], [0.1, 0.3], [0.1, 0.3]])}
    families = {"legal_basis": ["base"], "category": ["cat", "cat2"]}
    score = K.selection_score(values, families)
    assert score == pytest.approx([(0.3 + (0.7 + 0.1) / 2) / 2, (0.6 + (0.7 + 0.3) / 2) / 2])
    assert K.selection_score(values, families, rows=np.array([0, 0])) == pytest.approx([(0.2 + 0.3) / 2, (0.4 + 0.4) / 2])


def test_score_difference_interval_separates_a_real_gain_from_noise():
    rng = np.random.default_rng(0)
    base = rng.random((400, 1))
    values = {"d": np.hstack([base, base + 0.05, base + rng.normal(0, 0.2, (400, 1))])}
    families = {"f": ["d"]}
    diff, low, high = K.score_difference_interval(values, families, 1, 0, resamples=500, seed=1)
    assert diff == pytest.approx(0.05) and low == pytest.approx(0.05) and high == pytest.approx(0.05)   # a constant gain
    diff, low, high = K.score_difference_interval(values, families, 2, 0, resamples=500, seed=1)
    assert low < 0 < high
    assert K.score_difference_interval(values, families, 2, 0, resamples=500, seed=1) == (diff, low, high)


# ---------------------------------------------------------------- config
def test_candidates_config_is_consistent():
    cfg, methods = K.settings(), C.methods_config()
    assert cfg["current"] in cfg["candidates"] and cfg["candidates"][cfg["current"]] == {"compressed": 1.0}
    for weights in cfg["candidates"].values():
        assert set(weights) <= set(cfg["parts"]) and sum(weights.values()) == pytest.approx(1.0)
    assert {p["method"] for p in cfg["parts"].values()} <= set(methods["methods"])
    assert methods["methods"]["M5c"]["benchmark"] is False                    # not a row of the Phase 4 benchmark
    definitions = [d for family in cfg["selection"]["families"].values() for d in family]
    assert set(definitions) <= set(methods["relevance"]) and "R-base" in cfg["selection"]["families"]["legal_basis"]
    assert 0 < cfg["split"]["development_share"] < 1 and "a1_selection_split" in load_config("seeds")
