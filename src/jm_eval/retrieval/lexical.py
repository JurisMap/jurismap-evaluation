"""Lexical baselines of the A1 benchmark: TF-IDF (M1) and BM25 (M2). Local, no network at run time.

Tokens: lower-cased words and numbers, Portuguese stop words removed (NLTK list), RSLP stems.
The stop words and the RSLP rules are NLTK data files, downloaded once by scripts/44 into
data/cache/nltk_data; this module only reads them.

BM25 is Okapi as rank_bm25 defines it (k1 1.5, b 0.75, IDF log((N - n + 0.5) / (n + 0.5)), negative
IDFs replaced by 0.25 times the mean IDF, every occurrence of a query term counted). rank_bm25
scores one query at a time in Python loops; `bm25_scores` gives the same scores with sparse
matrices (tests/test_phase4_retrieval.py compares the two).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from jm_eval.io import load_config, resolve

TOKEN_RE = re.compile(r"[0-9a-zà-ÿ]+")


def nltk_data_dir() -> Path:
    return resolve(load_config("a1_methods")["nltk_data"])


@lru_cache(maxsize=1)
def _resources():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import RSLPStemmer

    path = str(nltk_data_dir())
    if path not in nltk.data.path:
        nltk.data.path.insert(0, path)
    try:
        return frozenset(stopwords.words("portuguese")), RSLPStemmer()
    except LookupError as e:
        raise RuntimeError(f"NLTK data missing in {path}: run scripts/44_a1_local_embeddings.py --setup-only") from e


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _resources()[1].stem(word)


def tokenize(text: str | None) -> list[str]:
    stop = _resources()[0]
    return [_stem(w) for w in TOKEN_RE.findall((text or "").lower()) if w not in stop]


def count_matrix(texts: list[str]) -> sparse.csr_matrix:
    """Term counts (documents x terms). Tokenising is the slow step: do it once and take row subsets."""
    return CountVectorizer(analyzer=tokenize, dtype=np.float64).fit_transform(texts).tocsr()


def _present_terms(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """Drop the terms that no document of this (sub)corpus contains, as a fit on it alone would."""
    return counts[:, np.flatnonzero(counts.getnnz(axis=0))].tocsr()


def tfidf_similarity_from_counts(counts: sparse.csr_matrix) -> np.ndarray:
    """Cosine similarity of TF-IDF vectors (sklearn defaults: smoothed IDF, L2 norm), fitted on these documents."""
    matrix = TfidfTransformer().fit_transform(_present_terms(counts))
    return np.asarray((matrix @ matrix.T).todense())


def tfidf_similarity(texts: list[str]) -> np.ndarray:
    return tfidf_similarity_from_counts(count_matrix(texts))


def bm25_scores_from_counts(counts: sparse.csr_matrix, k1: float = 1.5, b: float = 0.75,
                            epsilon: float = 0.25) -> np.ndarray:
    """scores[i, j]: BM25 score of document j for the query made of document i's tokens."""
    counts = _present_terms(counts)
    n_docs = counts.shape[0]
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
    idf[idf < 0] = epsilon * idf.mean()                      # rank_bm25: mean over all terms, before the floor
    weights = counts.tocoo(copy=True)
    norm = k1 * (1 - b + b * doc_len[weights.row] / doc_len.mean())
    weights.data = idf[weights.col] * weights.data * (k1 + 1) / (weights.data + norm)
    return np.asarray((counts @ weights.tocsr().T).todense())


def bm25_scores(texts: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    return bm25_scores_from_counts(count_matrix(texts), k1, b)
