"""Evidence passages for the claims of an answer (Phase 6, task 5.2). Local and free: no API call.

The decisions of a case set are split into passages of about 150 words, from every field the
chatbot received. For each claim the passages are ranked by reciprocal-rank fusion of a dense ranking
(BGE-M3) and BM25: a claim that names decisions gets the best passages of those decisions, any
other claim the best passages of the whole set.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from jm_eval.chatbot import scoring as S
from jm_eval.chatbot.judging import field_text
from jm_eval.retrieval import lexical
from jm_eval.retrieval.labels import process_keys

# Short fields go into one passage per decision; long fields are split.
HEADER_FIELDS = [("Ano da sessão", "ano_sessoes"), ("Número da sessão", "sessao_nr_int"),
                 ("Datas da sessão", "sessao_datas"), ("Recorrente", "acordao_recorrente"),
                 ("Recorrido", "acordao_recorrido"), ("Conselheiro Relator", "acordao_relator"),
                 ("Advogado do recorrente", "acordao_advogado"), ("Penalidade", "acordao_penalidade"),
                 ("Base legal", "acordao_base"), ("Resultado", "acordao_resultado")]
LONG_FIELDS = [("ementa", "acordao_ementa"), ("acórdão", "acordao_acordao"),
               ("participantes", "acordao_participantes"), ("relatório", "acordao_relatorio"),
               ("voto do relator", "acordao_voto_relator")]


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Windows of `size` words that share `overlap` words with the next one."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def build_passages(rows: pd.DataFrame, size: int = 150, overlap: int = 30) -> pd.DataFrame:
    """Passages of the decisions in `rows`: acordao_hash, title, field, text (with a title and field prefix)."""
    out = []
    for row in rows.to_dict("records"):
        title = field_text(row["acordao_titulo"])
        header = "; ".join(f"{label}: {field_text(row[field])}" for label, field in HEADER_FIELDS)
        out.append({"acordao_hash": row["acordao_hash"], "title": title, "field": "dados do caso", "body": header})
        for label, field in LONG_FIELDS:
            for chunk in split_words(field_text(row[field]), size, overlap):
                out.append({"acordao_hash": row["acordao_hash"], "title": title, "field": label, "body": chunk})
    passages = pd.DataFrame(out)
    passages["text"] = "[" + passages["title"] + " | " + passages["field"] + "] " + passages["body"]
    passages.insert(0, "passage_id", range(len(passages)))
    return passages


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion score of each item; `rankings` are score vectors (higher is better)."""
    fused = np.zeros(len(rankings[0]), dtype=np.float64)
    for scores in rankings:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    return fused


def bm25_query_scores(passages: list[str], queries: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """scores[q, p]: BM25 score of passage p for query q (Okapi, as rank_bm25 defines it)."""
    from rank_bm25 import BM25Okapi

    index = BM25Okapi([lexical.tokenize(p) for p in passages], k1=k1, b=b)
    return np.vstack([index.get_scores(lexical.tokenize(q)) for q in queries]) if queries else np.zeros((0, len(passages)))


def cited_hashes(processes: list[str], claim: str, rows: pd.DataFrame) -> list[str]:
    """Decisions of the set that a claim names, by the numbers in its `processes` or in its own text.

    A decision is named by a process or appeal number of its title, or by its acórdão number (the
    first one of its `acordao_acordao` field): answers use either.
    """
    text = " ".join(processes) + " " + claim
    named = set(S.process_numbers(text))
    acordaos = set(S.acordao_numbers(text)) | {f"A{int(a)}/{b}" for a, b in re.findall(r"(?<![\d/.])(\d{1,5})\s*/\s*(\d{4})(?![\d-])", " ".join(processes))}
    own = rows["acordao_acordao"].map(S.own_acordao_number) if "acordao_acordao" in rows else [None] * len(rows)
    return [h for h, title, number in zip(rows["acordao_hash"], rows["acordao_titulo"], own)
            if process_keys(field_text(title)) & named or (number is not None and number in acordaos)]


def select_evidence(dense: np.ndarray, bm25: np.ndarray, passages: pd.DataFrame, cited: list[str], *, top: int = 4,
                    named: dict[int, int] | None = None, beyond: int = 1, limit: int = 6, rrf_k: int = 60) -> list[int]:
    """Passage rows for one claim, best first.

    A claim that names decisions gets the best passages of each of them (`named`: passages per
    decision by the number of decisions named, `beyond` when more are named), at most `limit` in
    all. A claim that names none gets the `top` passages of the whole set.
    """
    fused = rrf([dense, bm25], rrf_k)
    order = np.argsort(-fused, kind="stable")
    if not cited:
        return [int(i) for i in order[:top]]
    per_decision = (named or {1: 4, 2: 2, 3: 2}).get(len(cited), beyond)
    hashes = passages["acordao_hash"].to_numpy()
    chosen: list[int] = []
    for h in cited:
        chosen += [int(i) for i in order if hashes[i] == h][:per_decision]
    return sorted(chosen, key=lambda i: -fused[i])[:limit] if len(chosen) > limit else chosen


def most_likely_decision(fused_top: list[int], passages: pd.DataFrame, cited: list[str]) -> str:
    """The decision a claim most likely comes from: the one it names, else that of its best passage."""
    return cited[0] if cited else str(passages["acordao_hash"].iloc[fused_top[0]])
