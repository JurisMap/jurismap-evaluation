"""The corpus of the A1 benchmark and where each method's vectors are kept.

Everything is aligned to the corpus sorted by `acordao_hash`. Vectors computed for the benchmark
live in data/v2/a1_embeddings/<method>.npy (float32, one row per decision of that order).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from jm_eval.io import latest_artefact_path, load_config, resolve

METHODS_CONFIG = "a1_methods"


def methods_config() -> dict:
    return load_config(METHODS_CONFIG)


def embeddings_dir() -> Path:
    return resolve(load_config("paths")["data"]["v2"]) / "a1_embeddings"


def load_corpus(columns: list[str]) -> pd.DataFrame:
    """The frozen v2 corpus sorted by `acordao_hash`, with the requested columns."""
    corpus_path = resolve(load_config("a1_labels")["corpus"])
    wanted = list(dict.fromkeys(["acordao_hash", *columns]))
    return pd.read_parquet(corpus_path, columns=wanted).sort_values("acordao_hash").reset_index(drop=True)


def load_texts(text_key: str) -> tuple[list[str], list[str]]:
    """(hashes, texts) of one of the `texts` of config/a1_methods.yaml."""
    column = methods_config()["texts"][text_key]
    corpus = load_corpus([column])
    return corpus["acordao_hash"].tolist(), corpus[column].fillna("").tolist()


def stored_vectors(column: str) -> np.ndarray:
    corpus = load_corpus([column])
    return np.vstack(corpus[column].to_numpy()).astype(np.float32)


def method_vectors(method: str) -> np.ndarray:
    """Vectors of a dense method, one row per decision in `acordao_hash` order."""
    spec = methods_config()["methods"][method]
    if spec["kind"] == "stored":
        return stored_vectors(spec["column"])
    if spec["kind"] == "fused":
        from jm_eval.retrieval.candidates import combine
        return combine({part: method_vectors(part) for part in spec["parts"]}, spec["parts"]).astype(np.float32)
    path = latest_artefact_path(embeddings_dir() / f"{method}.npy")
    if not path.exists():
        raise FileNotFoundError(f"{path}: run scripts/43_a1_cloud_embeddings.py or scripts/44_a1_local_embeddings.py")
    return np.load(path)


def load_labels() -> pd.DataFrame:
    """Silver labels in `acordao_hash` order, with the infraction categories of each labelling that exists."""
    labels_dir = resolve(load_config("paths")["data"]["labels"])
    labels = pd.read_parquet(latest_artefact_path(labels_dir / "a1_silver_labels.parquet"))
    for column, name in (("category", "a1_infraction_category.parquet"),
                         ("category_mistral", "a1_infraction_category_mistral.parquet")):
        category_path = latest_artefact_path(labels_dir / name)
        sets = column.replace("category", "categories")          # primary and secondary categories together
        if category_path.exists():
            category = pd.read_parquet(category_path, columns=["acordao_hash", "primary", "secondary"])
            category[sets] = [[p, *s] if isinstance(p, str) else None for p, s in zip(category["primary"], category["secondary"])]
            category = category.rename(columns={"primary": column}).drop(columns=["secondary"])
            labels = labels.merge(category, on="acordao_hash", how="left")
        else:
            labels[column], labels[sets] = None, None
    return labels.sort_values("acordao_hash").reset_index(drop=True)
