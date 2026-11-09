"""Local dense retrievers of the A1 benchmark (M6 BGE-M3, M7 multilingual-E5-large).

The models run on this machine through sentence-transformers. Their files are downloaded once by
scripts/44 into data/cache/hf; `offline=True` then loads them without any network access, which is
how the SUSEP script uses them (ground rule 1).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from jm_eval.io import load_config, resolve


def models_dir() -> Path:
    return resolve(load_config("a1_methods")["local_models_cache"])


def set_offline() -> None:
    """Make the Hugging Face libraries refuse network access (call before they are imported)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def load_model(name: str, max_seq_length: int, *, offline: bool, device: str | None = None):
    if offline:
        set_offline()
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(name, cache_folder=str(models_dir()), device=device, local_files_only=offline)
    model.max_seq_length = max_seq_length
    if device != "cpu":
        model.half()                      # long inputs fit in memory; cosine rankings are unaffected at this precision
    return model


def token_counts(model, texts: list[str], prefix: str = "") -> np.ndarray:
    """Tokens of each input as the model's tokenizer counts them, special tokens included, no truncation."""
    tokenizer = model.tokenizer
    return np.array([len(tokenizer(prefix + t, truncation=False, add_special_tokens=True)["input_ids"]) for t in texts])


def encode(model, texts: list[str], prefix: str = "", *, long_tokens: int = 1024,
           batch_size: int = 8, progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalised vectors and the token count of every input.

    Inputs longer than `long_tokens` are encoded one at a time (memory); the rest in batches.
    """
    counts = token_counts(model, texts, prefix)
    inputs = [prefix + t for t in texts]
    dimension = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()
    vectors = np.zeros((len(texts), dimension), dtype=np.float32)
    long = np.flatnonzero(counts > long_tokens)
    short = np.flatnonzero(counts <= long_tokens)
    for index, size in ((short, batch_size), (long, 1)):
        if len(index):
            out = model.encode([inputs[i] for i in index], batch_size=size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=progress)
            vectors[index] = out.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)     # exact unit length after half precision
    return vectors, counts
