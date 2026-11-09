"""Local multilingual NLI check of claims against evidence passages (Phase 6, task 5.3). No API call.

Model: config/a3_chatbot.yaml `faithfulness.nli_model` (mDeBERTa-v3 XNLI), run on this machine from
data/cache/hf. For each claim, each evidence passage is the premise and the claim the hypothesis.
A claim is "supported" if any passage entails it above the threshold, otherwise "contradicted" if
any passage contradicts it above the threshold, otherwise "not_found".
"""

from __future__ import annotations

import numpy as np

from jm_eval.retrieval.dense_local import models_dir, set_offline


def load(name: str, *, offline: bool = True, device: str | None = None):
    """(tokenizer, model, device); with `offline=False` the files are downloaded once into data/cache/hf."""
    if offline:
        set_offline()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cache = str(models_dir())
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache, local_files_only=offline)
    model = AutoModelForSequenceClassification.from_pretrained(name, cache_dir=cache, local_files_only=offline)
    return tokenizer, model.to(device).eval(), device


def probabilities(nli, premises: list[str], hypotheses: list[str], batch_size: int = 16) -> np.ndarray:
    """(pairs, 3) probabilities in the order entailment, neutral, contradiction."""
    import torch

    tokenizer, model, device = nli
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    order = [labels["entailment"], labels["neutral"], labels["contradiction"]]
    out = np.zeros((len(premises), 3), dtype=np.float64)
    for start in range(0, len(premises), batch_size):
        batch = tokenizer(premises[start:start + batch_size], hypotheses[start:start + batch_size],
                          truncation="only_first", max_length=512, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**batch).logits.float().cpu().numpy()
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        out[start:start + batch_size] = (exp / exp.sum(axis=1, keepdims=True))[:, order]
    return out


def claim_label(pair_probabilities: np.ndarray, threshold: float = 0.5) -> str:
    """Label of one claim from the probabilities of its (passage, claim) pairs."""
    if len(pair_probabilities) == 0:
        return "not_found"
    if pair_probabilities[:, 0].max() >= threshold:
        return "supported"
    if pair_probabilities[:, 2].max() >= threshold:
        return "contradicted"
    return "not_found"
