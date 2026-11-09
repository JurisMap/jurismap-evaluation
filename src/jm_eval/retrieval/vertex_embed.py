"""Vertex AI text embeddings for the A1 benchmark (cloud; public CRSNSP text only).

One call per text with the model's default truncation, as both JurisMap versions call it. The
response gives the exact token count of the input and whether the model truncated it. Vectors are
cached on disk by (model, dimensions, task type, text), so a rerun makes no call.

Never import this module in a script that reads SUSEP data (ground rule 1); scripts/47 refuses to
run if it is loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jm_eval.io import load_config, resolve

# Fallback Google Cloud project for Vertex AI calls; JM_EVAL_GCP_PROJECT overrides it.
GCP_PROJECT = os.environ.get("JM_EVAL_GCP_PROJECT", "your-gcp-project")


@dataclass
class Embedding:
    vector: np.ndarray
    token_count: int | None
    truncated: bool | None
    credentials: str | None
    cached: bool


class VertexEmbedder:
    """google-genai embed_content: the primary service-account key first, the fallback project on errors."""

    def __init__(self, model: str, dimensions: int, task_type: str, cache_dir: str | Path | None = None,
                 location: str = "global", max_attempts: int = 6):
        self.model, self.dimensions, self.task_type = model, dimensions, task_type
        self.location, self.max_attempts = location, max_attempts
        root = Path(cache_dir) if cache_dir else resolve(load_config("paths")["data"]["cache"]) / "embeddings"
        self.cache_dir = root / f"{model}_{dimensions}_{task_type}"
        self._clients: list[tuple[str, Any]] | None = None

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.cache_dir / key[:2] / key

    def clients(self) -> list[tuple[str, Any]]:
        if self._clients is None:
            from google import genai
            from google.genai import types
            from google.oauth2 import service_account

            options = types.HttpOptions(timeout=120_000)        # ms; a hung request is retried, not waited for
            clients = []
            key = load_config("paths").get("gemini_primary_credentials")
            if key and resolve(key).exists():
                creds = service_account.Credentials.from_service_account_file(
                    str(resolve(key)), scopes=["https://www.googleapis.com/auth/cloud-platform"])
                clients.append(("primary", genai.Client(vertexai=True, project=creds.project_id, location=self.location,
                                                        credentials=creds, http_options=options)))
            clients.append(("fallback", genai.Client(vertexai=True, project=GCP_PROJECT, location=self.location,
                                                     http_options=options)))
            self._clients = clients
        return self._clients

    def _call(self, text: str) -> dict:
        from google.genai import types

        config = types.EmbedContentConfig(task_type=self.task_type, output_dimensionality=self.dimensions)
        clients, delay, error = self.clients(), 2.0, None
        for attempt in range(self.max_attempts):
            for label, client in clients:
                try:
                    response = client.models.embed_content(model=self.model, contents=[text], config=config)
                except Exception as e:              # quota, transient errors: next credentials, then wait
                    error = e
                    continue
                embedding = response.embeddings[0]
                stats = embedding.statistics
                return {"values": embedding.values, "credentials": label,
                        "token_count": int(stats.token_count) if stats and stats.token_count is not None else None,
                        "truncated": bool(stats.truncated) if stats and stats.truncated is not None else None}
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise RuntimeError(f"embedding failed after {self.max_attempts} attempts: {error}")

    def embed(self, text: str) -> Embedding:
        path = self._path(text)
        vector_path, meta_path = path.with_suffix(".npy"), path.with_suffix(".json")
        if vector_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return Embedding(np.load(vector_path), meta["token_count"], meta["truncated"], meta["credentials"], True)
        out = self._call(text)
        vector = np.asarray(out["values"], dtype=np.float32)
        if vector.shape != (self.dimensions,):
            raise RuntimeError(f"{self.model}: expected {self.dimensions} dimensions, got {vector.shape}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = vector_path.with_name(vector_path.name + f".{time.monotonic_ns()}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, vector)
        tmp.replace(vector_path)
        meta = {k: out[k] for k in ("token_count", "truncated", "credentials")} | {"chars": len(text)}
        meta_path.write_text(json.dumps(meta))
        return Embedding(vector, out["token_count"], out["truncated"], out["credentials"], False)
