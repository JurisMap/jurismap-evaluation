"""On-disk LLM response cache (ground rule 6).

Responses are keyed by (model, prompt hash, params), so reruns are free and
deterministic. One JSON file per entry, sharded by the first two hex digits:
`data/cache/llm/<ab>/<key>.json`. Writes are atomic; an existing entry is never
overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from jm_eval.io import load_config, resolve


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def prompt_hash(prompt: Any) -> str:
    """Hash of a prompt: a string, or a structure such as a list of chat messages."""
    text = prompt if isinstance(prompt, str) else _canonical(prompt)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(model: str, prompt: Any, params: dict | None = None) -> str:
    material = _canonical({"model": model, "prompt_sha256": prompt_hash(prompt), "params": params or {}})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class CacheHit:
    key: str
    response: Any
    meta: dict


class LLMCache:
    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = resolve(load_config("paths")["data"]["cache"]) / "llm"
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, model: str, prompt: Any, params: dict | None = None) -> CacheHit | None:
        key = cache_key(model, prompt, params)
        p = self._path(key)
        if (log := os.environ.get("JM_EVAL_CACHE_LOG")):      # which entries a run reads (release packaging)
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{key}\t{int(p.exists())}\n")
        if not p.exists():
            return None
        entry = json.loads(p.read_text(encoding="utf-8"))
        return CacheHit(key=key, response=entry["response"], meta=entry.get("meta", {}))

    def put(self, model: str, prompt: Any, params: dict | None, response: Any,
            meta: dict | None = None) -> str:
        """Store a response. `meta` holds usage (tokens, cost, latency). Returns the key."""
        key = cache_key(model, prompt, params)
        p = self._path(key)
        if p.exists():
            return key
        p.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "model": model,
            "prompt_sha256": prompt_hash(prompt),
            "params": params or {},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "response": response,
            "meta": meta or {},
        }
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
            os.replace(tmp, p)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return key

    def call(self, model: str, prompt: Any, params: dict | None,
             fn: Callable[[], tuple[Any, dict]]) -> tuple[Any, dict, bool]:
        """Return the cached response, or run `fn() -> (response, meta)` and cache it.

        Returns (response, meta, from_cache).
        """
        hit = self.get(model, prompt, params)
        if hit is not None:
            return hit.response, hit.meta, True
        response, meta = fn()
        self.put(model, prompt, params, response, meta)
        return response, meta, False
