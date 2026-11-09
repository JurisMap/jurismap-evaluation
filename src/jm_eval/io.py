"""Paths, config loading, hashing and never-overwrite artefact writing."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def load_config(name: str) -> dict[str, Any]:
    """Load `config/<name>.yaml`."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config path: `~` expanded, relative paths taken from the repo root."""
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _versions(p: Path) -> list[tuple[int, Path]]:
    """Existing numbered versions `<stem>_<n><suffix>` of an artefact, as (n, path)."""
    pattern = re.compile(re.escape(p.stem) + r"_(\d+)" + re.escape(p.suffix))
    return [(int(m.group(1)), q) for q in p.parent.glob(f"{p.stem}_*{p.suffix}") if (m := pattern.fullmatch(q.name))]


def new_artefact_path(path: str | os.PathLike) -> Path:
    """Return `path`, or `<stem>_<n><suffix>` with n one above the highest version that exists.

    Artefacts are never overwritten (version management rule 3). A deleted version's name is not
    reused while a later version exists, so the highest number is always the newest.
    """
    p = Path(path)
    versions = _versions(p)
    if not p.exists() and not versions:
        return p
    return p.with_name(f"{p.stem}_{max([1, *(n for n, _ in versions)]) + 1}{p.suffix}")


def latest_artefact_path(path: str | os.PathLike) -> Path:
    """The newest existing version of an artefact: `<stem>_<n><suffix>` with the highest n, else `path`."""
    p = Path(path)
    versions = _versions(p)
    return max(versions)[1] if versions else p


MANIFEST_FIELDS = ["path", "sha256", "bytes", "created_utc", "script", "run_id", "note"]


def record_artefact(path: str | os.PathLike, script: str, run_id: str, note: str = "",
                    manifest: str | os.PathLike | None = None) -> dict[str, Any]:
    """Append an artefact's sha256 to `data/MANIFEST.csv` and return the row."""
    p = Path(path).resolve()
    manifest_path = Path(manifest) if manifest else resolve(load_config("paths")["data"]["manifest"])
    row = {
        "path": os.path.relpath(p, REPO_ROOT),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "run_id": run_id,
        "note": note,
    }
    new = not manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row
