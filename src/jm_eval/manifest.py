"""Run manifests (ground rule 5).

Every script run writes `<name>.manifest.json` next to its outputs with the git
SHAs of all repos involved, model IDs and parameters, the config hash, seeds,
start/end timestamps and package versions.

    with RunManifest("00_export_corpus", out_dir, models={...}, params={...}) as run:
        ...
        run.add_output(path)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jm_eval.io import CONFIG_DIR, REPO_ROOT, load_config, record_artefact, resolve, sha256_file

DEFAULT_SEED = 20260926


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_state(repo: Path) -> dict[str, Any]:
    """SHA, branch, exact tag and dirty flag of a repo (paths of dirty files are not recorded)."""
    if _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        return {"path": str(repo), "sha": None, "note": "not a git repo"}
    status = _git(repo, "status", "--porcelain", "--untracked-files=no") or ""
    return {
        "path": str(repo),
        "sha": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "tag": _git(repo, "describe", "--tags", "--exact-match"),
        "dirty": bool(status),
    }


def all_git_states() -> dict[str, Any]:
    repos = load_config("paths").get("repos", {})
    return {name: git_state(resolve(path)) for name, path in repos.items()}


def config_hash() -> dict[str, Any]:
    """sha256 of each config file, plus a combined hash over (name, sha) pairs."""
    files = sorted(p for p in CONFIG_DIR.iterdir() if p.suffix in {".yaml", ".yml", ".json"})
    per_file = {p.name: sha256_file(p) for p in files}
    combined = hashlib.sha256(json.dumps(per_file, sort_keys=True).encode()).hexdigest()
    return {"combined": combined, "files": per_file}


def package_versions() -> list[str]:
    """`uv pip freeze` for this repo's environment; falls back to importlib.metadata."""
    try:
        out = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable],
                             capture_output=True, text=True, check=True, cwd=REPO_ROOT)
        return sorted(line for line in out.stdout.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        from importlib.metadata import distributions
        return sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunManifest:
    """Context manager that writes a run manifest when the block exits (also on failure)."""

    def __init__(self, name: str, out_dir: str | Path, *, models: dict | None = None,
                 params: dict | None = None, seeds: dict | None = None, record_outputs: bool = True):
        self.name = name
        self.out_dir = Path(out_dir)
        self.run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.record_outputs = record_outputs
        self.data: dict[str, Any] = {
            "run_id": self.run_id,
            "script": name,
            "argv": sys.argv,
            "models": models or {},
            "params": params or {},
            "seeds": seeds if seeds is not None else load_config("seeds") or {"default": DEFAULT_SEED},
            "inputs": [],
            "outputs": [],
            "summary": {},
        }
        self.path: Path | None = None

    def add_input(self, path: str | Path, sha256: str | None = None, **extra: Any) -> None:
        p = Path(path)
        self.data["inputs"].append({"path": str(p), "sha256": sha256 or sha256_file(p), **extra})

    def add_output(self, path: str | Path, note: str = "") -> None:
        p = Path(path)
        entry = {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        self.data["outputs"].append(entry)
        if self.record_outputs:
            record_artefact(p, self.name, self.run_id, note)

    def __enter__(self) -> "RunManifest":
        self.data["started_utc"] = _now()
        self.data["git"] = all_git_states()
        self.data["config_hash"] = config_hash()
        self.data["python"] = {"version": sys.version, "executable": sys.executable,
                               "platform": platform.platform()}
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.data["ended_utc"] = _now()
        self.data["status"] = "ok" if exc is None else "failed"
        if exc is not None:
            self.data["error"] = "".join(traceback.format_exception_only(exc_type, exc)).strip()
        self.data["packages"] = package_versions()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{self.name}.manifest.json"
        n = 2
        while self.path.exists():
            self.path = self.out_dir / f"{self.name}_{n}.manifest.json"
            n += 1
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return False
