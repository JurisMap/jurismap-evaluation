"""Phase 11: shared helpers of the paper's assets and records (scripts/98 and scripts/99).

- `Sources` resolves the pinned files of config/paper_assets.yaml, checks that each is the newest version of
  its family, and records which files an asset or a number was read from.
- `producers` finds, for a results file, the run manifest whose recorded output has the file's current hash
  (so a file edited by hand after its run is reported as having no producing run).
- Formatting helpers turn numbers into the strings the paper prints.
"""

from __future__ import annotations

import json
import math
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from jm_eval.io import REPO_ROOT, latest_artefact_path, load_config, resolve, sha256_file

PRIVATE_PARTS = ("data/private",)


# ------------------------------------------------------------------------------------------------ sources

class StaleSource(RuntimeError):
    pass


class MissingOptionalSource(RuntimeError):
    """A source listed under `optional_sources` is absent (e.g. the B2 workbook, not in the data archive)."""

    def __init__(self, name: str, note: str):
        super().__init__(f"{name} not available: {note}")
        self.name, self.note = name, note


class Sources:
    """The pinned source files of config/paper_assets.yaml."""

    def __init__(self, cfg: dict | None = None, *, allow_stale: bool = False):
        self.cfg = cfg if cfg is not None else load_config("paper_assets")
        self.paths = {name: resolve(p) for name, p in self.cfg["sources"].items()}
        self.optional = self.cfg.get("optional_sources") or {}
        self.allow_stale = allow_stale
        self.used: dict[str, set[str]] = {}
        self._cache: dict[str, Any] = {}

    def check(self) -> list[str]:
        """Pinned files that are missing or not the newest version of their family (as messages)."""
        problems = []
        for name, p in self.paths.items():
            if not p.exists():
                if name not in self.optional:
                    problems.append(f"{name}: {rel(p)} does not exist")
                continue
            newest = latest_artefact_path(family_base(p))
            if newest != p:
                problems.append(f"{name}: {rel(p)} is pinned but {rel(newest)} is newer")
        return problems

    def path(self, name: str, asset: str | None = None) -> Path:
        if not self.paths[name].exists() and name in self.optional:
            raise MissingOptionalSource(name, self.optional[name])
        if asset:
            self.used.setdefault(asset, set()).add(name)
        return self.paths[name]

    def missing_optional(self) -> dict[str, str]:
        """Optional sources absent from this copy: {name: why}."""
        return {n: why for n, why in self.optional.items() if not self.paths[n].exists()}

    def csv(self, name: str, asset: str | None = None, **kw) -> pd.DataFrame:
        p = self.path(name, asset)
        key = f"csv:{name}:{sorted(kw.items())}"
        if key not in self._cache:
            self._cache[key] = pd.read_csv(p, **kw)
        return self._cache[key].copy()

    def json(self, name: str, asset: str | None = None) -> Any:
        p = self.path(name, asset)
        key = f"json:{name}"
        if key not in self._cache:
            self._cache[key] = json.loads(p.read_text(encoding="utf-8"))
        return self._cache[key]

    def excel(self, name: str, sheet: str, asset: str | None = None) -> pd.DataFrame:
        p = self.path(name, asset)
        key = f"xlsx:{name}:{sheet}"
        if key not in self._cache:
            self._cache[key] = pd.read_excel(p, sheet_name=sheet)
        return self._cache[key].copy()

    def text(self, name: str, asset: str | None = None) -> str:
        return self.path(name, asset).read_text(encoding="utf-8")


def family_base(p: Path) -> Path:
    """`a1_retrieval_main_5.csv` -> `a1_retrieval_main.csv` (the name latest_artefact_path takes)."""
    m = re.fullmatch(r"(.+)_(\d+)", p.stem)
    return p.with_name(f"{m.group(1)}{p.suffix}") if m else p


def rel(p: str | os.PathLike) -> str:
    """Path relative to the repo root where possible, else as given."""
    try:
        return os.path.relpath(Path(p).resolve(), REPO_ROOT) if Path(p).resolve().is_relative_to(REPO_ROOT) else str(p)
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------------------------- producers

def manifest_index(roots: list[Path] | None = None) -> dict[tuple[str, str], tuple[str, str]]:
    """(path relative to the repo, sha256) -> (run manifest relative to the repo, script) over every run manifest
    under results/ and data/ (private folders excluded)."""
    roots = roots or [resolve("results"), resolve("data")]
    index: dict[tuple[str, str], tuple[str, str]] = {}
    for root in roots:
        for mpath in sorted(root.rglob("*.manifest.json")):
            r = rel(mpath)
            if any(r.startswith(p) for p in PRIVATE_PARTS):
                continue
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for out in m.get("outputs", []):
                if out.get("path") and out.get("sha256"):
                    index[(rel(out["path"]), out["sha256"])] = (r, m.get("script", ""))
    return index


def producer(path: Path, index: dict) -> tuple[str, str] | None:
    """The run manifest and script that wrote this file as it is now, or None (hand-filled or edited)."""
    return index.get((rel(path), sha256_file(path)))


# ------------------------------------------------------------------------------------------------ formats

def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def rounded(x, digits: int) -> Decimal:
    """Round half up on the value as written (result files store a few decimals, so halves are common:
    6.25 -> 6.3, -0.0185 -> -0.019), not on its binary approximation."""
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def f(x, digits: int = 3) -> str:
    """Rounded half up; negative numbers with a minus sign (U+2212), as typeset."""
    return "" if is_missing(x) else f"{rounded(x, digits):.{digits}f}".replace("-", "−")


def signed(x, digits: int = 3) -> str:
    return "" if is_missing(x) else f"{rounded(x, digits):+.{digits}f}".replace("-", "−")


def interval(lo, hi, digits: int = 3) -> str:
    if is_missing(lo) or is_missing(hi):
        return ""
    return f"{f(lo, digits)} to {f(hi, digits)}"


def pval(p) -> str:
    """'< 0.001' below 0.001, else three decimals."""
    if is_missing(p):
        return ""
    return "< 0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def share(k: int, n: int) -> str:
    return f"{int(k)} of {int(n)}"


def pct(x, digits: int = 1) -> str:
    return "" if is_missing(x) else f"{rounded(Decimal(repr(float(x))) * 100, digits):.{digits}f}%"


def usd(x, digits: int = 3) -> str:
    return "" if is_missing(x) else f"${f(x, digits)}"
