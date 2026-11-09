"""Where the A3 evaluation keeps its files, and how its scripts read each other's outputs.

data/v2/a3/                 working files of the evaluated question set (git-ignored)
data/v2/a3_dev/             the same for the development sets, which are never reported as results
data/v2/a3_dev/candidates/<name>/   answers and scores of one candidate strategy (Phase 6.1)
data/v2/a3_questions.jsonl  the frozen question set (written by scripts/61 --freeze after the author's check)
data/v2/a5/<setting>/       Phase 7: questions, answers and scores at one context size (config/a5_context_size.yaml)
results/                    tables for the paper

A development run (`--dev`) reads and writes only under data/v2/a3_dev/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jm_eval.io import latest_artefact_path, load_config, resolve, sha256_file

CORPUS_COLUMNS = ["acordao_hash", "acordao_titulo", "ano_sessoes", "sessao_nr_int", "sessao_datas", "sessao_acordao",
                  "acordao_url", "acordao_recorrente", "acordao_recorrido", "acordao_relator", "acordao_advogado",
                  "acordao_ementa", "acordao_penalidade", "acordao_base", "acordao_acordao", "acordao_participantes",
                  "acordao_relatorio", "acordao_voto_relator", "acordao_resultado"]


_candidate: str | None = None
_setting: str | None = None


def use_candidate(name: str | None) -> None:
    """Phase 6.1: the working tables of this process belong to one candidate strategy (development sets)."""
    global _candidate
    if name is not None and name not in load_config("a3_candidates")["candidates"]:
        raise ValueError(f"unknown candidate {name!r} (config/a3_candidates.yaml)")
    _candidate = name


def candidate() -> str | None:
    return _candidate


def use_setting(name: str | None) -> None:
    """Phase 7: the questions and working tables of this process belong to one context size."""
    global _setting
    if name is not None and name not in load_config("a5_context_size")["settings"]:
        raise ValueError(f"unknown setting {name!r} (config/a5_context_size.yaml)")
    _setting = name


def setting() -> str | None:
    return _setting


def a5_dir() -> Path:
    path = resolve(load_config("paths")["data"]["v2"]) / "a5"
    path.mkdir(parents=True, exist_ok=True)
    return path


def a5_freeze_record_path() -> Path:
    return results_dir() / "a5_questions_freeze.json"


def root_dir(dev: bool = False) -> Path:
    """Where the case sets, key points and questions are: shared by every candidate."""
    if _setting and not dev:
        path = a5_dir() / _setting
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = resolve(load_config("paths")["data"]["v2"]) / ("a3_dev" if dev else "a3")
    path.mkdir(parents=True, exist_ok=True)
    return path


def work_dir(dev: bool = False) -> Path:
    """Where answers, scores and claims are written: per candidate when one is in use."""
    path = root_dir(dev)
    if dev and _candidate:
        path = path / "candidates" / _candidate
        path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    return resolve(load_config("paths")["results"])


def frozen_questions_path() -> Path:
    return resolve(load_config("paths")["data"]["v2"]) / "a3_questions.jsonl"


def freeze_record_path() -> Path:
    return results_dir() / "a3_questions_freeze.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")


def load_corpus() -> pd.DataFrame:
    """The frozen corpus with the fields the chatbot receives, in `acordao_hash` order."""
    corpus = pd.read_parquet(resolve(load_config("a3_chatbot")["corpus"]), columns=CORPUS_COLUMNS)
    return corpus.sort_values("acordao_hash").reset_index(drop=True)


def load_case_sets(dev: bool = False) -> list[dict]:
    """The evaluated sets, or the development sets (both are in the same file)."""
    path = latest_artefact_path(root_dir() / "case_sets.json")
    if not path.exists():
        raise FileNotFoundError(f"{path}: run scripts/60_a3_case_sets.py")
    return [s for s in json.loads(path.read_text(encoding="utf-8"))["sets"] if s["development"] == dev]


def all_set_decisions() -> set[str]:
    """Every decision that is a member or the seed of a set, development included."""
    path = latest_artefact_path(root_dir() / "case_sets.json")
    sets = json.loads(path.read_text(encoding="utf-8"))["sets"]
    return {h for s in sets for h in [s["seed_hash"], *s["case_hashes"]]}


def load_questions(dev: bool = False) -> list[dict]:
    """The frozen question set (its hash is checked against the freeze record), or the development questions."""
    if _setting and not dev:                     # Phase 7: the questions of one context size, checked against their hash
        record = latest_artefact_path(a5_freeze_record_path())
        if not record.exists():
            raise FileNotFoundError(f"{record}: run scripts/72_a5_settings.py")
        entry = json.loads(record.read_text(encoding="utf-8"))["settings"][_setting]
        path = root_dir() / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path} does not match the hash of {record.name}")
        return read_jsonl(path)
    if dev:
        path = latest_artefact_path(root_dir(True) / "questions.jsonl")
        if not path.exists():
            raise FileNotFoundError(f"{path}: run scripts/61_a3_questions.py --dev")
        return read_jsonl(path)
    path, record = frozen_questions_path(), freeze_record_path()
    if not path.exists() or not record.exists():
        raise FileNotFoundError("the question set is not frozen: The author checks the sample, then "
                                "scripts/61_a3_questions.py --freeze")
    expected = json.loads(record.read_text(encoding="utf-8"))["sha256"]
    if sha256_file(path) != expected:
        raise ValueError(f"{path} does not match the hash of {record.name}")
    return read_jsonl(path)


def load_table(name: str, dev: bool = False) -> pd.DataFrame:
    """The newest version of a working table (e.g. "answers.parquet")."""
    path = latest_artefact_path(work_dir(dev) / name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    return pd.read_parquet(path)
