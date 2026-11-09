"""Phase 2 step 4: keep verbatim prompts private and check that results/ does not leak them.

1. Export: copies every prompt of both versions (backend prompt files, inline prompts in code, data
   pipeline prompts) at the v1 tag and the v2 freeze tag into data/private/prompts_verbatim/<version>/
   <repo>/<path>, with an index (path, sha256, words). data/private/ is git-ignored and blocked by the
   pre-commit hook. An existing file with different content is never overwritten (the run fails).
2. Check: every «...» excerpt in results/prompt_specs.md must occur verbatim in a source prompt and
   have at most 25 words; outside those excerpts, no run of MIN_RUN or more consecutive words from any
   prompt may appear in a text file under results/. Exits 1 on a violation.

Outputs: data/private/prompts_verbatim/ (private), results/prompt_privacy_check.json,
         results/21_prompt_privacy.manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path

from jm_eval.io import CONFIG_DIR, REPO_ROOT, load_config, new_artefact_path, resolve, sha256_bytes
from jm_eval.manifest import RunManifest

SCRIPT = "21_prompt_privacy"
MAX_EXCERPT_WORDS = 25
MIN_RUN = 10  # a leak = this many consecutive prompt words outside a declared excerpt
TEXT_SUFFIXES = {".md", ".csv", ".mmd", ".svg", ".json", ".txt"}
EXCERPT_RE = re.compile(r"«(.+?)»", re.S)

# (repo, path glob or file) per version. Python files hold inline prompts (legal-search queries, OCR,
# the GPT-4o curation message); they are exported whole but stay private.
SOURCES = {
    "v1": [("backend", "prompts/"), ("backend", "agents/LegalSearcherAgent.py"),
           ("backend", "utils/file_parsers.py"), ("backend", "utils/curateNames.py"),
           ("backend", "agents/defense_structure.json"), ("scrapers", "data_pipeline/prompts/")],
    "v2": [("backend", "prompts/"), ("backend", "utils/file_parsers.py"),
           ("scrapers", "crsnsp_scraper/prompts/")],
}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def list_files(repo: Path, rev: str, spec: str) -> list[str]:
    if not spec.endswith("/"):
        return [spec]
    names = git(repo, "ls-tree", "-r", "--name-only", rev, spec).decode().splitlines()
    return [n for n in names if n.endswith((".yaml", ".yml", ".json"))]


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"[0-9a-zà-ÿ]+", text)


def export(repos: dict[str, Path], revs: dict[str, str], out_root: Path) -> tuple[list[dict], dict[str, str]]:
    index, texts = [], {}
    for version, specs in SOURCES.items():
        for repo_name, spec in specs:
            repo, rev = repos[repo_name], revs[version]
            for path in list_files(repo, rev, spec):
                data = git(repo, "show", f"{rev}:{path}")
                dest = out_root / version / repo_name / path
                if dest.exists() and dest.read_bytes() != data:
                    raise SystemExit(f"{dest} exists with different content; not overwritten")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(data)
                text = data.decode("utf-8", errors="replace")
                key = f"{version}/{repo_name}/{path}"
                texts[key] = text
                index.append({"file": key, "rev": rev, "sha256": sha256_bytes(data), "words": len(words(text))})
    return index, texts


def shingles(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def leaked_runs(tokens: list[str], grams: set[tuple[str, ...]], n: int) -> list[tuple[int, int]]:
    """Maximal [start, end) token spans every n-gram of which occurs in the prompts."""
    hits = [tuple(tokens[i:i + n]) in grams for i in range(len(tokens) - n + 1)]
    runs, i = [], 0
    while i < len(hits):
        if hits[i]:
            j = i
            while j + 1 < len(hits) and hits[j + 1]:
                j += 1
            runs.append((i, j + n))
            i = j + 1
        else:
            i += 1
    return runs


def check(texts: dict[str, str], results_dir: Path) -> dict:
    prompt_tokens = {k: words(v) for k, v in texts.items()}
    grams = set().union(*(shingles(t, MIN_RUN) for t in prompt_tokens.values()))
    joined = {k: " ".join(t) for k, t in prompt_tokens.items()}

    specs = (results_dir / "prompt_specs.md").read_text(encoding="utf-8")
    excerpts, bad_excerpts = [], []
    for raw in EXCERPT_RE.findall(specs):
        toks = words(raw.replace("…", " "))
        found = [k for k, j in joined.items() if " ".join(toks) in j]
        rec = {"excerpt_words": len(toks), "found_in": found[:3]}
        excerpts.append(rec)
        if not found or len(toks) > MAX_EXCERPT_WORDS:
            bad_excerpts.append({"excerpt": raw[:60] + "…", **rec})
    allowed = [words(e.replace("…", " ")) for e in EXCERPT_RE.findall(specs)]

    leaks = []
    for f in sorted(results_dir.rglob("*")):
        if not f.is_file() or f.suffix not in TEXT_SUFFIXES or f.name == "prompt_privacy_check.json":
            continue
        toks = words(f.read_text(encoding="utf-8", errors="replace"))
        for s, e in leaked_runs(toks, grams, MIN_RUN):
            run = toks[s:e]
            inside_excerpt = any(" ".join(run) in " ".join(a) for a in allowed)
            if not inside_excerpt:
                leaks.append({"file": str(f.relative_to(REPO_ROOT)), "words": len(run),
                              "start": " ".join(run[:6]) + " …"})
    return {"min_run_words": MIN_RUN, "max_excerpt_words": MAX_EXCERPT_WORDS, "excerpts": len(excerpts),
            "max_excerpt_len": max((e["excerpt_words"] for e in excerpts), default=0),
            "bad_excerpts": bad_excerpts, "leaks": leaks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", default="v2.2_freeze.json")
    args = ap.parse_args()
    paths = load_config("paths")
    freeze = json.loads((CONFIG_DIR / args.freeze).read_text(encoding="utf-8"))
    revs = {"v1": paths["v1_tag"], "v2": freeze["tag"]}
    repos = {name: resolve(p) for name, p in paths["repos"].items()}
    out_root = resolve(paths["data"]["private"]) / "prompts_verbatim"
    results_dir = resolve(paths["results"])

    with RunManifest(SCRIPT, results_dir, params={"revisions": revs, "min_run": MIN_RUN,
                                                  "max_excerpt_words": MAX_EXCERPT_WORDS},
                     record_outputs=False) as run:
        index, texts = export(repos, revs, out_root)
        (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        report = check(texts, results_dir)
        report["prompt_files"] = len(index)
        report["passed"] = not report["bad_excerpts"] and not report["leaks"]
        out = new_artefact_path(results_dir / "prompt_privacy_check.json")
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        run.data["summary"] = {k: report[k] for k in ("prompt_files", "excerpts", "passed")}
        # data/private is not recorded in data/MANIFEST.csv (the file list itself stays private)
        run.data["outputs"].append({"path": str(out.relative_to(REPO_ROOT))})
        print(json.dumps({k: v for k, v in report.items() if k != "excerpts"}, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
