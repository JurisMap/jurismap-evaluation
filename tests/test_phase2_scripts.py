import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look up their module
    spec.loader.exec_module(mod)
    return mod


settings = _load("settings_table", "20_settings_table.py")
privacy = _load("prompt_privacy", "21_prompt_privacy.py")


def _repo(tmp_path, files):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("add", ".")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    run("tag", "t1")
    return repo


def test_evidence_check_reports_line_and_missing_pattern(tmp_path):
    repo = _repo(tmp_path, {"app.py": "a = 1\ntop_k = min(10, n)\n"})
    src = settings.Sources({"backend": repo}, {("v1", "backend"): "t1"})
    err, where = src.check("v1", settings.Ev("backend", "app.py", r"top_k = min\(10"))
    assert err is None and where.endswith("app.py:2")
    err, _ = src.check("v1", settings.Ev("backend", "app.py", r"top_k = 30"))
    assert err and "pattern not found" in err


def test_env_files_keep_only_model_lines(tmp_path):
    repo = _repo(tmp_path, {".env": "API_KEY=secret\nCHATBOT_MODEL=m1\nVERTEX_REGION=r\n"})
    src = settings.Sources({"backend": repo}, {("v1", "backend"): "t1"})
    text = src.text("v1", "backend", ".env")
    assert "secret" not in text and "CHATBOT_MODEL=m1" in text and "VERTEX_REGION=r" in text


def test_pins_first_pin_wins(tmp_path):
    repo = _repo(tmp_path, {"requirements.txt": "Flask==3.1.0\nnumpy==2.1.3\nflask>=2.2\n"})
    src = settings.Sources({"backend": repo}, {("v1", "backend"): "t1"})
    assert src.pins("v1", "backend", "requirements.txt", ["Flask", "numpy", "umap-learn"]) == \
        "Flask==3.1.0, numpy==2.1.3, umap-learn (not listed)"


def test_leak_detection_finds_long_runs_only():
    prompt = privacy.words("um dois tres quatro cinco seis sete oito nove dez onze doze treze")
    grams = privacy.shingles(prompt, 10)
    text = privacy.words("antes um dois tres quatro cinco seis sete oito nove dez onze depois")
    runs = privacy.leaked_runs(text, grams, 10)
    assert [(e - s) for s, e in runs] == [11]
    short = privacy.words("antes um dois tres quatro cinco seis sete oito nove depois")
    assert privacy.leaked_runs(short, grams, 10) == []


def test_check_flags_unknown_or_long_excerpts(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    long_excerpt = " ".join(f"w{i}" for i in range(26))
    (results / "prompt_specs.md").write_text(f"«alfa beta gama» «delta epsilon» «{long_excerpt}»", encoding="utf-8")
    report = privacy.check({"p": "alfa beta gama " + long_excerpt}, results)
    assert len(report["bad_excerpts"]) == 2  # one not in any prompt, one over 25 words
    assert report["leaks"] == []  # the long run is declared as an excerpt (reported above instead)


def test_check_sets_accepted_runs_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "REPO_ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    (results / "prompt_specs.md").write_text("no excerpts", encoding="utf-8")
    prompt = "um dois tres quatro cinco seis sete oito nove dez onze doze treze"
    (results / "a.md").write_text("antes um dois tres quatro cinco seis sete oito nove dez onze depois", encoding="utf-8")
    (results / "b.md").write_text("antes tres quatro cinco seis sete oito nove dez onze doze treze", encoding="utf-8")
    first = privacy.run_sha256(privacy.words("um dois tres quatro cinco seis sete oito nove dez onze"))
    report = privacy.check({"p": prompt}, results, {first: "accepted for the test"})
    assert [r["file"] for r in report["accepted_runs"]] == ["results/a.md"]
    assert [r["file"] for r in report["leaks"]] == ["results/b.md"]   # other words: another fingerprint
    assert "start" not in report["accepted_runs"][0]                 # an accepted run's text is not written
