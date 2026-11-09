"""The SUSEP script must be unable to send data anywhere (ground rule 1), and its metrics must
follow the definitions of the 2024 paper. No private file is read here."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jm_eval.retrieval import gold as G

ROOT = Path(__file__).resolve().parents[1]
PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY", "http_proxy", "https_proxy", "all_proxy", "ftp_proxy")


def run_python(code: str, **env) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if k not in PROXIES}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env={**clean, **env})


# ---------------------------------------------------------------- network guard
def test_guard_blocks_connections_and_name_lookups():
    code = """
import socket
from jm_eval.retrieval import guard
guard.enforce_local_only()
for attempt in (lambda: socket.create_connection(("93.184.216.34", 80), timeout=2),
                lambda: socket.create_connection(("example.com", 80), timeout=2),
                lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex(("93.184.216.34", 443)),
                lambda: socket.getaddrinfo("huggingface.co", 443)):
    try:
        attempt()
    except guard.NetworkBlocked:
        print("blocked")
    else:
        print("NOT BLOCKED")
import os
print(os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"])
"""
    out = run_python(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["blocked"] * 4 + ["1", "1"]


def test_guard_blocks_an_http_client():
    code = """
import urllib.request
from jm_eval.retrieval import guard
guard.enforce_local_only()
try:
    urllib.request.urlopen("https://example.com", timeout=2)
except Exception as e:
    print(type(e).__name__, "NetworkBlocked" in repr(e) or "blocked" in str(e))
else:
    print("NOT BLOCKED")
"""
    out = run_python(code)
    assert out.returncode == 0, out.stderr
    assert "NOT BLOCKED" not in out.stdout and out.stdout.strip().endswith("True")


def test_guard_refuses_when_a_cloud_client_is_loaded():
    out = run_python("import google.genai\nfrom jm_eval.retrieval import guard\nguard.enforce_local_only()")
    assert out.returncode != 0 and "cloud client modules are loaded" in out.stderr
    out = run_python("import jm_eval.llm\nfrom jm_eval.retrieval import guard\nguard.enforce_local_only()")
    assert out.returncode != 0 and "jm_eval.llm" in out.stderr


@pytest.mark.parametrize("variable", ["HTTPS_PROXY", "http_proxy"])
def test_guard_refuses_when_a_proxy_is_set(variable):
    out = run_python("from jm_eval.retrieval import guard\nguard.enforce_local_only()", **{variable: "http://proxy:3128"})
    assert out.returncode != 0 and "proxy environment variables are set" in out.stderr


def test_susep_script_runs_the_guard_first_and_loads_no_cloud_client():
    source = (ROOT / "scripts/47_a1_susep.py").read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]                                  # after the module docstring
    guard_call = body.index("guard.enforce_local_only()")
    other_imports = [body.index(line) for line in body.splitlines()
                     if line.startswith(("import ", "from ")) and "__future__" not in line and "guard" not in line]
    assert guard_call < min(other_imports)
    # with a proxy set, the script stops before it parses arguments or opens a file
    out = subprocess.run([sys.executable, "scripts/47_a1_susep.py", "--help"], capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "HTTPS_PROXY": "http://proxy:3128"})
    assert out.returncode != 0 and "refusing to read SUSEP data" in out.stderr
    # everything the script imports leaves no cloud client in the process
    imports = "\n".join(line for line in body.splitlines() if line.startswith(("import ", "from ")) and "__future__" not in line)
    code = ("from jm_eval.retrieval import guard\nguard.enforce_local_only()\n" + imports.replace("  # noqa: E402", "") +
            "\nimport torch, sentence_transformers\nprint(guard.loaded_cloud_modules())")
    out = run_python(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_cloud_modules_are_never_imported_by_the_susep_script():
    source = (ROOT / "scripts/47_a1_susep.py").read_text(encoding="utf-8")
    for name in ("google", "vertex_embed", "jm_eval.llm", "requests", "openai", "anthropic"):
        assert f"import {name}" not in source and f"from {name}" not in source and f"{name} import" not in source


# ---------------------------------------------------------------- gold-standard metrics
def test_pair_matrix_is_symmetric_with_missing_pairs():
    m = G.pair_matrix([10, 20, 30], np.array([10, 10]), np.array([20, 30]), np.array([2.0, 0.0]))
    assert m[0, 1] == m[1, 0] == 2.0 and m[0, 2] == 0.0 and np.isnan(m[1, 2]) and np.isnan(m[0, 0])


def test_recall_curves_follow_the_2024_definition():
    # 4 cases. Case 0 ranks 1, 2, 3; its relevant cases are 2 and 3.
    sim = np.array([[1.0, 0.9, 0.8, 0.7], [0.9, 1.0, 0.1, 0.2], [0.8, 0.1, 1.0, 0.3], [0.7, 0.2, 0.3, 1.0]])
    gold = np.zeros((4, 4))
    gold[0, 2] = gold[2, 0] = 3.0
    gold[0, 3] = gold[3, 0] = 1.0
    recall = G.recall_curves(sim, gold)
    assert recall.shape == (4, 4)
    assert recall[0].tolist() == [0.0, 0.5, 1.0, 1.0]                 # k = 1..4; k = 4 lists every other case
    assert recall[1].tolist() == [1.0] * 4                            # no relevant pair: Recall is 1
    per_case = G.per_case_metrics(sim, gold)
    assert per_case.loc[0, "mAR"] == pytest.approx(2.5 / 4)
    assert per_case.loc[0, "Recall@5"] == 1.0 and np.isnan(per_case.loc[1, "nDCG@10"])
    dcg = 0 / np.log2(2) + 3 / np.log2(3) + 1 / np.log2(4)
    ideal = 3 / np.log2(2) + 1 / np.log2(3)
    assert per_case.loc[0, "nDCG@10"] == pytest.approx(dcg / ideal)
    assert G.pair_spearman(gold + np.eye(4), gold) == pytest.approx(1.0)


def test_icc_matches_shrout_and_fleiss():
    # Shrout & Fleiss (1979), Table 2: 6 targets rated by 4 judges; ICC(2,1) = .29, ICC(2,4) = .62
    ratings = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
    icc = G.icc2(ratings)
    assert icc["icc_2_1"] == pytest.approx(0.29, abs=0.005) and icc["icc_2_k"] == pytest.approx(0.62, abs=0.005)
