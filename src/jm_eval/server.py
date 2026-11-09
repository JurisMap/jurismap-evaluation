"""Start and talk to a local JurisMap v2 server (HTTP only; no backend code is imported).

    with LocalServer(run_dir) as server:
        server.session.post(server.url("/API_chatbot"), json={...})

The server runs app/backend with the backend's own virtualenv and the process environment from
config/v2_server.env (EVAL_MODE=1, DATA_VERSION=v2, ...). Its evaluation log (one JSON line per
request) is written to <run_dir>/server_logs/ and its stdout/stderr to <run_dir>/server.log.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import requests
from dotenv import dotenv_values

from jm_eval.io import CONFIG_DIR, load_config, resolve

SERVER_ENV_FILE = CONFIG_DIR / "v2_server.env"
PATH_VARS = ("DATA_DIR", "GEMINI_PRIMARY_CREDENTIALS")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_env(run_dir: Path, port: int) -> dict[str, str]:
    values = {k: v for k, v in dotenv_values(SERVER_ENV_FILE).items() if v is not None}
    for var in PATH_VARS:
        if var in values:
            values[var] = str(resolve(values[var]))
    values["EVAL_LOG_DIR"] = str(run_dir / "server_logs")
    values["PORT"] = str(port)
    return values


class LocalServer:
    def __init__(self, run_dir: str | Path, port: int | None = None, startup_timeout: float = 600):
        self.run_dir = Path(run_dir)
        self.port = port or free_port()
        self.startup_timeout = startup_timeout
        self.backend = resolve(load_config("paths")["repos"]["backend"])
        self.env_overrides = server_env(self.run_dir, self.port)
        self.process: subprocess.Popen | None = None
        self.session = requests.Session()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def __enter__(self) -> "LocalServer":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        code = (f"from app import app; app.run(host='127.0.0.1', port={self.port}, "
                "threaded=True, use_reloader=False, debug=False)")
        self.log_file = open(self.run_dir / "server.log", "ab")
        self.process = subprocess.Popen(
            [str(self.backend / ".venv/bin/python"), "-c", code], cwd=self.backend,
            env={**os.environ, **self.env_overrides}, stdout=self.log_file, stderr=subprocess.STDOUT,
            start_new_session=True)
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with {self.process.returncode}; see {self.run_dir / 'server.log'}")
            try:
                if requests.get(self.url("/health"), timeout=2).ok:
                    return self
            except requests.ConnectionError:
                pass
            time.sleep(2)
        self.__exit__(None, None, None)
        raise TimeoutError("server did not become healthy")

    def __exit__(self, *exc) -> bool:
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
        self.log_file.close()
        return False

    def eval_log(self, request_id: str, timeout: float = 30) -> dict[str, Any]:
        """The server's evaluation log record for a request (written when the response closes)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for path in sorted((self.run_dir / "server_logs").glob("requests_*.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    if record["request_id"] == request_id:
                        return record
            time.sleep(0.5)
        raise KeyError(f"no evaluation log for request {request_id}")


def sse_messages(response: requests.Response) -> Iterator[dict[str, Any]]:
    """Parse a text/event-stream response (`data: {...}` lines)."""
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield json.loads(line[len("data: "):])


# Failures of this machine's environment, not of the system under test: the network (outages of 2026-10-20)
# and the fallback credentials (a primary-key failure sends calls to the user's login for the cooldown; when
# that login has expired every call fails at once, 2026-10-20). A request that failed only so is sent again.
NETWORK_ERRORS = ("NameResolutionError", "Failed to resolve", "oauth2.googleapis.com", "Max retries exceeded",
                  "ConnectionError", "Connection aborted", "Temporary failure in name resolution",
                  "Reauthentication is needed")


def network_failure(errors) -> bool:
    """A failure caused by this machine's network (Google unreachable), not by the system under test."""
    text = " ".join(map(str, errors or []))
    return any(marker in text for marker in NETWORK_ERRORS)


def wait_for_network(host: str = "oauth2.googleapis.com", limit_s: int = 1800) -> None:
    """Block until `host` resolves again (the outages of 2026-10-20 were DNS failures on this machine)."""
    deadline = time.time() + limit_s
    while True:
        try:
            socket.gethostbyname(host)
            return
        except OSError:
            if time.time() > deadline:
                raise RuntimeError(f"{host} has not resolved for {limit_s} s: stopping") from None
            print(f"waiting for the network ({host} does not resolve)", flush=True)
            time.sleep(30)


def record_cost(record: dict) -> float:
    """Cost of a request from the server's log: each model call at its price (thinking counted as output).
    Calls without token counts (embeddings: the server logs none) and cache hits count zero."""
    from jm_eval.llm import cost_usd

    total = 0.0
    for c in record["llm_calls"]:
        if c.get("cache_hit"):
            continue
        out = (c.get("candidates_token_count") or 0) + (c.get("thoughts_token_count") or 0)
        total += cost_usd(c.get("model"), c.get("prompt_token_count") or 0, out) or 0.0
    return total
