"""Tests for jm_eval.server helpers (no server is started)."""

from pathlib import Path

from jm_eval import server


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def iter_lines(self, decode_unicode=True):
        yield from self.lines


def test_sse_messages_parses_data_lines():
    lines = ['data: {"role": "system", "status": "generating"}', "", ": comment",
             'data: {"agent_id": "MergerAgent", "content": "texto"}']
    messages = list(server.sse_messages(FakeResponse(lines)))
    assert [m.get("agent_id") for m in messages] == [None, "MergerAgent"]
    assert messages[1]["content"] == "texto"


def test_server_env_resolves_paths_and_never_uses_production(tmp_path):
    env = server.server_env(tmp_path, 8123)
    assert env["EVAL_MODE"] == "1"
    assert env["DATA_VERSION"] == "v2"
    assert Path(env["DATA_DIR"]).is_absolute()
    assert env["EVAL_LOG_DIR"] == str(tmp_path / "server_logs")
    assert env["PORT"] == "8123"
    assert env["SESSION_TYPE"] == "filesystem"
    assert ".crsnsp_eval." in env["BIGQUERY_TABLE"]
    assert Path(env["GEMINI_PRIMARY_CREDENTIALS"]).is_absolute()
