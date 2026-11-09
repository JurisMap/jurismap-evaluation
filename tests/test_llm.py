"""Tests for jm_eval.llm (no network: the API call is replaced)."""

import pytest

from jm_eval import llm
from jm_eval.cache import LLMCache


def test_parse_json_with_fence_and_surrounding_text():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Resposta: {"a": 2} fim') == {"a": 2}
    with pytest.raises(ValueError):
        llm.parse_json("sem json")


def test_cost_from_prices():
    assert llm.cost_usd("gemini-3.8-flash", 1_000_000, 1_000_000) == pytest.approx(4.5)
    assert llm.cost_usd("unknown-model", 10, 10) is None


def test_all_judges_configured():
    for name, api in (("judge_a", llm.AzureOpenAIJudge), ("judge_grok", llm.OpenAIChatJudge),
                      ("judge_b", llm.RawPredictJudge), ("judge_gemini", llm.GeminiJudge)):
        assert isinstance(llm.get_judge(name, cache=LLMCache("/tmp/unused")), api)
    assert llm.get_judge("judge_gemini").model == "gemini-3.8-flash"


def test_complete_caches_and_reports_usage(tmp_path, monkeypatch):
    judge = llm.get_judge("judge_b", cache=LLMCache(tmp_path))
    calls = []

    def fake_call(prompt, system):
        calls.append(prompt)
        return {"text": '```json\n{"ok": true}\n```', "input_tokens": 1000, "output_tokens": 500,
                "reasoning_tokens": None, "credentials": "the fallback project"}

    monkeypatch.setattr(judge, "_call", fake_call)
    judge.limiter = llm.RateLimiter(0)
    first = judge.complete("pergunta", json_output=True)
    second = judge.complete("pergunta", json_output=True)
    assert first.parsed == {"ok": True} and not first.cached
    assert second.cached and second.text == first.text and len(calls) == 1
    assert first.cost_usd == pytest.approx(1000 / 1e6 * 0.40 + 500 / 1e6 * 2.00)


def test_retries_on_429(tmp_path, monkeypatch):
    import requests

    judge = llm.get_judge("judge_a", cache=LLMCache(tmp_path))
    judge.limiter = llm.RateLimiter(0)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    responses = iter([429, 429, None])

    def fake_call(prompt, system):
        status = next(responses)
        if status:
            err = requests.HTTPError()
            err.response = type("R", (), {"status_code": status})()
            raise err
        return {"text": "ok", "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(judge, "_call", fake_call)
    assert judge.complete("x").text == "ok"


def test_gemini_judge_falls_back(tmp_path, monkeypatch):
    judge = llm.get_judge("judge_gemini", cache=LLMCache(tmp_path))
    judge.limiter = llm.RateLimiter(0)

    class Usage:
        prompt_token_count, candidates_token_count, thoughts_token_count = 100, 20, 30

    class Models:
        def __init__(self, fail):
            self.fail = fail

        def generate_content(self, **kwargs):
            if self.fail:
                raise RuntimeError("403")
            return type("Resp", (), {"text": "OK", "usage_metadata": Usage()})()

    judge._clients = [("primary", type("C", (), {"models": Models(True)})()),
                      ("fallback", type("C", (), {"models": Models(False)})())]
    result = judge.complete("x")
    assert result.text == "OK" and result.credentials == "fallback"
    assert result.output_tokens == 50 and result.reasoning_tokens == 30


def _fake_post(calls, fail_projects=()):
    import requests

    def post(url, headers=None, json=None, timeout=None):
        project = url.split("/projects/")[1].split("/")[0]
        calls.append((project, headers["Authorization"]))
        response = requests.Response()
        if project in fail_projects:
            response.status_code = 404
            raise requests.HTTPError(response=response)
        response.status_code = 200
        response._content = (b'{"choices": [{"message": {"content": "ok"}}], '
                             b'"usage": {"prompt_tokens": 5, "completion_tokens": 2}}')
        return response
    return post


def test_partner_model_goes_to_the_primary_project_first_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_primary_target", lambda: ("primary-project", "primary-token"))
    monkeypatch.setattr(llm, "_google_token", lambda: "own-token")
    grok = llm.get_model("judges", "judge_grok", cache=LLMCache(tmp_path), credentials="primary_with_fallback")
    grok.limiter = llm.RateLimiter(0)

    calls = []
    monkeypatch.setattr(llm.requests, "post", _fake_post(calls))
    result = grok.complete("first")
    assert result.credentials == "primary" and calls == [("primary-project", "Bearer primary-token")]

    calls.clear()
    monkeypatch.setattr(llm.requests, "post", _fake_post(calls, fail_projects={"primary-project"}))
    result = grok.complete("second")
    assert result.credentials == "the fallback project"
    assert calls == [("primary-project", "Bearer primary-token"), (llm.GCP_PROJECT, "Bearer own-token")]

    calls.clear()
    monkeypatch.setattr(llm, "_primary_target", lambda: None)                # no key on this machine
    assert grok.complete("third").credentials == "the fallback project" and [c[0] for c in calls] == [llm.GCP_PROJECT]


def test_primary_only_never_uses_another_project(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(llm, "_primary_target", lambda: ("primary-project", "primary-token"))
    monkeypatch.setattr(llm, "_google_token", lambda: "own-token")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    grok = llm.get_model("judges", "judge_grok", cache=LLMCache(tmp_path), credentials="primary_only", max_attempts=2)
    grok.limiter = llm.RateLimiter(0)

    calls = []
    monkeypatch.setattr(llm.requests, "post", _fake_post(calls))
    assert grok.complete("first").credentials == "primary" and calls == [("primary-project", "Bearer primary-token")]

    calls.clear()                                                             # an error is not sent elsewhere
    monkeypatch.setattr(llm.requests, "post", _fake_post(calls, fail_projects={"primary-project"}))
    with pytest.raises(requests.HTTPError):
        grok.complete("second")
    assert calls == [("primary-project", "Bearer primary-token")]

    monkeypatch.setattr(llm, "_primary_target", lambda: None)                # no key: the call fails
    calls.clear()
    with pytest.raises(RuntimeError):
        grok.complete("third")
    assert calls == []


def test_grok_leaves_the_primary_project_while_it_does_not_answer_and_returns_to_it(tmp_path, monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(llm, "_primary_target", lambda: ("primary-project", "primary-token"))
    monkeypatch.setattr(llm, "_google_token", lambda: "own-token")
    monkeypatch.setattr(llm.PartnerModelJudge, "_primary_down", {})
    grok = llm.get_judge("judge_grok", cache=LLMCache(tmp_path))
    grok.limiter = llm.RateLimiter(0)
    assert grok.config["credentials"] == "primary_with_fallback" and grok.config["primary_recheck_s"] == 180
    projects = lambda calls: [c[0] for c in calls]                           # noqa: E731

    calls = []
    monkeypatch.setattr(llm.requests, "post", _fake_post(calls, fail_projects={"primary-project"}))
    assert grok.complete("a").credentials == "the fallback project" and projects(calls) == ["primary-project", llm.GCP_PROJECT]

    calls.clear()                                                             # marked as not answering: straight to the fallback project
    clock["now"] += 100
    assert grok.complete("b").credentials == "the fallback project" and projects(calls) == [llm.GCP_PROJECT]

    calls.clear()                                                             # 3 minutes on: one probe, still not answering
    clock["now"] += 100
    assert grok.complete("c").credentials == "the fallback project" and projects(calls) == ["primary-project", llm.GCP_PROJECT]
    calls.clear()
    assert grok.complete("d").credentials == "the fallback project" and projects(calls) == [llm.GCP_PROJECT]

    calls.clear()                                                             # next probe is answered: back to the primary
    clock["now"] += 200
    monkeypatch.setattr(llm.requests, "post", _fake_post(calls))
    assert grok.complete("e").credentials == "primary" and projects(calls) == ["primary-project", "primary-project"]
    calls.clear()
    assert grok.complete("f").credentials == "primary" and projects(calls) == ["primary-project"]


def test_partner_model_without_the_setting_uses_perfect_crow_only(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_primary_target", lambda: ("primary-project", "primary-token"))
    monkeypatch.setattr(llm, "_google_token", lambda: "own-token")
    # Mistral with no credentials setting, as it was until 2026-10-18 (now routed to your-mistral-gcp-project)
    mistral = llm.get_model("judges", "judge_b", cache=LLMCache(tmp_path), credentials=None)
    mistral.limiter = llm.RateLimiter(0)
    calls = []
    monkeypatch.setattr(llm.requests, "post", _fake_post(calls))
    assert mistral.complete("x").credentials == "the fallback project" and calls == [(llm.GCP_PROJECT, "Bearer own-token")]


def test_mistral_is_routed_to_own_project_only(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_primary_target", lambda: ("primary-project", "primary-token"))
    monkeypatch.setattr(llm, "_google_token", lambda: "own-token")
    mistral = llm.get_judge("judge_b", cache=LLMCache(tmp_path))
    mistral.limiter = llm.RateLimiter(0)
    calls = []
    monkeypatch.setattr(llm.requests, "post", _fake_post(calls))
    assert mistral.complete("x").credentials == "your-mistral-gcp-project" and calls == [("your-mistral-gcp-project", "Bearer own-token")]


def test_a_call_that_times_out_is_repeated(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(llm, "_primary_target", lambda: ("primary-project", "primary-token"))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    grok = llm.get_model("judges", "judge_grok", cache=LLMCache(tmp_path), credentials="primary_only")
    grok.limiter = llm.RateLimiter(0)
    assert grok.timeout_s == 300 and grok.max_attempts == 12
    calls, answer = [], _fake_post([])

    def post(url, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) < 3:
            raise requests.ReadTimeout("hung")
        return answer(url, **kwargs)

    monkeypatch.setattr(llm.requests, "post", post)
    assert grok.complete("x").credentials == "primary" and len(calls) == 3


def test_judge_a_is_the_azure_deployment_and_keeps_its_key_out_of_the_result(tmp_path, monkeypatch):
    import requests

    env = tmp_path / ".env"
    env.write_text("AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/\nAZURE_OPENAI_KEY=secret-key\nAZURE_OPENAI_MODEL=gpt-5.4\n")
    paths = llm.load_config("paths")
    monkeypatch.setattr(llm, "load_config", lambda name: {**paths, "azure_openai_env": str(env)} if name == "paths" else
                        __import__("jm_eval.io", fromlist=["load_config"]).load_config(name))
    judge = llm.get_judge("judge_a", cache=LLMCache(tmp_path / "cache"))
    judge.limiter = llm.RateLimiter(0)
    assert judge.model == "gpt-5.4" and judge.params() == {"reasoning_effort": "medium"}
    seen = {}

    def post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        response = requests.Response()
        response.status_code = 200
        response._content = (b'{"model": "gpt-5.4-2026-03-05", "choices": [{"message": {"content": "{\\"a\\": 1}"}}], '
                             b'"usage": {"prompt_tokens": 1000, "completion_tokens": 400, '
                             b'"completion_tokens_details": {"reasoning_tokens": 300}}}')
        return response

    monkeypatch.setattr(llm.requests, "post", post)
    result = judge.complete("claim", system="rules", json_output=True)
    assert seen["url"] == "https://example.openai.azure.com/openai/v1/chat/completions"
    assert seen["headers"] == {"api-key": "secret-key"} and seen["body"]["model"] == "gpt-5.4"
    assert seen["body"]["reasoning_effort"] == "medium" and [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]
    assert result.parsed == {"a": 1} and result.credentials == "azure"
    assert result.output_tokens == 400 and result.reasoning_tokens == 300           # reasoning is part of the output tokens
    assert result.cost_usd == pytest.approx(1000 / 1e6 * 2.5 + 400 / 1e6 * 15.0)
    assert result.meta["model_version"] == "gpt-5.4-2026-03-05"
    assert "secret-key" not in "".join(p.read_text() for p in (tmp_path / "cache").rglob("*.json"))


def test_azure_errors_are_repeated_and_do_not_show_the_endpoint_or_the_key(tmp_path, monkeypatch):
    import requests

    env = tmp_path / ".env"
    env.write_text("AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/\nAZURE_OPENAI_KEY=secret-key\n")
    paths = llm.load_config("paths")
    monkeypatch.setattr(llm, "load_config", lambda name: {**paths, "azure_openai_env": str(env)} if name == "paths" else
                        __import__("jm_eval.io", fromlist=["load_config"]).load_config(name))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    judge = llm.get_model("judges", "judge_a", cache=LLMCache(tmp_path / "cache"), max_attempts=3)
    judge.limiter = llm.RateLimiter(0)
    statuses = iter([403, 200])

    def post(url, headers=None, json=None, timeout=None):
        response = requests.Response()
        response.status_code = next(statuses)
        response._content = (b'{"error": {"code": "Forbidden", "message": "denied for secret-key"}}' if response.status_code != 200 else
                             b'{"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}')
        return response

    monkeypatch.setattr(llm.requests, "post", post)
    assert judge.complete("x").text == "ok"                                  # the 403 was repeated
    statuses = iter([403, 403, 403])
    with pytest.raises(requests.HTTPError) as error:
        judge.complete("y")
    message = str(error.value)
    assert "403" in message and "Forbidden" in message and "my-resource" not in message and "secret-key" not in message


def test_flex_paygo_headers_and_discount():
    gemini = llm.get_model("judges", "judge_gemini", cache=None)
    options = gemini.http_options()
    assert options.headers == {"X-Vertex-AI-LLM-Request-Type": "shared", "X-Vertex-AI-LLM-Shared-Request-Type": "flex"}
    assert options.timeout == 1_200_000
    plain = llm.get_model("judges", "judge_gemini", cache=None, flex=False).http_options()
    assert plain.headers is None and plain.timeout == 600_000
    full = llm.cost_usd("gemini-3.8-flash", 1_000_000, 1_000_000)
    assert llm.cost_usd("gemini-3.8-flash", 1_000_000, 1_000_000, "ON_DEMAND_FLEX") == round(full / 2, 6)
    assert llm.cost_usd("gemini-3.8-flash", 1_000_000, 1_000_000, "ON_DEMAND") == full   # no discount unless Flex served it
