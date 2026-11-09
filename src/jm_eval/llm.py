"""LLM judges (config/models.yaml `judges`) and labelling models (`labelers`).

    judge = get_judge("judge_a")           # judge_a: GPT-5.4 (Azure OpenAI), judge_b: Mistral Medium 3,
    result = judge.complete(prompt, json_output=True)   # judge_gemini: gemini-3.8-flash (reported separately)
    result.text, result.parsed, result.cost_usd, result.cached

Mistral, Gemini and Grok (`judge_grok`, judge A until 2026-10-11) are served on Vertex AI; GPT-5.4 on
an Azure OpenAI deployment whose endpoint and key are read from this repo's git-ignored .env.

Every call goes through the on-disk cache (data/cache/llm/), keyed by (model, prompt, params), so
reruns are free and repeatable; the cache entry keeps the token usage, cost and latency (ground
rule 6). Calls are spaced to the judge's requests-per-minute quota and retried on 429/5xx.

`get_model("labelers", name)` gives a labelling model (Phase 4a category labels) with the same
caching, pacing and cost log.

Never send SUSEP texts to a judge (ground rule 1): only public CRSNSP material and generated text.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from jm_eval.cache import LLMCache
from jm_eval.io import load_config, resolve

# Fallback Google Cloud project for Vertex AI calls; JM_EVAL_GCP_PROJECT overrides it.
GCP_PROJECT = os.environ.get("JM_EVAL_GCP_PROJECT", "your-gcp-project")
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass
class JudgeResult:
    judge: str
    model: str
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float | None = None
    cached: bool = False
    credentials: str | None = None
    parsed: Any = None
    meta: dict = field(default_factory=dict)


def parse_json(text: str) -> Any:
    """Parse a JSON answer, tolerating a ```json fence or text around one JSON object."""
    match = FENCE_RE.match(text or "")
    body = match.group(1) if match else (text or "").strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            return json.loads(body[start:end + 1])
        raise


FLEX_TRAFFIC = "ON_DEMAND_FLEX"     # the traffic type Vertex AI reports for a Flex PayGo call
FLEX_HEADERS = {"X-Vertex-AI-LLM-Request-Type": "shared", "X-Vertex-AI-LLM-Shared-Request-Type": "flex"}


def cost_usd(model: str, input_tokens: int | None, output_tokens: int | None, traffic_type: str | None = None) -> float | None:
    """Cost from config/prices.yaml (output tokens include reasoning); None if the price is unknown.
    A call Vertex AI served as Flex PayGo (traffic type ON_DEMAND_FLEX) gets prices.yaml's flex_discount."""
    prices = load_config("prices")
    price = (prices.get("prices") or {}).get(model)
    if not price or input_tokens is None or output_tokens is None:
        return None
    cost = input_tokens / 1e6 * price["input"] + output_tokens / 1e6 * price["output"]
    if traffic_type and FLEX_TRAFFIC in str(traffic_type):
        cost *= 1 - prices["flex_discount"]["share"]
    return round(cost, 6)


class RateLimiter:
    """Spaces request starts to at most `rpm` per minute (thread-safe)."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)


_gcloud_token: tuple[float, str] | None = None
_token_lock = threading.Lock()


def _forget_token() -> None:
    global _gcloud_token
    with _token_lock:
        _gcloud_token = None


def _google_token() -> str:
    """Access token of the application-default credentials.

    If they need a new interactive login, the token of the account logged in with `gcloud auth login`
    is used instead (same user, same project in the request URL). It is kept for 5 minutes and dropped
    when a call answers 401, so an expired token is replaced.
    """
    import google.auth
    import google.auth.exceptions
    import google.auth.transport.requests

    global _gcloud_token
    with _token_lock:
        if _gcloud_token and time.monotonic() - _gcloud_token[0] < 300:
            return _gcloud_token[1]
        try:
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(google.auth.transport.requests.Request())
            return credentials.token
        except google.auth.exceptions.RefreshError:
            import subprocess

            out = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
            if out.returncode != 0 or not out.stdout.strip():
                raise
            _gcloud_token = (time.monotonic(), out.stdout.strip())
            return _gcloud_token[1]


class Judge:
    """Base class: caching, rate limiting, retries and cost; subclasses implement `_call`."""

    retry_status = RETRY_STATUS

    def __init__(self, name: str, config: dict, cache: LLMCache | None = None, max_attempts: int = 6):
        self.name = name
        self.config = config
        self.model = config["model"]
        self.cache = cache or LLMCache()
        self.limiter = RateLimiter((config.get("quota") or {}).get("rpm", 0))
        self.max_attempts = config.get("max_attempts", max_attempts)
        self.timeout_s = config.get("timeout_s", 600)

    def params(self) -> dict:
        """Generation parameters (part of the cache key)."""
        return {}

    def _call(self, prompt: str, system: str | None) -> dict:
        """One API call -> {"text", "input_tokens", "output_tokens", "reasoning_tokens", "credentials"}."""
        raise NotImplementedError

    def _call_with_retries(self, prompt: str, system: str | None) -> dict:
        delay = 2.0
        for attempt in range(1, self.max_attempts + 1):
            self.limiter.wait()
            try:
                return self._call(prompt, system)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 401:                       # expired token: fetch a new one and try again
                    _forget_token()
                if status not in self.retry_status | {401} or attempt == self.max_attempts:
                    raise
            except (requests.ConnectionError, requests.Timeout):   # a call that hangs is cut off and repeated
                if attempt == self.max_attempts:
                    raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

    def complete(self, prompt: str, *, system: str | None = None, json_output: bool = False,
                 cache_extra: dict | None = None) -> JudgeResult:
        """`cache_extra` is added to the cache key only (e.g. {"run": 2} for an independent repeat)."""
        cache_prompt = {"system": system, "prompt": prompt}
        cache_params = {**self.params(), **(cache_extra or {})}

        def run() -> tuple[str, dict]:
            t0 = time.perf_counter()
            out = self._call_with_retries(prompt, system)
            meta = {k: out.get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "credentials")}
            if out.get("traffic_type"):
                meta["traffic_type"] = out["traffic_type"]           # e.g. ON_DEMAND_FLEX (Flex PayGo)
            if out.get("model_version"):
                meta["model_version"] = out["model_version"]
            meta["latency_s"] = round(time.perf_counter() - t0, 2)
            meta["cost_usd"] = cost_usd(self.model, out.get("input_tokens"), out.get("output_tokens"), out.get("traffic_type"))
            meta["judge"] = self.name
            return out["text"], meta

        text, meta, cached = self.cache.call(self.model, cache_prompt, cache_params, run)
        result = JudgeResult(judge=self.name, model=self.model, text=text, cached=cached, meta=meta,
                             **{k: meta.get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens",
                                                         "cost_usd", "latency_s", "credentials")})
        if json_output:
            result.parsed = parse_json(text)
        return result


_primary_credentials: Any = None


def _primary_target() -> tuple[str, str] | None:
    """(project, access token) of the primary service-account key of config/paths.yaml; None without a key."""
    import google.auth.transport.requests
    from google.oauth2 import service_account

    global _primary_credentials
    key = load_config("paths").get("gemini_primary_credentials")
    if not key or not resolve(key).exists():
        return None
    with _token_lock:
        if _primary_credentials is None:
            _primary_credentials = service_account.Credentials.from_service_account_file(
                str(resolve(key)), scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not _primary_credentials.valid:
            _primary_credentials.refresh(google.auth.transport.requests.Request())
        return _primary_credentials.project_id, _primary_credentials.token


_key_credentials: dict = {}


def _key_token(key: str) -> str:
    """Access token of a service-account key file (config path), refreshed when it expires."""
    import google.auth.transport.requests
    from google.oauth2 import service_account

    with _token_lock:
        credentials = _key_credentials.get(key)
        if credentials is None:
            credentials = _key_credentials[key] = service_account.Credentials.from_service_account_file(
                str(resolve(key)), scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not credentials.valid:
            credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token


class PartnerModelJudge(Judge):
    """Model Garden partner model called over REST in a project of ours.

    `credentials` in the config decides which project is called and billed:
    - `primary_only`: the primary key's project and no other. An error is retried there by the
      caller; without a usable key the call fails.
    - `primary_with_fallback`: the primary key's project first, the fallback project on any error, as the
      Gemini calls do. With `primary_recheck_s` in the config, a failed call marks the primary
      project as not answering: calls then go straight to the fallback project, and every
      `primary_recheck_s` seconds a small probe asks the primary project again; when it answers,
      calls return to it.
    - `project_only`: the project named by `project` in the config and no other (the author, 2026-10-18:
      Mistral through your-mistral-gcp-project), with the service-account key at `key` if one is given,
      else the user's own credentials. No fallback: an error is retried there by the caller.
    - unset: the fallback project only.
    """

    # model -> monotonic time of the next probe of the primary project (absent: the project answers)
    _primary_down: dict[str, float] = {}
    _primary_lock = threading.Lock()

    def _probe_body(self) -> dict | None:
        """A minimal request that shows whether the model answers; None if the API has no cheap one."""
        return None

    def _primary_answers(self, url: str, token: str) -> bool:
        """False while the primary project is marked as not answering; probes it when a recheck is due."""
        recheck = self.config.get("primary_recheck_s")
        cls = PartnerModelJudge
        if not recheck or self.model not in cls._primary_down:
            return True
        with cls._primary_lock:
            due = self.model in cls._primary_down and time.monotonic() >= cls._primary_down[self.model]
            if due:                                   # this caller probes; the others keep to the fallback meanwhile
                cls._primary_down[self.model] = time.monotonic() + recheck
        if not due:
            return self.model not in cls._primary_down
        body = self._probe_body()
        try:
            if body is None:
                return False
            r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=body,
                              timeout=self.config.get("primary_probe_timeout_s", 60))
            r.raise_for_status()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
            return False
        with cls._primary_lock:
            cls._primary_down.pop(self.model, None)
        return True

    def _mark_primary_down(self) -> None:
        recheck = self.config.get("primary_recheck_s")
        if recheck:
            with PartnerModelJudge._primary_lock:
                PartnerModelJudge._primary_down[self.model] = time.monotonic() + recheck

    def _post(self, url_of, body: dict) -> tuple[requests.Response, str]:
        """POST `body` to `url_of(project)`: (successful response, label of the credentials that served it)."""
        mode, targets = self.config.get("credentials"), []
        if mode == "project_only":
            project = self.config.get("project")
            if not project:
                raise RuntimeError(f"{self.name}: credentials project_only needs a `project`")
            targets.append((project, project, _key_token(self.config["key"]) if self.config.get("key") else None))
        elif mode in ("primary_only", "primary_with_fallback"):
            try:
                primary = _primary_target()
            except Exception:                           # unusable key
                primary = None
            if primary and (mode == "primary_only" or self._primary_answers(url_of(primary[0]), primary[1])):
                targets.append(("primary", *primary))
            elif mode == "primary_only":
                raise RuntimeError(f"{self.name} is set to the primary key only, and the key is not usable")
        if mode not in ("primary_only", "project_only"):
            targets.append(("the fallback project", GCP_PROJECT, None))
        for label, project, token in targets:
            timeout = self.config.get("primary_timeout_s", self.timeout_s) if label == "primary" else self.timeout_s
            try:
                r = requests.post(url_of(project), headers={"Authorization": f"Bearer {token or _google_token()}"},
                                  json=body, timeout=timeout)
                r.raise_for_status()
                return r, label
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
                if label == targets[-1][0]:             # the last credentials: the caller retries or gives up
                    raise
                self._mark_primary_down()
        raise RuntimeError("unreachable")


class OpenAIChatJudge(PartnerModelJudge):
    """Vertex AI Model Garden partner model on the OpenAI-compatible endpoint (e.g. Grok)."""

    def _probe_body(self) -> dict:
        return {"model": self.model, "messages": [{"role": "user", "content": "Answer with the single word: ok"}]}

    def _call(self, prompt: str, system: str | None) -> dict:
        region = self.config.get("region", "global")
        host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r, credentials = self._post(
            lambda project: f"https://{host}/v1/projects/{project}/locations/{region}/endpoints/openapi/chat/completions",
            {"model": self.model, "messages": messages, **self.params()})
        body = r.json()
        usage = body.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        # completion_tokens excludes reasoning here (total = prompt + completion + reasoning); both are billed
        output = (usage.get("completion_tokens") or 0) + reasoning
        return {"text": body["choices"][0]["message"]["content"], "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": output, "reasoning_tokens": reasoning, "credentials": credentials}


class RawPredictJudge(PartnerModelJudge):
    """Vertex AI Model Garden partner model via rawPredict (e.g. Mistral)."""

    def params(self) -> dict:
        return {"temperature": 0}

    def _call(self, prompt: str, system: str | None) -> dict:
        region = self.config["region"]
        publisher = self.config.get("publisher", "mistralai")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r, credentials = self._post(
            lambda project: (f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}"
                             f"/publishers/{publisher}/models/{self.model}:rawPredict"),
            {"model": self.model, "messages": messages, **self.params()})
        body = r.json()
        usage = body.get("usage") or {}
        return {"text": body["choices"][0]["message"]["content"], "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"), "reasoning_tokens": None,
                "credentials": credentials}


class AzureOpenAIJudge(Judge):
    """A model deployed on Azure OpenAI (chat completions, v1 API).

    The endpoint and the key come from the git-ignored .env of this repo (`azure_openai_env` in
    config/paths.yaml: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY); they are never logged or cached.
    `model` in the config is the deployment name. Output tokens include the reasoning tokens.
    An error is raised with the service's own code and message and without the URL, which names the
    resource. The deployment answered 403 once in some 1,500 calls and was fine a minute later, so a
    403 is repeated like a 429.
    """

    retry_status = RETRY_STATUS | {403}

    def params(self) -> dict:
        effort = self.config.get("reasoning_effort")
        return {"reasoning_effort": effort} if effort else {}

    def _target(self) -> tuple[str, str]:
        from dotenv import dotenv_values

        values = dotenv_values(resolve(load_config("paths").get("azure_openai_env", ".env")))
        endpoint, key = values.get("AZURE_OPENAI_ENDPOINT"), values.get("AZURE_OPENAI_KEY")
        if not endpoint or not key:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY are not set in the .env file")
        return endpoint.strip().rstrip("/") + "/openai/v1/chat/completions", key.strip()

    def _call(self, prompt: str, system: str | None) -> dict:
        url, key = self._target()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = requests.post(url, headers={"api-key": key}, json={"model": self.model, "messages": messages, **self.params()},
                          timeout=self.timeout_s)
        if not r.ok:
            try:
                error = r.json().get("error") or {}
            except ValueError:
                error = {}
            detail = f"{error.get('code', '')}: {str(error.get('message', ''))[:200]}".replace(key, "<key>")
            raise requests.HTTPError(f"Azure OpenAI answered {r.status_code} ({detail})", response=r)
        body = r.json()
        usage = body.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        return {"text": body["choices"][0]["message"]["content"] or "", "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"), "reasoning_tokens": reasoning,
                "credentials": "azure", "model_version": body.get("model")}


class GeminiJudge(Judge):
    """Gemini via google-genai: the primary service-account key first, the fallback project on errors.

    With `flex: true` in the config the calls ask for Vertex AI Flex PayGo (the author, 2026-10-19: the judges
    on Flex): the two request headers of FLEX_HEADERS, on the global endpoint, with a longer timeout. Flex is
    a serving tier, not a model parameter, so it is not part of the cache key; each call records the
    traffic type Vertex AI reports (ON_DEMAND_FLEX when Flex served it)."""

    def __init__(self, name: str, config: dict, cache: LLMCache | None = None, max_attempts: int = 6):
        super().__init__(name, config, cache, max_attempts)
        self._clients: list[tuple[str, Any]] | None = None

    def params(self) -> dict:
        params = {"thinking_level": self.config.get("thinking_level")}
        for key in ("temperature", "max_output_tokens", "response_schema"):   # only when set
            if self.config.get(key) is not None:
                params[key] = self.config[key]
        return params

    def clients(self) -> list[tuple[str, Any]]:
        if self._clients is None:
            from google import genai
            from google.genai import types
            from google.oauth2 import service_account

            options = self.http_options()
            clients = []
            key = load_config("paths").get("gemini_primary_credentials")
            if key and resolve(key).exists():
                creds = service_account.Credentials.from_service_account_file(
                    str(resolve(key)), scopes=["https://www.googleapis.com/auth/cloud-platform"])
                clients.append(("primary", genai.Client(vertexai=True, project=creds.project_id, location="global",
                                                        credentials=creds, http_options=options)))
            clients.append(("fallback", genai.Client(vertexai=True, project=GCP_PROJECT, location="global",
                                                     http_options=options)))
            self._clients = clients
        return self._clients

    def http_options(self):
        from google.genai import types

        flex = bool(self.config.get("flex"))
        return types.HttpOptions(
            timeout=1_200_000 if flex else 600_000,                 # Flex is slower (up to 30 min allowed)
            headers=dict(FLEX_HEADERS) if flex else None,
            retry_options=types.HttpRetryOptions(attempts=self.max_attempts, initial_delay=2.0, max_delay=60.0,
                                                 http_status_codes=sorted(RETRY_STATUS)))

    def _call(self, prompt: str, system: str | None) -> dict:
        from google.genai import types

        schema = self.config.get("response_schema")
        config = types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_level=self.config.get("thinking_level")),
            temperature=self.config.get("temperature"),
            max_output_tokens=self.config.get("max_output_tokens"),
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
        )
        clients = self.clients()
        for i, (label, client) in enumerate(clients):
            try:
                response = client.models.generate_content(model=self.model, contents=prompt, config=config)
            except Exception:
                if i == len(clients) - 1:
                    raise
                continue
            usage = response.usage_metadata
            output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            traffic = getattr(usage, "traffic_type", None)
            return {"text": response.text or "", "input_tokens": usage.prompt_token_count,
                    "output_tokens": output, "reasoning_tokens": usage.thoughts_token_count, "credentials": label,
                    "traffic_type": getattr(traffic, "value", None) or (str(traffic) if traffic else None)}
        raise RuntimeError("no Gemini client")


API_CLASSES = {"openai_chat": OpenAIChatJudge, "raw_predict": RawPredictJudge, "genai": GeminiJudge,
               "azure_openai": AzureOpenAIJudge}


def get_model(section: str, name: str, cache: LLMCache | None = None, **overrides: Any) -> Judge:
    """The model `name` of a config/models.yaml section; `overrides` replace config values for this run."""
    config = {**load_config("models")[section][name], **overrides}
    if not config.get("model"):
        raise ValueError(f"{section}.{name} has no model configured")
    return API_CLASSES[config["api"]](name, config, cache)


def get_judge(name: str, cache: LLMCache | None = None) -> Judge:
    return get_model("judges", name, cache)
