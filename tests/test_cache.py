from jm_eval.cache import LLMCache, cache_key


def test_key_depends_on_model_prompt_params():
    k = cache_key("m", "hello", {"temperature": 0})
    assert k == cache_key("m", "hello", {"temperature": 0})
    assert k != cache_key("m2", "hello", {"temperature": 0})
    assert k != cache_key("m", "hello!", {"temperature": 0})
    assert k != cache_key("m", "hello", {"temperature": 1})
    # param order does not matter
    assert cache_key("m", "p", {"a": 1, "b": 2}) == cache_key("m", "p", {"b": 2, "a": 1})


def test_call_runs_once_then_hits(tmp_path):
    cache = LLMCache(tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return {"text": "answer"}, {"input_tokens": 3, "output_tokens": 1}

    msgs = [{"role": "user", "content": "q"}]
    r1, m1, hit1 = cache.call("m", msgs, {"t": 0}, fn)
    r2, m2, hit2 = cache.call("m", msgs, {"t": 0}, fn)
    assert (r1, hit1) == ({"text": "answer"}, False)
    assert (r2, m2, hit2) == (r1, m1, True)
    assert len(calls) == 1


def test_put_never_overwrites(tmp_path):
    cache = LLMCache(tmp_path)
    cache.put("m", "p", None, "first")
    cache.put("m", "p", None, "second")
    assert cache.get("m", "p", None).response == "first"
