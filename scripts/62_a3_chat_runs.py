"""Phase 6, A3 (task 3): the chatbot's answers to the question set.

Each question is sent to the local version 2 server (EVAL_MODE=1) with the 10 decisions of its case
set as `case_hashes` and `top_k` = 10, so the whole set is the context: one run of each templated
and unanswerable question, three runs of each open question. At most four calls run at a time.

The evaluated set is read only from the frozen file (scripts/61 --freeze); `--dev` runs the
development questions instead. Every answer is cached on disk (keyed by the version tag, the
request and the run number), so a rerun sends only what is missing; a failed request is recorded
and not cached. The backend must be checked out at the frozen tag, with no uncommitted change.

Outputs (never overwritten):
  data/v2/a3/answers.parquet          one row per answer: text, tokens, latency, cost, credentials
  data/v2/a3/chat_runs_<time>/        server log and evaluation log of the run
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from jm_eval.cache import LLMCache
from jm_eval.chatbot import store
from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.llm import cost_usd
from jm_eval.manifest import RunManifest, git_state
from jm_eval.server import LocalServer

SCRIPT = "62_a3_chat_runs"


def plan(questions: list[dict], runs: dict) -> list[dict]:
    return [{"question": q, "run": r} for q in questions for r in range(1, runs[q["type"]] + 1)]


def request_body(question: dict, system: dict) -> dict:
    return {"user_message": question["question"], "case_hashes": question["case_hashes"],
            "top_k": system["top_k"], "context_mode": system["context_mode"]}


def ask_server(server: LocalServer, body: dict) -> tuple[str, dict]:
    """One chatbot call: (answer, meta from the server's evaluation log). Raises on any failure."""
    response = server.session.post(server.url("/API_chatbot"), json=body, timeout=900)
    record = server.eval_log(response.headers["X-Eval-Request-Id"], timeout=60)
    answer = response.json().get("assistant_message", "") if response.ok else ""
    chat = [c for c in record["llm_calls"] if c.get("role") == "chat"]
    if not response.ok or not answer or record["errors"] or not chat:
        raise RuntimeError(f"status {response.status_code}, errors {record['errors']}: {response.text[:200]}")
    totals = record["totals"]
    output = (totals["candidates_token_count"] or 0) + (totals["thoughts_token_count"] or 0)
    model = chat[-1]["model"]
    return answer, {
        "request_id": record["request_id"], "model": model, "latency_ms": record["latency_ms"],
        "llm_calls": totals["llm_calls"], "chat_attempts": len(chat), "prompt_tokens": totals["prompt_token_count"],
        "output_tokens": output, "thoughts_tokens": totals["thoughts_token_count"],
        "cost_usd": cost_usd(model, totals["prompt_token_count"], output),
        "finish_reason": chat[-1].get("finish_reason"), "credentials_used": record["credentials_used"],
        "retrieved_hashes": record["flags"]["retrieved_hashes"], "top_k_effective": record["flags"]["top_k_effective"],
        "chat_strategy": record["flags"].get("chat_strategy"), "quote_check": record["flags"].get("quote_check"),
        "draft_chars": record["flags"].get("draft_chars"), "started_utc": record["started_utc"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true", help="the development questions (one run each)")
    ap.add_argument("--candidate", help="Phase 6.1: a strategy of config/a3_candidates.yaml, on the development sets")
    ap.add_argument("--limit", type=int, help="send at most N calls that are not cached (pilot)")
    ap.add_argument("--only", nargs="+", help="only these question IDs (pilot); the answers table holds them alone")
    ap.add_argument("--setting", help="Phase 7: a context size of config/a5_context_size.yaml, on the evaluated questions")
    args = ap.parse_args()
    cfg = load_config("a3_chatbot")
    args.dev = args.dev or bool(args.candidate)
    store.use_candidate(args.candidate)
    store.use_setting(args.setting)
    system = cfg["system"]
    if args.setting:                         # Phase 7: the frozen system with another context size, one run each
        if args.dev or args.candidate:
            raise SystemExit("--setting runs the evaluated questions only")
        setting = load_config("a5_context_size")["settings"][args.setting]
        if setting.get("reuse_phase6"):
            raise SystemExit(f"{args.setting} reuses Phase 6's answers (scripts/72); nothing to run")
        system = {**system, **{k: setting[k] for k in ("top_k", "context_mode", "concurrency")},
                  "runs": {t: load_config("a5_context_size")["runs"] for t in system["runs"]}}
    work = store.work_dir(args.dev)
    backend = git_state(resolve(load_config("paths")["repos"]["backend"]))
    strategy = None
    if args.candidate:                       # the strategies live on a branch of the backend, not at the frozen tag
        candidates = load_config("a3_candidates")
        strategy = {"chat_strategy": args.candidate, "version": candidates["candidates"][args.candidate]["version"]}
        if backend.get("branch") != candidates["backend_branch"] or backend.get("dirty"):
            raise SystemExit(f"the backend must be on {candidates['backend_branch']} with no uncommitted change; it is at {backend}")
    elif backend.get("tag") != system["tag"] or backend.get("dirty"):
        raise SystemExit(f"the backend must be at {system['tag']} with no uncommitted change; it is at {backend}")

    questions = store.load_questions(args.dev)
    if args.only:
        questions = [q for q in questions if q["question_id"] in set(args.only)]
    runs = {k: 1 for k in system["runs"]} if args.dev else system["runs"]
    jobs = plan(questions, runs)
    cache = LLMCache()
    cache_model = (f"jurismap/{candidates['backend_branch']}{system['endpoint']}@{strategy['version']}" if strategy
                   else f"jurismap/{system['tag']}{system['endpoint']}")
    run_dir = work / f"chat_runs_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    with RunManifest(SCRIPT, run_dir, models={"system_under_test": system},
                     params={"dev": args.dev, "runs": runs, "questions": len(questions), "limit": args.limit,
                             "only": args.only, "candidate": strategy, "setting": args.setting}) as run:
        for job in jobs:
            job["body"] = request_body(job["question"], system)
            if strategy:
                job["body"]["chat_strategy"] = strategy["chat_strategy"]
            job["hit"] = cache.get(cache_model, job["body"], {"run": job["run"]})
        missing = [j for j in jobs if j["hit"] is None][:args.limit]
        failures: list[dict] = []
        if missing:
            with LocalServer(run_dir) as server:
                run.data["params"]["server_env"] = server.env_overrides

                def one(job: dict) -> None:
                    try:
                        answer, meta = ask_server(server, job["body"])
                    except Exception as e:                                   # a failed request is a failure, not an answer
                        failures.append({"question_id": job["question"]["question_id"], "run": job["run"], "error": str(e)[:300]})
                        return
                    cache.put(cache_model, job["body"], {"run": job["run"]}, answer, meta)

                with ThreadPoolExecutor(max_workers=system["concurrency"]) as pool:
                    list(pool.map(one, missing))

        rows = []
        for job in jobs:
            hit = cache.get(cache_model, job["body"], {"run": job["run"]})
            q = job["question"]
            if hit is None:
                continue
            meta = hit.meta
            assert sorted(meta["retrieved_hashes"]) == sorted(q["case_hashes"]), "the context is not the case set"
            rows.append({"answer_id": f"{q['question_id']}#{job['run']}", "question_id": q["question_id"],
                         "set_id": q["set_id"], "type": q["type"], "subtype": q["subtype"], "run": job["run"],
                         "answer": hit.response, "generated_in_this_run": job["hit"] is None,
                         "credentials_used": json.dumps(meta["credentials_used"]),
                         "chat_strategy": meta.get("chat_strategy"), "quote_check": json.dumps(meta.get("quote_check")),
                         **{k: meta[k] for k in ("request_id", "model", "latency_ms", "llm_calls", "chat_attempts",
                                                 "prompt_tokens", "output_tokens", "thoughts_tokens", "cost_usd",
                                                 "finish_reason", "top_k_effective", "started_utc")},
                         **({"setting": args.setting, "context_mode": system["context_mode"]} if args.setting else {})})
        answers = pd.DataFrame(rows)
        path = new_artefact_path(work / "answers.parquet")
        answers.to_parquet(path, index=False)
        run.add_output(path, "A3: chatbot answers" + (" (development set)" if args.dev else ""))
        fallback = sum(sum(v for k, v in json.loads(c).items() if k != "primary") for c in answers["credentials_used"])
        summary = {"planned": len(jobs), "answers": len(answers), "sent_in_this_run": len(missing),
                   "failures": failures, "prompt_tokens": int(answers["prompt_tokens"].sum()),
                   "output_tokens": int(answers["output_tokens"].sum()), "cost_usd": round(float(answers["cost_usd"].sum()), 4),
                   "latency_s_median": round(float(answers["latency_ms"].median()) / 1000, 1),
                   "prompt_tokens_median": int(answers["prompt_tokens"].median()),
                   "model_calls_not_served_by_the_primary_key": int(fallback)}
        run.data["summary"] = summary
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 1 if failures or len(answers) < len(jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
