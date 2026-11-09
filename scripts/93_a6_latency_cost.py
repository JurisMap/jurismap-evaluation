"""Phase 9, A6: latency and cost of the map view, case upload, chat and defense endpoints (local v2.7 server).

One local server (backend `main` at tag v2.7, EVAL_MODE=1), one request at a time. After every server
start, 3 warm-up rounds (development notices and questions; not reported). Then 30 measured rounds
(config/a6_latency_cost.yaml): a map view, the upload of the round's test notice, one frozen chat
question, and one draft of each defense arm in the round's order. Calls already measured are skipped,
so a run resumes; a call that failed only on this machine's network is sent again, any other failure
is a measured failure.

  --pilot   the warm-up rounds only, then the projection of the phase's cost (budget rule 2)

Outputs:
  data/v2/a6/calls.jsonl              one row per call: latency (client and server), tokens, cost, credentials
  data/v2/a6/responses/<round>_<unit>.json   the response (draft, answer) and the server's records
  data/v2/a6/server_<time>/           server log and evaluation log of each server start
  results/a6_pilot.json               (after the first warm-up) cost per call and the projection
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import requests

from jm_eval import latency as L
from jm_eval.chatbot import store
from jm_eval.drafting import notices as N
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest, git_state
from jm_eval.server import LocalServer, network_failure, record_cost, sse_messages, wait_for_network

SCRIPT = "93_a6_latency_cost"
NETWORK_TRIES = 3
DRAFT_AGENTS = ("MergerAgent", "SinglePrompt")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def usage(records: list[dict]) -> dict:
    """Tokens, cost and credentials of one or more server records (output includes thinking)."""
    tot = lambda k: sum((r["totals"].get(k) or 0) for r in records)  # noqa: E731
    creds: dict[str, int] = {}
    for r in records:
        for k, v in (r.get("credentials_used") or {}).items():
            creds[k] = creds.get(k, 0) + v
    return {"llm_calls": tot("llm_calls"), "prompt_tokens": tot("prompt_token_count"),
            "output_tokens": tot("candidates_token_count") + tot("thoughts_token_count"),
            "thoughts_tokens": tot("thoughts_token_count"), "cost_usd": round(sum(record_cost(r) for r in records), 5),
            "credentials": creds, "fallback_calls": sum(v for k, v in creds.items() if k != "primary"),
            "server_s": round(sum(r.get("latency_ms") or 0 for r in records) / 1000, 2),
            "request_ids": [r["request_id"] for r in records], "server_errors": [e for r in records for e in r.get("errors") or []]}


class Runner:
    def __init__(self, server: LocalServer, cfg: dict):
        self.server, self.cfg = server, cfg
        self.ep = cfg["endpoints"]

    def post(self, path: str, **kw):
        return self.server.session.post(self.server.url(path), **kw)

    def map_view(self) -> tuple[bool, dict, dict]:
        r = self.post(self.ep["map_view"]["path"], json=self.ep["map_view"]["body"], timeout=300)
        body = r.json() if r.ok else {}
        records = [self.server.eval_log(r.headers["X-Eval-Request-Id"])] if "X-Eval-Request-Id" in r.headers else []
        n = len(body.get("filtered_data_ids") or [])
        return r.ok and n > 0, {"status": r.status_code, "map_points": n, **usage(records)}, {}

    def upload(self, text: str) -> tuple[bool, dict, dict]:
        up, read = self.ep["case_upload"]["paths"]
        r1 = self.post(up, files={"case_file": (self.ep["case_upload"]["upload_name"], text.encode("utf-8"), "text/plain")}, timeout=600)
        r2 = self.server.session.get(self.server.url(read), timeout=300) if r1.ok else None
        records = [self.server.eval_log(r.headers["X-Eval-Request-Id"]) for r in (r1, r2) if r is not None and "X-Eval-Request-Id" in r.headers]
        data = r2.json() if r2 is not None and r2.ok else {}
        similar = (data.get("similar_cases") or {}).get("filtered_data_ids") or []
        ok = bool(r1.ok and r2 is not None and r2.ok and len(similar) == 10)
        errors = [] if r1.ok else [r1.text[:300]]
        return ok, {"status": r1.status_code, "preselected": len(similar), "errors": errors or None, **usage(records)}, {"similar": similar}

    def chat(self, q: dict) -> tuple[bool, dict, dict]:
        spec = self.ep["chat"]
        body = {"user_message": q["question"], "case_hashes": q["case_hashes"], "top_k": spec["top_k"], "context_mode": spec["context_mode"]}
        r = self.post(spec["path"], json=body, timeout=spec["timeout_s"])
        record = self.server.eval_log(r.headers["X-Eval-Request-Id"], timeout=60)
        answer = r.json().get("assistant_message", "") if r.ok else ""
        ok = bool(r.ok and answer and not record["errors"])
        return ok, {"status": r.status_code, "answer_chars": len(answer), "chat_strategy": record["flags"].get("chat_strategy"),
                    "errors": record["errors"] or (None if r.ok else [r.text[:300]]), **usage([record])}, {"answer": answer}

    def defense(self, arm: str, cases: list[str]) -> tuple[bool, dict, dict]:
        spec = self.ep["defense"]
        a = spec["arms"][arm]
        body = {"human_message": spec["human_message"], "skip_supervisor": spec["skip_supervisor"], "case_hashes": cases, **a["body"]}
        t0 = time.perf_counter()
        r = self.post(spec["path"], stream=True, timeout=spec["timeout_s"], json=body)
        messages, first = [], None
        if r.ok:
            for m in sse_messages(r):
                if first is None:
                    first = round(time.perf_counter() - t0, 2)
                messages.append(m)
        r.close()
        record = self.server.eval_log(r.headers["X-Eval-Request-Id"], timeout=120)
        drafts = [m["content"] for m in messages if m.get("agent_id") in DRAFT_AGENTS and m.get("content")]
        draft = drafts[-1] if drafts else ""
        errors = [m.get("content") or m.get("error") for m in messages if m.get("status") == "error" or m.get("error")]
        flags = record["flags"]
        got = {"ablation_mode": flags.get("ablation_mode"), "fact_controls": flags.get("fact_controls")}
        if got != a["expect"]:
            raise RuntimeError(f"{arm}: the server ran {got}, expected {a['expect']}")
        roles = [c["role"] for c in record["llm_calls"]]
        ok = bool(r.ok and len(draft) > 1000 and not errors)
        return ok, {"status": r.status_code, "first_event_s": first, "draft_chars": len(draft), **got,
                    "blocked_retries": roles.count("agent_blocked_retry"), "legal_search": "agent:LegalSearcher" in roles,
                    "errors": errors or record.get("errors") or None, **usage([record])}, {"draft": draft, "messages": messages}


def measure(fn, *args) -> tuple[bool, dict, dict, float]:
    """Run one call, sending it again (up to NETWORK_TRIES) when it failed only on this machine's network."""
    for attempt in range(1, NETWORK_TRIES + 1):
        wait_for_network()
        started = now_utc()
        t0 = time.perf_counter()
        try:
            ok, meta, payload = fn(*args)
        except requests.exceptions.Timeout as e:          # the system did not answer in time: a measured failure
            ok, meta, payload = False, {"errors": [f"timeout: {e}"[:300]], "cost_usd": None, "fallback_calls": 0}, {}
        wall = round(time.perf_counter() - t0, 2)
        if ok or not network_failure((meta.get("errors") or []) + (meta.get("server_errors") or [])) or attempt == NETWORK_TRIES:
            return ok, {**meta, "started_utc": started, "attempts": attempt}, payload, wall
        print(f"    network failure; sending again after the network check", flush=True)
        time.sleep(60)
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="warm-up rounds and the cost projection only")
    args = ap.parse_args()
    cfg = L.settings()
    backend = git_state(resolve(load_config("paths")["repos"]["backend"]))
    if backend.get("sha") != cfg["system"]["sha"] or backend.get("dirty"):
        raise SystemExit(f"the backend must be at {cfg['system']['tag']} ({cfg['system']['sha'][:7]}) with no uncommitted change; it is at {backend}")

    work = resolve(load_config("paths")["data"]["v2"]) / "a6"
    (work / "responses").mkdir(parents=True, exist_ok=True)
    calls_path = work / "calls.jsonl"
    arms = list(cfg["endpoints"]["defense"]["arms"])

    notices = N.load_frozen()
    precedents = json.loads((resolve(load_config("paths")["data"]["v2"]) / "a4" / "precedents.json").read_text(encoding="utf-8"))
    questions = L.sample_questions(store.load_questions(), cfg["endpoints"]["chat"]["sample"], cfg["seed"])
    plan = L.round_plan(cfg, notices, questions)
    by_notice = {n["notice_id"]: n for n in notices}
    by_question = {q["question_id"]: q for q in questions}
    dev_dir = resolve(load_config("paths")["data"]["v2"]) / "a4_dev"
    dev_notices = [json.loads(x) for x in latest_artefact_path(dev_dir / "notices.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    dev_precedents = json.loads((dev_dir / "precedents.json").read_text(encoding="utf-8"))
    dev_questions = sorted(store.load_questions(dev=True), key=lambda q: q["question_id"])
    rng = np.random.default_rng(cfg["seed"])
    warm = [{"round": -(i + 1), "notice": dev_notices[i], "cases": dev_precedents[dev_notices[i]["notice_id"]]["precedents"],
             "question": dev_questions[j], "arms": [arms[i % len(arms)]]}
            for i, j in enumerate(rng.choice(len(dev_questions), size=cfg["warmup_rounds"], replace=False))]

    done = set()
    if calls_path.exists():
        done = {(r["round"], r["unit"]) for r in map(json.loads, calls_path.read_text(encoding="utf-8").splitlines()) if not r["warmup"]}
    units = lambda p: [*L.ENDPOINTS, *p["arms"]]  # noqa: E731
    todo = [p for p in plan if any((p["round"], u) not in done for u in units(p))]
    print(f"{len(plan)} rounds; {len(todo)} with calls to send ({len(done)} calls measured)", flush=True)
    if not todo and not args.pilot:
        return 0

    run_dir = work / f"server_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    template = resolve(load_config("paths")["repos"]["frontend_v1"]) / cfg["endpoints"]["defense"]["template"]
    with RunManifest(SCRIPT, work, params={"config": cfg, "plan": plan, "pilot": args.pilot, "backend": backend}) as run, \
            LocalServer(run_dir) as server:
        run.data["params"]["server_env"] = server.env_overrides
        run.add_input(template)
        server.session.post(server.url("/API_upload_defense_template"),
                            files={"defense_template": (template.name, template.read_bytes())}).raise_for_status()
        runner = Runner(server, cfg)

        def write(round_no: int, unit: str, endpoint: str, ok: bool, meta: dict, payload: dict, wall: float, **extra) -> None:
            row = {"round": round_no, "warmup": round_no < 0, "unit": unit, "endpoint": endpoint, "ok": ok, "wall_s": wall,
                   **extra, **meta, "server_dir": run_dir.name}
            row.setdefault("first_event_s", None)
            with open(calls_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if payload:
                (work / "responses" / f"{'W' if round_no < 0 else 'R'}{abs(round_no):02d}_{unit}_{run_dir.name[7:]}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {'W' if round_no < 0 else 'R'}{abs(round_no)} {unit}: ok={ok} {wall} s ${meta.get('cost_usd')}"
                  f"{' fallback=' + str(meta['fallback_calls']) if meta.get('fallback_calls') else ''}", flush=True)

        def one_round(round_no: int, notice: dict, cases: list[str], question: dict, round_arms: list[str], skip: set) -> None:
            if (round_no, "map_view") not in skip:
                write(round_no, "map_view", "map_view", *measure(runner.map_view))
            # the round's notice is the session's case for the drafts: uploaded again (not measured) if already measured
            ok, meta, payload, wall = measure(runner.upload, notice["notice"])
            if (round_no, "case_upload") not in skip:
                write(round_no, "case_upload", "case_upload", ok, meta, payload, wall, notice_id=notice["notice_id"])
            if (round_no, "chat") not in skip:
                write(round_no, "chat", "chat", *measure(runner.chat, question), question_id=question["question_id"],
                      question_type=question["type"])
            for pos, arm in enumerate(round_arms, 1):
                if (round_no, arm) not in skip:
                    write(round_no, arm, "defense", *measure(runner.defense, arm, cases), notice_id=notice["notice_id"], position=pos)

        for w in warm:
            one_round(w["round"], w["notice"], w["cases"], w["question"], w["arms"], set())

        rows = [json.loads(x) for x in calls_path.read_text(encoding="utf-8").splitlines()]
        mine = [r for r in rows if r["server_dir"] == run_dir.name and r["warmup"]]
        pilot_path = resolve(load_config("paths")["results"]) / "a6_pilot.json"
        if args.pilot or not pilot_path.exists():      # the first start: the warm-ups are the pilot (budget rule 2)
            cost = {u: float(np.mean([r["cost_usd"] for r in mine if r["unit"] == u])) for u in {r["unit"] for r in mine}}
            b = cfg["budget"]
            projected = sum(cost.values()) * cfg["rounds"] + sum(r["cost_usd"] for r in mine)
            threshold = (b["plan_line_usd"] + b["added_arms_usd"]) * b["stop_ratio"]
            pilot = {"warmup_cost_per_call_usd": {k: round(v, 4) for k, v in sorted(cost.items())},
                     "warmup_wall_s": {u: [r["wall_s"] for r in mine if r["unit"] == u] for u in sorted(cost)},
                     "projected_phase_usd": round(projected, 2), "stop_threshold_usd": threshold,
                     "within_threshold": projected <= threshold,
                     "projected_hours": round(sum(np.mean([r["wall_s"] for r in mine if r["unit"] == u]) for u in cost) * cfg["rounds"] / 3600, 1)}
            path = new_artefact_path(pilot_path)
            path.write_text(json.dumps(pilot, indent=1), encoding="utf-8")
            run.add_output(path, "A6: warm-up cost and the projection of the phase")
            print(json.dumps(pilot, indent=1), flush=True)
            if not pilot["within_threshold"]:
                print("STOP: the projection exceeds 150% of the phase's line (budget rule 2)")
                return 2
        if args.pilot:
            return 0

        for p in todo:
            notice = by_notice[p["notice_id"]]
            one_round(p["round"], notice, precedents[p["notice_id"]]["precedents"], by_question[p["question_id"]], p["arms"], done)

        rows = [json.loads(x) for x in calls_path.read_text(encoding="utf-8").splitlines()]
        mine = [r for r in rows if r["server_dir"] == run_dir.name and not r["warmup"]]
        run.add_output(calls_path, "A6: one row per call")
        run.data["summary"] = {"calls": len(mine), "ok": sum(r["ok"] for r in mine), "cost_usd": round(sum(r["cost_usd"] for r in mine), 2),
                               "fallback_calls": sum(r["fallback_calls"] for r in mine)}
        print(json.dumps(run.data["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
