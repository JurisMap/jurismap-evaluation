"""Phase 8, A4 (task 3): defense drafts of the frozen test notices on the local v2.6 server.

For each notice: upload it as a user uploads a case (a .txt file under a neutral name; the template
is uploaded once per session), take the app's 10 pre-selected decisions without the source decision
and its process as the precedents (fixed at the notice's first upload), then generate a draft in
each mode with those precedents as the exact case set and the supervisor skipped (EVAL_MODE=1).
Drafts already written are skipped, so a run resumes, and the pilot's drafts are run 1 of their cells.

  --pilot     one notice per legal-basis key, every mode, run 1 (25 drafts); prints the projection
  (default)   every notice, every mode, runs 1 to 3 (225 drafts)
  --dev-arms  improvement round: every development notice (data/v2/a4_dev/notices.jsonl), the round's
              mode, one draft per arm of config `improvement.arms`, sent as fact_controls (backend branch
              feature/fact_premises_guard); outputs under data/v2/a4_dev/

Outputs:
  data/v2/a4/precedents.json                 per notice: pre-selected decisions, precedents, what was removed
  data/v2/a4/drafts/<notice>_<mode>_r<k>.md  the draft (and .messages.json: the stream; .record.json: the server's log)
  data/v2/a4/drafts.jsonl                    one row per draft: status, tokens, cost, latency, fact guard, retries
  results/a4_drafting_pilot.json             (pilot) cost and time per mode and the projection for the phase
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from jm_eval.drafting import notices as N
from jm_eval.io import load_config, new_artefact_path, resolve, sha256_file
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C
from jm_eval.server import LocalServer, network_failure, record_cost, sse_messages, wait_for_network

SCRIPT = "83_a4_drafts"
NETWORK_TRIES = 3
DRAFT_AGENTS = ("MergerAgent", "SinglePrompt")


def a4_dir():
    return N.work_dir()                                    # data/v2/a4 (v2.6) or data/v2/a4_v27 (JM_A4_VERSION=v2.7)


def pilot_notices(frozen: list[dict], seed: int) -> list[str]:
    """One notice per legal-basis key, drawn at random."""
    rng = np.random.default_rng(seed)
    keys = sorted({r["base_key"] for r in frozen})
    return [str(rng.choice([r["notice_id"] for r in frozen if r["base_key"] == k])) for k in keys]


def precedents_for(notice: dict, similar: dict, groups: dict) -> dict:
    """The pre-selected decisions without the source decision and the decisions of its process."""
    pre = list(similar.get("filtered_data_ids") or [])
    source_group = groups[notice["acordao_hash"]]
    removed = [h for h in pre if h == notice["acordao_hash"] or groups.get(h) == source_group]
    return {"preselected": pre, "similarity": similar.get("similarity") or [],
            "precedents": [h for h in pre if h not in removed], "removed": removed,
            "source_rank": pre.index(notice["acordao_hash"]) + 1 if notice["acordao_hash"] in pre else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", help="write again the drafts that failed (not done by default)")
    ap.add_argument("--dev-arms", action="store_true", help="improvement round: development notices x fact-control arms")
    ap.add_argument("--notices", nargs="*", help="only these notices (e.g. N10 after its correction)")
    ap.add_argument("--fact-controls", nargs="*", help="send these fact_controls ('none' for an empty list: v2.6's behaviour "
                                                       "on the v2.7 backend, whose prompts with every control off are v2.6's)")
    args = ap.parse_args()
    cfg = N.settings()
    spec = cfg["drafting"]
    arms = {"": []}                                    # the evaluated runs send no fact_controls (v2.6 behaviour)
    if args.dev_arms:
        imp = cfg["improvement"]
        base = resolve(load_config("paths")["data"]["v2"]) / "a4_dev"
        from jm_eval.io import latest_artefact_path
        frozen = [json.loads(x) for x in latest_artefact_path(base / "notices.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        spec = {**spec, "modes": [imp["mode"]], "runs": imp["runs"]}
        arms = {name: controls for name, controls in imp["arms"].items()}
    else:
        frozen = N.load_frozen(cfg)
    by_id = {r["notice_id"]: r for r in frozen}
    ids = pilot_notices(frozen, spec["pilot"]["seed"]) if args.pilot else [r["notice_id"] for r in frozen]
    if args.notices:
        ids = [i for i in ids if i in args.notices]
    sent_controls = None if args.fact_controls is None else [c for c in args.fact_controls if c != "none"]
    runs = spec["pilot"]["runs"] if args.pilot else list(range(1, spec["runs"] + 1))
    labels = C.load_labels()
    groups = dict(zip(labels["acordao_hash"], labels["process_group"]))
    work = (resolve(load_config("paths")["data"]["v2"]) / "a4_dev") if args.dev_arms else a4_dir()
    out_dir = work / "drafts"
    out_dir.mkdir(parents=True, exist_ok=True)
    precedents_path = work / "precedents.json"
    v26_precedents = resolve(load_config("paths")["data"]["v2"]) / "a4" / "precedents.json"
    if not precedents_path.exists() and N.a4_version() != "v2.6" and v26_precedents.exists():
        precedents_path.write_text(v26_precedents.read_text(encoding="utf-8"), encoding="utf-8")   # same precedents as v2.6
    precedents = json.loads(precedents_path.read_text(encoding="utf-8")) if precedents_path.exists() else {}
    rows_path = work / "drafts.jsonl"
    done = set()
    if rows_path.exists():
        # A draft that failed (e.g. a recitation block not recovered) is a measured failure: it is not
        # written again unless --retry-failed is given (plan: a request that ends in an error is a failure).
        done = {(r["notice_id"], r.get("arm", "") + r["mode"], r["run"]) for r in map(json.loads, rows_path.read_text(encoding="utf-8").splitlines())
                if r.get("ok") or not args.retry_failed}
    # A cell is (notice, mode, run); with arms the mode is prefixed by the arm ("premise_rule:full").
    cells = [(a + ":" + m) if a else m for a in arms for m in spec["modes"]]
    todo = [(n, c, k) for n in ids for k in runs for c in cells if (n, c.replace(":", ""), k) not in done]
    print(f"{'pilot' if args.pilot else 'main'}: notices {ids}, runs {runs}; {len(todo)} drafts to write ({len(done)} done)")
    if not todo:
        return 0

    run_dir = work / f"server_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    template = resolve(load_config("paths")["repos"]["frontend_v1"]) / spec["template"]
    with RunManifest(SCRIPT, work, params={"drafting": spec, "pilot": args.pilot, "notices": ids, "runs": runs, "arms": arms}) as run, \
            LocalServer(run_dir) as server:
        run.data["params"]["server_env"] = server.env_overrides
        run.add_input(template)
        s = server.session
        r = s.post(server.url("/API_upload_defense_template"), files={"defense_template": (template.name, template.read_bytes())})
        r.raise_for_status()
        for nid in ids:
            notice = by_id[nid]
            todo_n = [(m, k) for (n, m, k) in todo if n == nid]
            retry: set = set()
            if not todo_n:
                continue
            for attempt in range(1, NETWORK_TRIES + 1):        # an upload that fails on the network is retried
                wait_for_network()
                r = s.post(server.url("/API_upload_case"),
                           files={"case_file": (spec["upload_name"], notice["notice"].encode("utf-8"), "text/plain")})
                if r.ok or attempt == NETWORK_TRIES:
                    break
                print(f"{nid}: upload failed ({r.status_code}), retrying after the network check", flush=True)
                time.sleep(60)
            r.raise_for_status()
            uploaded = s.get(server.url("/API_get_uploaded_case_data")).json()
            found = precedents_for(notice, uploaded.get("similar_cases") or {}, groups)
            if nid not in precedents:                    # fixed at the first upload
                precedents[nid] = {**found, "fixed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                precedents_path.write_text(json.dumps(precedents, indent=1), encoding="utf-8")
            elif found["preselected"] != precedents[nid]["preselected"]:
                print(f"{nid}: this upload's pre-selection differs from the fixed one; the fixed precedents are used")
            cases = precedents[nid]["precedents"]
            print(f"{nid}: {len(cases)} precedents (source at rank {precedents[nid]['source_rank']})", flush=True)
            for cell, k, attempt in ((c, k, a) for c, k in todo_n for a in range(1, NETWORK_TRIES + 1)):
                arm, _, mode = cell.rpartition(":")
                if attempt > 1 and (nid, cell, k) not in retry:
                    continue                                   # written (or failed on the system) at an earlier attempt
                retry.discard((nid, cell, k))
                wait_for_network()
                t0 = time.perf_counter()
                body = {"human_message": spec["human_message"], "ablation_mode": mode,
                        "skip_supervisor": spec["skip_supervisor"], "case_hashes": cases}
                if args.dev_arms:
                    body["fact_controls"] = arms[arm]
                elif sent_controls is not None:
                    body["fact_controls"] = sent_controls
                r = s.post(server.url(spec["endpoint"]), stream=True, timeout=spec["timeout_s"], json=body)
                messages = list(sse_messages(r)) if r.ok else []
                r.close()
                record = server.eval_log(r.headers["X-Eval-Request-Id"], timeout=120)
                drafts = [m["content"] for m in messages if m.get("agent_id") in DRAFT_AGENTS and m.get("content")]
                draft = drafts[-1] if drafts else ""
                errors = [m.get("content") or m.get("error") for m in messages if m.get("status") == "error" or m.get("error")]
                if not draft and network_failure(errors) and attempt < NETWORK_TRIES:
                    # this machine lost the network: not a failure of the system; the cell is written again
                    print(f"  {nid}_{cell}_r{k}: network failure, retrying after the network check", flush=True)
                    retry.add((nid, cell, k))
                    time.sleep(60)
                    continue
                fact = [m.get("fact_check") for m in messages if m.get("agent_id") == "FactGuard"]
                stem = f"{nid}_{arm + '_' if arm else ''}{mode}_r{k}"
                (out_dir / f"{stem}.md").write_text(draft, encoding="utf-8")
                (out_dir / f"{stem}.messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=1), encoding="utf-8")
                (out_dir / f"{stem}.record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
                roles = [c["role"] for c in record["llm_calls"]]
                row = {"notice_id": nid, "mode": mode, "run": k, **({"arm": arm, "fact_controls": arms[arm]} if args.dev_arms else {}),
                       **({"fact_controls": sent_controls} if sent_controls is not None and not args.dev_arms else {}),
                       "ok": bool(r.ok and len(draft) > 1000 and not errors),
                       "status": r.status_code, "wall_s": round(time.perf_counter() - t0, 1),
                       "latency_ms": record.get("latency_ms"), "draft_chars": len(draft),
                       "draft_sha256": sha256_file(out_dir / f"{stem}.md"), "n_precedents": len(cases),
                       "llm_calls": record["totals"]["llm_calls"], "prompt_tokens": record["totals"]["prompt_token_count"],
                       "output_tokens": record["totals"]["candidates_token_count"],
                       "thoughts_tokens": record["totals"]["thoughts_token_count"], "cost_usd": round(record_cost(record), 4),
                       "blocked_retries": roles.count("agent_blocked_retry"), "legal_search": "agent:LegalSearcher" in roles,
                       "fact_guard": record.get("fact_guard") or (fact[-1] if fact else None),
                       "credentials": record.get("credentials_used"), "agent_order": record.get("agent_order"),
                       "errors": errors or record.get("errors") or None, "request_id": record["request_id"],
                       "server_dir": run_dir.name, "pilot": args.pilot}
                with open(rows_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  {stem}: ok={row['ok']} {row['wall_s']} s ${row['cost_usd']} chars={len(draft)} "
                      f"retries={row['blocked_retries']} fact_guard={row['fact_guard']}", flush=True)
        run.add_output(precedents_path, "A4: precedents of each notice")
        rows = [r for r in map(json.loads, rows_path.read_text(encoding="utf-8").splitlines())]
        mine = [r for r in rows if r["server_dir"] == run_dir.name]
        summary = {"drafts": len(mine), "ok": sum(r["ok"] for r in mine), "cost_usd": round(sum(r["cost_usd"] for r in mine), 2),
                   "wall_h": round(sum(r["wall_s"] for r in mine) / 3600, 2)}
        if args.pilot:
            per_mode = {m: {"mean_cost_usd": round(float(np.mean([r["cost_usd"] for r in mine if r["mode"] == m])), 4),
                            "mean_wall_s": round(float(np.mean([r["wall_s"] for r in mine if r["mode"] == m])), 1)}
                        for m in spec["modes"] if any(r["mode"] == m for r in mine)}
            n_total = len(frozen) * spec["runs"]
            projected = sum(v["mean_cost_usd"] for v in per_mode.values()) * n_total
            threshold = spec["budget"]["plan_line_usd"] * spec["budget"]["stop_ratio"]
            summary |= {"per_mode": per_mode, "projected_drafting_usd": round(projected, 2),
                        "projected_drafting_h": round(sum(v["mean_wall_s"] for v in per_mode.values()) * n_total / 3600, 1),
                        "stop_threshold_usd": threshold, "within_threshold": projected <= threshold}
            pilot_path = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_drafting_pilot.json")
            pilot_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
            run.add_output(pilot_path, "A4: pilot cost, time and projection")
        run.data["summary"] = summary
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
