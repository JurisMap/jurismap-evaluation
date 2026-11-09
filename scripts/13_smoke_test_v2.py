"""Phase 1 step 6: end-to-end smoke test of JurisMap v2 on a local server.

filters -> map -> chat (full_text and hierarchical_summary) -> upload template and notice ->
uploaded case on the map -> the 10 pre-selected similar cases (v2.4) -> defense generation in every
ablation mode with those cases, plus one run through the SupervisorAgent. Inputs: the fictitious notice and template in frontend_old/ (not real cases).

Outputs (data/v2/smoke_<run>/, git-ignored): server log, evaluation logs, drafts.
         results/v2_smoke_test.json: status, latency, tokens and agent order per step (no draft text).
The script exits non-zero if any step fails.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.server import LocalServer, sse_messages

SCRIPT = "13_smoke_test_v2"
MODES = ["full", "single_prompt", "no_legal_search", "no_precedents", "no_template"]
NOTICE = "Ficticious_Infraction_Notice_Drone_Association.docx"
TEMPLATE = "Ficticious_Defense_Template.docx"
# Default filter payload of the React frontend (Home.js); the backend expects all keys
FRONTEND_FILTERS = {"recorrentes": [], "relatores": [], "resultados": [], "advogados": [], "period_start": None,
                    "period_end": None, "titulo_keyword": "", "ementa_keyword": "",
                    "ementa_search_mode": "Correspondência exata"}
CHAT_QUESTION = "Quais foram os principais fundamentos das decisões sobre operação irregular de seguros por associações?"


def log_summary(record: dict) -> dict:
    return {
        "status_code": record.get("status_code"),
        "latency_ms": record.get("latency_ms"),
        "llm_calls": record["totals"]["llm_calls"],
        "prompt_tokens": record["totals"]["prompt_token_count"],
        "output_tokens": record["totals"]["candidates_token_count"],
        "thoughts_tokens": record["totals"]["thoughts_token_count"],
        "models": sorted({c["model"] for c in record["llm_calls"] if c.get("model")}),
        "agent_order": record.get("agent_order"),
        "errors": record.get("errors"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="*", default=MODES)
    args = ap.parse_args()

    paths = load_config("paths")
    run_dir = resolve(paths["data"]["v2"]) / f"smoke_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    frontend_v1 = resolve(paths["repos"]["frontend_v1"])
    steps: dict[str, dict] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        steps.setdefault(name, {})["ok"] = ok
        if not ok:
            failures.append(f"{name}: {detail}")

    with RunManifest(SCRIPT, run_dir, params={"modes": args.modes, "notice": NOTICE, "template": TEMPLATE}) as run, \
            LocalServer(run_dir) as server:
        run.data["params"]["server_env"] = server.env_overrides
        s = server.session

        # 1. Filters (reads crsnsp_eval.scraper_output_v2)
        r = s.post(server.url("/API_get_filters"), json={})
        filters = r.json() if r.ok else {}
        steps["filters"] = {"status": r.status_code, **{k: len(v) for k, v in filters.items() if isinstance(v, list)},
                            "most_recent_date": filters.get("most_recent_date")}
        check("filters", r.ok and len(filters.get("relatores", [])) > 0, r.text[:200])

        # 2. Map with a date filter
        r = s.post(server.url("/API_plot_umap_embeddings"),
                   json={**FRONTEND_FILTERS, "period_start": "2023-01-01", "period_end": "2026-12-31"})
        plot = r.json() if r.ok else {}
        ids = plot.get("filtered_data_ids", [])
        steps["map"] = {"status": r.status_code, "points": len(ids),
                        "x_range": [min(plot["embeddings_2D"]["x"]), max(plot["embeddings_2D"]["x"])] if ids else None}
        check("map", r.ok and len(ids) > 100, r.text[:200])
        r_all = s.post(server.url("/API_plot_umap_embeddings"), json=FRONTEND_FILTERS)      # every decision, for step 4
        ids_all = r_all.json().get("filtered_data_ids", []) if r_all.ok else []

        # 3. Chat in both context modes
        retrieved: list[str] = []
        for name, body in [("chat_full_text", {"top_k": 10}),
                           ("chat_hierarchical", {"top_k": 30, "context_mode": "hierarchical_summary"})]:
            r = s.post(server.url("/API_chatbot"),
                       json={"user_message": CHAT_QUESTION, "filtered_data_ids": ids, **body})
            answer = r.json().get("assistant_message", "") if r.ok else ""
            record = server.eval_log(r.headers["X-Eval-Request-Id"])
            steps[name] = {"status": r.status_code, "answer_chars": len(answer),
                           "top_k_effective": record["flags"].get("top_k_effective"), **log_summary(record)}
            (run_dir / f"{name}_answer.md").write_text(answer, encoding="utf-8")
            check(name, r.ok and len(answer) > 100 and record["flags"].get("top_k_effective") == body["top_k"],
                  r.text[:200])
            if name == "chat_full_text":
                retrieved = record["flags"]["retrieved_hashes"]

        # 4. Upload template and notice; the notice is projected onto the map
        with open(frontend_v1 / TEMPLATE, "rb") as f:
            r = s.post(server.url("/API_upload_defense_template"), files={"defense_template": (TEMPLATE, f)})
        steps["upload_template"] = {"status": r.status_code}
        check("upload_template", r.ok, r.text[:200])
        with open(frontend_v1 / NOTICE, "rb") as f:
            r = s.post(server.url("/API_upload_case"), files={"case_file": (NOTICE, f)})
        record = server.eval_log(r.headers["X-Eval-Request-Id"])
        steps["upload_case"] = {"status": r.status_code, **log_summary(record)}
        check("upload_case", r.ok, r.text[:200])
        r = s.get(server.url("/API_get_uploaded_case_data"))
        uploaded = r.json() if r.ok else {}
        coords = uploaded.get("embeddings_2D")
        steps["uploaded_case_on_map"] = {"status": r.status_code, "coordinates": coords}
        check("uploaded_case_on_map", r.ok and coords is not None, r.text[:200])
        # From v2.4 the upload pre-selects the 10 decisions most similar to the case; the frontend
        # offers them as the context cases, so the defenses below use them as the app does.
        similar = uploaded.get("similar_cases") or {}
        preselected, similarity = similar.get("filtered_data_ids") or [], similar.get("similarity") or []
        steps["preselected_cases"] = {"cases": len(preselected), "similarity": similarity,
                                      "also_retrieved_by_the_chat_question": len(set(preselected) & set(retrieved))}
        check("preselected_cases", len(preselected) == 10 and len(set(preselected)) == 10
              and similarity == sorted(similarity, reverse=True) and set(preselected) <= set(ids_all),
              f"{len(preselected)} cases, similarity {similarity}")
        context_cases = preselected or retrieved

        # 5. Defense generation: every ablation mode (no supervisor), then full mode via the supervisor
        runs = [(mode, {"ablation_mode": mode, "skip_supervisor": True, "case_hashes": context_cases}) for mode in args.modes]
        runs.append(("full_via_supervisor", {"ablation_mode": "full", "phase": "supervisor", "selected_cases_ids": context_cases}))
        for name, body in runs:
            t0 = time.perf_counter()
            r = s.post(server.url("/API_generate_defense"),
                       json={"human_message": "Por favor, gere uma defesa jurídica para o caso apresentado.", **body},
                       stream=True, timeout=1800)
            messages = list(sse_messages(r)) if r.ok else []
            r.close()
            record = server.eval_log(r.headers["X-Eval-Request-Id"], timeout=60)
            drafts = [m["content"] for m in messages
                      if m.get("agent_id") in ("MergerAgent", "SinglePrompt") and m.get("content")]
            draft = drafts[-1] if drafts else ""
            errors = [m for m in messages if m.get("status") == "error" or m.get("error")]
            supervisor = [m for m in messages if m.get("agent_id") == "SupervisorAgent" and m.get("is_final")]
            fact_checks = [m.get("fact_check") for m in messages if m.get("agent_id") == "FactGuard"]
            (run_dir / "drafts").mkdir(exist_ok=True)
            (run_dir / "drafts" / f"{name}.md").write_text(draft, encoding="utf-8")
            (run_dir / "drafts" / f"{name}.messages.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=1), encoding="utf-8")
            steps[f"defense_{name}"] = {
                "status": r.status_code, "wall_s": round(time.perf_counter() - t0, 1), "draft_chars": len(draft),
                "completed": any(m.get("status") == "completed" for m in messages),
                "supervisor_status": supervisor[-1].get("status") if supervisor else None,
                "fact_check": fact_checks[-1] if fact_checks else None,
                "stream_errors": [m.get("content") or m.get("error") for m in errors], **log_summary(record)}
            check(f"defense_{name}", r.ok and len(draft) > 1000 and not errors,
                  f"status {r.status_code}, draft {len(draft)} chars, errors {errors[:1]}")

        run.data["summary"] = {"steps": steps, "failures": failures}

    summary_path = new_artefact_path(resolve(paths["results"]) / "v2_smoke_test.json")
    summary_path.write_text(json.dumps({"run_dir": run_dir.name, "steps": steps, "failures": failures},
                                       indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"steps": steps, "failures": failures}, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
