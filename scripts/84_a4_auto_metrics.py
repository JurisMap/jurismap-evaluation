"""Phase 8, A4 (task 4): the automatic metrics of the drafts that need no model and no outside source.

Per draft (data/v2/a4/drafts.jsonl and the draft files): template section coverage; precedents cited
(supplied, other corpus decisions, the source decision, process numbers not in the corpus, ruling
numbers not resolved); statute validity of the cited articles against the official texts
(results/a4_statutes*.json, scripts/85); the fact guard's replacements and flags by kind; recitation
blocks retried and drafts lost; tokens, cost and latency. The invented facts the guard missed are
measured separately (a comparison with the notice).

Outputs (never overwritten):
  results/a4_drafting.csv               one row per draft
  results/a4_drafting_by_mode.csv       means per mode (and shares where the metric is a yes/no)
  results/a4_citations_unresolved.csv   every citation not found in the corpus, for a manual check
  results/a4_statutes_review.csv        every cited article not found in its act, of an unknown act or with no act
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from jm_eval.drafting import metrics as M
from jm_eval.drafting import notices as N
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C

SCRIPT = "84_a4_auto_metrics"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = N.settings()
    mcfg = cfg["metrics"]
    a4 = N.work_dir()                                      # JM_A4_VERSION selects v2.6 (default) or v2.7
    results = resolve(load_config("paths")["results"])
    with RunManifest(SCRIPT, a4, params={"metrics": mcfg}) as run:
        frozen = {r["notice_id"]: r for r in N.load_frozen(cfg)}
        precedents = json.loads((a4 / "precedents.json").read_text(encoding="utf-8"))
        drafts = [json.loads(line) for line in (a4 / "drafts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        latest = {}
        for d in drafts:                                  # each cell's first attempt: a failure stays a failure
            latest.setdefault((d["notice_id"], d["mode"], d["run"]), d)
        corpus = pd.read_parquet(resolve(cfg["corpus"]), columns=["acordao_hash", "acordao_acordao"])
        index = M.CorpusIndex(corpus, C.load_labels(), resolve(load_config("paths")["data"]["v2"]) / "reextract_20260927" / "sources")
        statutes_path = latest_artefact_path(results / "a4_statutes.json")
        run.add_input(statutes_path)
        statutes = json.loads(statutes_path.read_text(encoding="utf-8"))["acts"]
        rows, unresolved, review = [], [], []
        for (nid, mode, k), d in sorted(latest.items()):
            if not d["ok"]:                               # a lost draft: counted, no text metric
                rows.append({"notice_id": nid, "mode": mode, "run": k, "outcome": frozen[nid]["outcome"], "ok": False,
                             "lost": True, "blocked_retries": d["blocked_retries"], "cost_usd": d["cost_usd"],
                             "latency_s": round((d.get("latency_ms") or 0) / 1000, 1), "pilot": d.get("pilot", False),
                             "error": "; ".join(map(str, d.get("errors") or []))[:300]})
                continue
            text = (a4 / "drafts" / f"{nid}_{mode}_r{k}.md").read_text(encoding="utf-8")
            supplied = [] if mode == "no_precedents" else precedents[nid]["precedents"]
            cov = M.section_coverage(text, mcfg["template_sections"], mcfg["heading_ratio"])
            cit = M.classify_citations(text, index, precedents[nid]["precedents"], frozen[nid]["acordao_hash"], frozen[nid]["notice"])
            law = M.classify_statutes(text, statutes)
            review += [{"notice_id": nid, "mode": mode, "run": k, **x} for x in law["statutes"] if x["status"] != "found"]
            fg = d.get("fact_guard") or {}
            kinds = pd.Series([f["kind"] for f in fg.get("findings", []) if f.get("action") == "flagged"], dtype=object)
            rows.append({
                "notice_id": nid, "mode": mode, "run": k, "outcome": frozen[nid]["outcome"], "ok": d["ok"],
                "draft_chars": d["draft_chars"], "template_coverage": cov["coverage"],
                **{f"has_{s.split('–')[0].strip()}": v for s, v in cov["sections_found"].items()},
                "n_citations": cit["n_citations"], "cited_supplied": cit["cited_supplied"],
                "supplied_used_share": cit["supplied_used_share"] if supplied else None,
                "cited_corpus_not_supplied": cit["cited_corpus_not_supplied"], "cited_source": cit["cited_source"],
                "process_not_in_corpus": cit["process_not_in_corpus"], "ruling_unresolved": cit["ruling_unresolved"],
                "ruling_process_mismatch": cit["ruling_process_mismatch"],
                **{k: v for k, v in law.items() if k != "statutes"},
                "fact_guard_replaced": fg.get("replaced"), "fact_guard_flagged": fg.get("flagged"),
                "flagged_dates": int((kinds == "date").sum()), "flagged_amounts": int((kinds == "amount").sum()),
                "flagged_legal_acts": int((kinds == "legal_act").sum()),
                "blocked_retries": d["blocked_retries"], "lost": not d["ok"],
                "prompt_tokens": d["prompt_tokens"], "output_tokens": d["output_tokens"], "thoughts_tokens": d["thoughts_tokens"],
                "cost_usd": d["cost_usd"], "latency_s": round((d.get("latency_ms") or 0) / 1000, 1),
                "pilot": d.get("pilot", False)})
            unresolved += [{"notice_id": nid, "mode": mode, "run": k, **{x: c[x] for x in ("kind", "key", "text")}}
                           for c in cit["citations"] if c["status"] == "not_found"]
            unresolved += [{"notice_id": nid, "mode": mode, "run": k, "kind": "mismatch", "key": x["process"],
                            "text": f"{x['ruling']} with {x['process']}"} for x in cit["mismatches"]]
        table = pd.DataFrame(rows)
        path = new_artefact_path(N.result_path("a4_drafting.csv"))
        table.to_csv(path, index=False)
        run.add_output(path, "A4: automatic metrics, one row per draft")
        numeric = [c for c in table.columns if c not in ("notice_id", "mode", "outcome", "run")]
        by_mode = table.groupby("mode")[numeric].mean(numeric_only=True).round(4)
        by_mode.insert(0, "drafts", table.groupby("mode").size())
        mode_path = new_artefact_path(N.result_path("a4_drafting_by_mode.csv"))
        by_mode.to_csv(mode_path)
        run.add_output(mode_path, "A4: automatic metrics, means per mode")
        unresolved_path = new_artefact_path(N.result_path("a4_citations_unresolved.csv"))
        pd.DataFrame(unresolved, columns=["notice_id", "mode", "run", "kind", "key", "text"]).to_csv(unresolved_path, index=False)
        run.add_output(unresolved_path, "A4: citations not found in the corpus, for a manual check")
        review_path = new_artefact_path(N.result_path("a4_statutes_review.csv"))
        pd.DataFrame(review, columns=["notice_id", "mode", "run", "act", "article", "status", "text"]).to_csv(review_path, index=False)
        run.add_output(review_path, "A4: cited articles not found in their act, of unknown acts or with no act, for a manual check")
        run.data["summary"] = {"drafts": len(table), "ok": int(table["ok"].sum()), "unresolved_citations": len(unresolved),
                               "cited_source": int(table["cited_source"].sum())}
        print(by_mode[["drafts", "template_coverage", "cited_supplied", "cited_corpus_not_supplied", "process_not_in_corpus",
                       "ruling_unresolved", "ruling_process_mismatch", "statute_found", "statute_beyond", "statute_in_range", "statute_unknown_act",
                       "fact_guard_flagged", "blocked_retries", "cost_usd", "latency_s"]].to_string())
        print(json.dumps(run.data["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
