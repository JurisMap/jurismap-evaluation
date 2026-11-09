"""Phase 6, A3 (tasks 5.5, 7, 8, 10): results of the chatbot evaluation, from the working tables.

Reads the newest answers, deterministic scores, claims and coverage judgements, and writes the
tables for the paper. It calls one model, Mistral Medium 3, only to
draft the English translation of the six example answers (marked as a draft until the author reviews it).

Outputs (never overwritten):
  results/a3_chatbot.csv            every figure: section, group, metric, rater, n, value, interval
  results/a3_claims.parquet         claims, evidence and every label
  results/a3_judge_agreement.csv    agreement between the judges and with the NLI model
  results/a3_examples.md            3 good and 3 bad answers, Portuguese and English
  results/costs_phase6.csv          calls, tokens and cost per step and model; largest prompt per step
`--dev` writes the same files for the development set under data/v2/a3_dev/report/ (never reported).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jm_eval.cache import LLMCache
from jm_eval.chatbot import judging as J
from jm_eval.chatbot import metrics as M
from jm_eval.chatbot import store
from jm_eval.io import latest_artefact_path, load_config, new_artefact_path
from jm_eval.manifest import RunManifest
from jm_eval.stats import paired_randomisation_test, wilson_interval

SCRIPT = "68_a3_report"
PLAN_BUDGET_USD = 13.0
INTERNAL_STEPS = {"pairs"}          # an internal check that is not reported (DECISIONS.md, 2026-10-11)
TRANSLATION_SYSTEM = "You translate Brazilian Portuguese legal text into English. Answer with the translation only."
TRANSLATION_PROMPT = """Translate the Portuguese text below into English. It is about decisions of a Brazilian administrative appeals council for insurance (CRSNSP). Give a free, faithful translation: keep process numbers, names and amounts as they are, keep the paragraphs, and add nothing. Return the translation only.

TEXT:
{text}"""


def translate(asker: J.Asker, text: str, item: str) -> str:
    """Draft English translation (to be reviewed by the author); "(no translation)" if the model gives none."""
    answer = asker.ask(TRANSLATION_PROMPT.format(text=text), lambda t: (t.strip(), None) if t and t.strip() else (None, "empty"),
                       item=item, system=TRANSLATION_SYSTEM, as_json=False)
    return answer or "(no translation)"


def row(section, group, metric, rater, n, value, low=None, high=None, note=""):
    return {"section": section, "group": group, "metric": metric, "rater": rater, "n": n,
            "value": None if value is None or value != value else round(float(value), 4),
            "ci_low": None if low is None or low != low else round(float(low), 4),
            "ci_high": None if high is None or high != high else round(float(high), 4), "note": note}


def cluster_bootstrap(numerator: pd.Series, denominator: pd.Series, clusters: pd.Series, resamples: int, seed: int):
    """Percentile interval of sum(numerator) / sum(denominator) when whole questions are resampled."""
    table = pd.DataFrame({"n": numerator.to_numpy(float), "d": denominator.to_numpy(float), "c": clusters.to_numpy()})
    sums = table.groupby("c")[["n", "d"]].sum().to_numpy()
    if len(sums) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(resamples, len(sums)))
    totals = sums[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = totals[:, 0] / totals[:, 1]
    low, high = np.nanquantile(ratios, [0.025, 0.975])
    return float(low), float(high)


def deterministic_rows(det: pd.DataFrame) -> list[dict]:
    rows = []
    templated, unanswerable = det[det["type"] == "templated"], det[det["type"] == "unanswerable"]
    if len(templated):
        k, n = int(templated["correct"].sum()), len(templated)
        rows.append(row("templated", "all", "accuracy", "code", n, k / n, *wilson_interval(k, n), note=f"{k} of {n} correct"))
        rows.append(row("templated", "all", "mean_score", "code", n, templated["score"].mean(), note="share of the gold items stated"))
        ambiguous = int(templated["details"].str.contains('"ambiguous": true').sum())
        rows.append(row("templated", "all", "ambiguous_answers", "code", n, ambiguous, note="gold value stated next to another value; scored 0"))
        for subtype, sub in templated.groupby("subtype"):
            rows.append(row("templated", subtype, "accuracy", "code", len(sub), sub["correct"].mean(),
                            note=f"{int(sub['correct'].sum())} of {len(sub)} correct"))
    if len(unanswerable):
        scored = unanswerable[unanswerable["correct"].notna()]
        k, n = int(scored["correct"].sum()), len(unanswerable)
        rows.append(row("unanswerable", "all", "abstention_rate", "code", n, k / n, *wilson_interval(k, n),
                        note=f"{k} of {n} abstained; {int((unanswerable['scored_by'] == 'code').sum())} classified by the patterns"))
        for subtype, sub in unanswerable.groupby("subtype"):
            rows.append(row("unanswerable", subtype, "abstention_rate", "code", len(sub), (sub["correct"] == True).mean()))  # noqa: E712
    citing = det[det["cited"] > 0]
    n = len(det)
    rows.append(row("citations", "all", "answers_citing_a_process_number", "code", n, len(citing) / n if n else None))
    rows.append(row("citations", "all", "process_numbers_cited", "code", n, det["cited"].sum()))
    for kind in ("supplied", "in_question", "in_context", "unknown"):
        rows.append(row("citations", "all", f"process_numbers_{kind}", "code", int(det["cited"].sum()), det[f"cited_{kind}"].sum()))
    ok = int(det["citations_all_supplied"].sum())
    rows.append(row("citations", "all", "answers_citing_only_supplied_decisions", "code", n, ok / n if n else None,
                    *wilson_interval(ok, n), note="numbers named in the question count as supplied"))
    bad = int((~det["citations_none_unknown"]).sum())
    rows.append(row("citations", "all", "answers_with_an_unknown_process_number", "code", n, bad / n if n else None,
                    *wilson_interval(bad, n), note="a number found neither in the set nor in its texts"))
    return rows


def faithfulness_rows(claims: pd.DataFrame, raters: list[str], separate: list[str], resamples: int, seed: int) -> tuple[list[dict], pd.DataFrame]:
    control = claims.loc[claims["full_text_label"].notna(), ["claim_id", "full_text_label"]]
    rate = M.miss_rate(control)
    rows = [row("miss_control", "all", "claims_not_found_by_a_judge", "judge_a or judge_b",
                len(claims), int((claims[[r for r in raters if r.startswith("judge")]] == "not_found").any(axis=1).sum())),
            row("miss_control", "all", "miss_rate", "judge_a (full text)", rate["judged"], rate["miss_rate"],
                *wilson_interval(rate["flipped_to_supported"], rate["judged"]),
                note=f"{rate['flipped_to_supported']} of {rate['judged']} re-judged claims are supported by the full decision; "
                     f"{rate['flipped_to_contradicted']} contradicted")]
    per_answer = []
    for rater in raters + separate:
        raw = M.faithfulness(claims, rater)
        fixed = M.corrected(claims, rater, control, rate["miss_rate"])
        judged = claims[claims[rater].isin(M.LABELS)]
        by_answer = judged.groupby("answer_id").agg(
            question_id=("question_id", "first"), claims=(rater, "size"),
            supported=(rater, lambda s: (s == "supported").sum()), contradicted=(rater, lambda s: (s == "contradicted").sum()),
            bad=(rater, lambda s: int((s != "supported").any())))
        by_answer["rater"] = rater
        per_answer.append(by_answer.reset_index())
        ones = pd.Series(1.0, index=by_answer.index)
        note = "same family as the system: reported separately" if rater in separate else ""
        ci = cluster_bootstrap(by_answer["supported"], by_answer["claims"], by_answer["question_id"], resamples, seed)
        rows.append(row("faithfulness", "all", "faithfulness", rater, raw["claims_judged"], raw["faithfulness"], *ci, note=note))
        ci = cluster_bootstrap(by_answer["contradicted"], by_answer["claims"], by_answer["question_id"], resamples, seed)
        rows.append(row("faithfulness", "all", "contradiction_rate", rater, raw["claims_judged"], raw["contradiction_rate"], *ci, note=note))
        rows.append(row("faithfulness", "all", "not_found_rate", rater, raw["claims_judged"], raw["not_found_rate"], note=note))
        ci = cluster_bootstrap(by_answer["bad"], ones, by_answer["question_id"], resamples, seed)
        rows.append(row("faithfulness", "all", "hallucination_rate", rater, raw["answers"], raw["hallucination_rate"], *ci,
                        note=("answers with a contradicted or not-found claim. " + note).strip()))
        rows.append(row("faithfulness", "all", "answers_with_a_contradicted_claim", rater, raw["answers"],
                        raw["answers_with_a_contradicted_claim"] / raw["answers"] if raw["answers"] else None, note=note))
        for metric, value in fixed.items():
            rows.append(row("faithfulness", "all", metric, rater, raw["claims_judged"], value,
                            note=("after the retrieval-miss control. " + note).strip()))
        for set_id, part in judged.groupby("set_id"):            # per case set: the sets differ much in size
            by_set = M.corrected(part, rater, control, rate["miss_rate"])
            rows.append(row("faithfulness", set_id, "faithfulness", rater, len(part), (part[rater] == "supported").mean(), note=note))
            rows.append(row("faithfulness", set_id, "faithfulness_corrected", rater, len(part), by_set["faithfulness_corrected"],
                            note=("after the retrieval-miss control (overall miss rate). " + note).strip()))
            rows.append(row("faithfulness", set_id, "contradiction_rate", rater, len(part), (part[rater] == "contradicted").mean(), note=note))
    return rows, pd.concat(per_answer, ignore_index=True)


def completeness_rows(cov: pd.DataFrame, judges: list[str], separate: list[str], resamples: int, seed: int) -> tuple[list[dict], pd.DataFrame]:
    cov = cov.dropna(subset=["coverage"]).assign(value=lambda d: d["coverage"].map(M.COVERAGE_VALUE),
                                                 strict=lambda d: (d["coverage"] == "covered").astype(float))
    per_answer = cov.groupby(["judge", "answer_id"]).agg(question_id=("question_id", "first"), subtype=("subtype", "first"),
                                                         points=("value", "size"), covered=("value", "sum"),
                                                         strict=("strict", "sum")).reset_index()
    rows = []
    for judge in judges + separate:
        sub = per_answer[per_answer["judge"] == judge]
        note = "same family as the system: reported separately" if judge in separate else ""
        ci = cluster_bootstrap(sub["covered"], sub["points"], sub["question_id"], resamples, seed)
        rows.append(row("completeness", "all", "coverage", judge, len(sub), sub["covered"].sum() / sub["points"].sum(), *ci,
                        note=("covered = 1, partly = 0.5, missing = 0. " + note).strip()))
        rows.append(row("completeness", "all", "coverage_strict", judge, len(sub), sub["strict"].sum() / sub["points"].sum(), note=note))
        for subtype, part in sub.groupby("subtype"):
            rows.append(row("completeness", subtype, "coverage", judge, len(part), part["covered"].sum() / part["points"].sum(), note=note))
    per_answer["coverage"] = per_answer["covered"] / per_answer["points"]
    return rows, per_answer


def variability_rows(faith: pd.DataFrame, cov: pd.DataFrame, judges: list[str]) -> list[dict]:
    """Spread across the three runs of each open question."""
    rows = []
    majority = faith[faith["rater"] == "majority"].assign(value=lambda d: d["supported"] / d["claims"])
    coverage = cov[cov["judge"].isin(judges)].groupby(["question_id", "answer_id"])["coverage"].mean().reset_index(name="value")
    for name, table in (("faithfulness (majority)", majority), ("coverage (mean of the judges)", coverage)):
        spread = table.groupby("question_id")["value"].agg(["std", "min", "max", "count"])
        spread = spread[spread["count"] > 1]
        if len(spread):
            rows.append(row("variability", "runs of a question", f"{name}: mean SD across runs", "", len(spread), spread["std"].mean()))
            rows.append(row("variability", "runs of a question", f"{name}: largest range across runs", "", len(spread),
                            (spread["max"] - spread["min"]).max()))
    return rows


def system_rows(answers: pd.DataFrame) -> list[dict]:
    rows = []
    for group, sub in [("all", answers), *answers.groupby("type")]:
        rows += [row("system", group, "latency_s_median", "", len(sub), sub["latency_ms"].median() / 1000),
                 row("system", group, "latency_s_p95", "", len(sub), sub["latency_ms"].quantile(0.95) / 1000),
                 row("system", group, "prompt_tokens_median", "", len(sub), sub["prompt_tokens"].median()),
                 row("system", group, "output_tokens_median", "", len(sub), sub["output_tokens"].median(), note="thinking included"),
                 row("system", group, "cost_usd_per_answer", "", len(sub), sub["cost_usd"].mean())]
    served = answers["credentials_used"].map(json.loads)
    calls = sum(sum(c.values()) for c in served)
    rows.append(row("system", "all", "model_calls_served_by_the_primary_key", "", calls,
                    sum(c.get("primary", 0) for c in served) / calls if calls else None))
    rows.append(row("system", "all", "answers_needing_a_retry", "", len(answers), int((answers["chat_attempts"] > 1).sum())))
    return rows


def agreement_rows(claims: pd.DataFrame, cov: pd.DataFrame, per_answer_cov: pd.DataFrame, verifiers: list[str],
                   separate: list[str], results: Path, dev: bool) -> list[dict]:
    rows = []

    def add(item, a, b, stats, note=""):
        rows.append({"item": item, "rater_a": a, "rater_b": b, "n": stats.get("n"),
                     "agreement": None if stats.get("agreement") is None else round(stats["agreement"], 4),
                     "krippendorff_alpha": None if stats.get("alpha") is None or stats["alpha"] != stats["alpha"] else round(stats["alpha"], 4),
                     "statistic": stats.get("statistic"), "value": stats.get("value"), "note": note})

    raters = ["nli", *verifiers, *separate]
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            note = "same-family judge: reported separately" if a in separate or b in separate else ""
            add("claim label (supported / contradicted / not found)", a, b, M.nominal_alpha(claims[a], claims[b], M.LABELS), note)
    wide = cov.pivot_table(index=["answer_id", "point"], columns="judge", values="coverage", aggfunc="first")
    answers = per_answer_cov.pivot(index="answer_id", columns="judge", values="coverage")
    judges = [c for c in wide.columns]
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            note = "same-family judge: reported separately" if a in separate or b in separate else ""
            numeric = wide[[a, b]].apply(lambda s: s.map(M.COVERAGE_VALUE))
            stats = {**M.interval_alpha(numeric[a], numeric[b], "ordinal"),
                     "agreement": float((wide[a] == wide[b])[wide[[a, b]].notna().all(axis=1)].mean())}
            add("coverage of a reference point (covered / partly / missing), ordinal", a, b, stats, note)
            add("coverage of an answer, interval", a, b, M.interval_alpha(answers[a], answers[b]), note)
    # Mistral wrote the reference material and the claims and also judges: does it score differently from judge A?
    if {"judge_a", "judge_b"} <= set(answers.columns):
        both = answers[["judge_a", "judge_b"]].dropna()
        diff, p = paired_randomisation_test(both["judge_b"].to_numpy(), both["judge_a"].to_numpy())
        add("coverage of an answer: Mistral minus GPT-5.4", "judge_b", "judge_a",
            {"n": len(both), "statistic": "mean difference (paired randomisation p)", "value": f"{diff:.4f} (p = {p:.4f})"},
            "Mistral wrote the reference points")
        judged = claims[claims["judge_a"].isin(M.LABELS) & claims["judge_b"].isin(M.LABELS)]
        add("claims labelled supported: Mistral minus GPT-5.4", "judge_b", "judge_a",
            {"n": len(judged), "statistic": "difference of shares",
             "value": f"{(judged['judge_b'] == 'supported').mean() - (judged['judge_a'] == 'supported').mean():.4f}"},
            "Mistral split the answers into claims")
    return rows


def examples(answers: pd.DataFrame, questions: dict, faith: pd.DataFrame, per_answer_cov: pd.DataFrame, claims: pd.DataFrame,
             judges: list[str], asker: J.Asker, n: int = 3) -> str:
    majority = faith[faith["rater"] == "majority"].assign(faithfulness=lambda d: d["supported"] / d["claims"]).set_index("answer_id")
    coverage = per_answer_cov[per_answer_cov["judge"].isin(judges)].groupby("answer_id")["coverage"].mean()
    table = answers[answers["type"] == "open"].set_index("answer_id")
    table = table.assign(faithfulness=majority["faithfulness"], coverage=coverage).dropna(subset=["faithfulness", "coverage"])
    table["quality"] = table["faithfulness"] + table["coverage"]
    ranked = table.sort_values(["quality", "answer_id"] if "answer_id" in table else ["quality"], ascending=False)

    def pick(frame: pd.DataFrame) -> list[str]:
        chosen, seen = [], set()
        for answer_id, r in frame.iterrows():
            if r["question_id"] not in seen:
                chosen.append(answer_id)
                seen.add(r["question_id"])
            if len(chosen) == n:
                break
        return chosen

    good = pick(ranked)
    bad = pick(ranked.iloc[::-1].drop(index=good, errors="ignore"))
    lines = ["# A3 chatbot evaluation: example answers", "",
             "Three answers with the highest and three with the lowest sum of faithfulness (majority of the NLI model, "
             "GPT-5.4 and Mistral Medium 3) and coverage of the reference points (mean of GPT-5.4 and Mistral), one per "
             "question. The answers are quoted in Portuguese as generated. **The English translations were drafted by "
             "Mistral Medium 3 and have not yet been reviewed by the author.**", ""]
    for title, ids in (("Good answers", good), ("Bad answers", bad)):
        lines += [f"## {title}", ""]
        for answer_id in ids:
            r, q = table.loc[answer_id], questions[table.at[answer_id, "question_id"]]
            translation = translate(asker, r["answer"], answer_id)
            question_en = translate(asker, q["question"], f"{answer_id}|question")
            mine = claims[claims["answer_id"] == answer_id]
            flagged = mine[mine["majority"].isin(["contradicted", "not_found"]) | mine["majority"].isna()]
            lines += [f"### {answer_id} (case set {q['set_id']}, {q['subtype']})", "",
                      f"- Faithfulness (majority): {r['faithfulness']:.2f} ({len(mine)} claims). Coverage: {r['coverage']:.2f}.",
                      f"- **Question (pt):** {q['question']}",
                      f"- **Question (en, draft):** {question_en}", "",
                      "**Answer (pt):**", "", "> " + r["answer"].strip().replace("\n", "\n> "), "",
                      "**Answer (en, draft translation by Mistral Medium 3, not reviewed):**", "",
                      "> " + translation.replace("\n", "\n> "), ""]
            if len(flagged):
                lines += ["Claims without a majority for \"supported\" (labels: NLI / GPT-5.4 / Mistral; full-text control where run):", ""]
                for c in flagged.to_dict("records"):
                    control = f"; full text: {c['full_text_label']}" if isinstance(c.get("full_text_label"), str) else ""
                    lines.append(f"- {c['claim']} ({c['nli']} / {c['judge_a']} / {c['judge_b']}{control})")
                lines.append("")
    return "\n".join(lines)


def cost_rows() -> pd.DataFrame:
    """Calls, tokens and cost per step and model.

    Judge and material calls come from the response cache, which holds every call that was paid
    (also those of a run that failed), each once. A call belongs to the evaluated set if a run on
    the evaluated set logged its cache key, otherwise to the development work. The chatbot's own
    calls come from the answers tables.
    """
    evaluated_keys, problems = set(), {}
    work = store.root_dir()
    manifests = (list(work.glob("*.manifest.json")) + list(store.results_dir().glob("61_a3_questions*.manifest.json"))
                 + list(store.results_dir().glob("68_a3_report*.manifest.json"))
                 + list(store.root_dir(True).rglob("*.manifest.json")))
    for path in manifests:
        for c in json.loads(path.read_text(encoding="utf-8")).get("calls", []):
            if not c.get("dev") and c.get("cache_entry"):
                evaluated_keys.add(c["cache_entry"])
            if c.get("problem") and c.get("cached") is False:
                key = ("development" if c.get("dev") else "evaluated", c["step"], c["model"])
                problems[key] = problems.get(key, 0) + 1
    phase7_keys = set()                  # Phase 7 calls the same judges with the same prompts: its costs are its own
    for path in [*store.a5_dir().rglob("*.manifest.json"), *store.results_dir().glob("72_a5_settings*.manifest.json")]:
        phase7_keys |= {c["cache_entry"] for c in json.loads(path.read_text(encoding="utf-8")).get("calls", [])
                        if c.get("cached") is False and c.get("cache_entry")}
    entries = []
    for path in LLMCache().root.glob("*/*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("params", {}).get("a3") != J.PROMPTS_VERSION or entry["params"].get("step") in INTERNAL_STEPS:
            continue
        if entry["key"] in phase7_keys:
            continue
        meta = entry.get("meta", {})
        entries.append({"set": "evaluated" if entry["key"] in evaluated_keys else "development",
                        "step": entry["params"]["step"], "model": entry["model"],
                        "input_tokens": meta.get("input_tokens") or 0, "output_tokens": meta.get("output_tokens") or 0,
                        "cost_usd": meta.get("cost_usd") or 0.0, "primary": meta.get("credentials") == "primary"})
    rows = []
    tables = [("evaluated", store.root_dir(False) / "answers.parquet"), ("development", store.root_dir(True) / "answers.parquet")]
    tables += [(f"development, strategy {d.name}", d / "answers.parquet")           # Phase 6.1 candidates
               for d in sorted((store.root_dir(True) / "candidates").glob("*")) if d.is_dir()]
    for label, table_path in tables:
        versions = sorted(table_path.parent.glob(f"{table_path.stem}*{table_path.suffix}"))     # every run that was paid
        if versions:
            a = pd.concat([pd.read_parquet(v) for v in versions], ignore_index=True).drop_duplicates("request_id")
            served = a["credentials_used"].map(json.loads)
            total_calls = sum(sum(c.values()) for c in served)
            rows.append({"set": label, "step": "chatbot answers", "model": a["model"].iloc[0], "calls": len(a),
                         "input_tokens": int(a["prompt_tokens"].sum()), "output_tokens": int(a["output_tokens"].sum()),
                         "cost_usd": round(float(a["cost_usd"].sum()), 4), "max_input_tokens": int(a["prompt_tokens"].max()),
                         "unusable_answers": 0,
                         "served_by_primary_key": round(sum(c.get("primary", 0) for c in served) / total_calls, 4) if total_calls else None,
                         "basis": "the system under test, not a judge: the 8K limit does not apply; every version evaluated is "
                                  "counted; the embeddings of the questions are not"})
    table = pd.DataFrame(entries)
    if len(table):
        for (label, step, model), sub in table.groupby(["set", "step", "model"], sort=True):
            rows.append({"set": label, "step": step, "model": model, "calls": len(sub),
                         "input_tokens": int(sub["input_tokens"].sum()), "output_tokens": int(sub["output_tokens"].sum()),
                         "cost_usd": round(float(sub["cost_usd"].sum()), 4), "max_input_tokens": int(sub["input_tokens"].max()),
                         "unusable_answers": problems.get((label, step, model), 0),
                         "served_by_primary_key": round(float(sub["primary"].mean()), 4),
                         "basis": "response cache: every paid call, once; output tokens include reasoning"})
    table = pd.DataFrame(rows).sort_values(["set", "step", "model"], ascending=[False, True, True])
    table.insert(0, "phase", "6")
    judge_rows = table[table["step"] != "chatbot answers"]
    total = {"phase": "6", "set": "all", "step": "TOTAL", "cost_usd": round(float(table["cost_usd"].sum()), 4),
             "max_input_tokens": int(judge_rows["max_input_tokens"].max()) if len(judge_rows) else None,
             "basis": f"plan line for Phase 6: {PLAN_BUDGET_USD:.0f} USD; max_input_tokens here is the largest judge prompt"}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--costs-only", action="store_true", help="write only results/costs_phase6.csv (interim report)")
    args = ap.parse_args()
    if args.costs_only:
        with RunManifest(f"{SCRIPT}_costs", store.results_dir(), params={"costs_only": True}) as run:
            costs = cost_rows()
            costs_path = new_artefact_path(store.results_dir() / "costs_phase6.csv")
            costs.to_csv(costs_path, index=False)
            run.add_output(costs_path, "Phase 6: calls, tokens and cost per step and model (interim)")
            run.data["summary"] = {"total_usd": float(costs.iloc[-1]["cost_usd"])}
        print(costs.drop(columns=["basis", "phase"]).to_string(index=False))
        return 0
    cfg = load_config("a3_chatbot")
    resamples, seed = load_config("a1_methods")["bootstrap_resamples"], load_config("seeds")["bootstrap"]
    verifiers, separate = cfg["faithfulness"]["verifiers"], cfg["faithfulness"]["separate"]
    results = store.results_dir()
    out = results if not args.dev else store.work_dir(True) / "report"
    out.mkdir(parents=True, exist_ok=True)
    with RunManifest(SCRIPT, out, models={"translation": load_config("models")["judges"]["judge_b"]["model"]},
                     params={"dev": args.dev, "bootstrap_resamples": resamples}, seeds={"bootstrap": seed}) as run:
        questions = {q["question_id"]: q for q in store.load_questions(args.dev)}
        answers = store.load_table("answers.parquet", args.dev)
        det = store.load_table("deterministic.parquet", args.dev)
        claims = store.load_table("claims.parquet", args.dev)
        cov = store.load_table("completeness.parquet", args.dev)

        separate = [name for name in separate if name in claims.columns]       # a development run may leave it out
        rows = deterministic_rows(det)
        faith_rows, faith = faithfulness_rows(claims, ["nli", *verifiers, "majority"], separate, resamples, seed)
        cov_rows, per_answer_cov = completeness_rows(cov, cfg["completeness"]["judges"], cfg["completeness"]["separate"], resamples, seed)
        rows += faith_rows + cov_rows + variability_rows(faith, per_answer_cov, cfg["completeness"]["judges"]) + system_rows(answers)
        main_path = new_artefact_path(out / "a3_chatbot.csv")
        pd.DataFrame(rows).to_csv(main_path, index=False)
        run.add_output(main_path, "A3: results of the chatbot evaluation")

        claims_path = new_artefact_path(out / "a3_claims.parquet")
        claims.to_parquet(claims_path, index=False)
        run.add_output(claims_path, "A3: claims, evidence and labels")

        agreement = pd.DataFrame(agreement_rows(claims, cov, per_answer_cov, verifiers, separate, results, args.dev))
        agreement_path = new_artefact_path(out / "a3_judge_agreement.csv")
        agreement.to_csv(agreement_path, index=False)
        run.add_output(agreement_path, "A3: agreement between the judges and with the NLI model")

        asker = J.Asker("judge_b", "translation", dev=args.dev)
        run.data["calls"] = asker.calls
        text = examples(answers, questions, faith, per_answer_cov, claims, cfg["completeness"]["judges"], asker)
        examples_path = new_artefact_path(out / "a3_examples.md")
        examples_path.write_text(text, encoding="utf-8")
        run.add_output(examples_path, "A3: example answers, Portuguese and draft English")

    with RunManifest(f"{SCRIPT}_costs", out, params={"dev": args.dev}) as run:      # after the report's own calls are logged
        costs = cost_rows()
        costs_path = new_artefact_path(out / "costs_phase6.csv")
        costs.to_csv(costs_path, index=False)
        run.add_output(costs_path, "Phase 6: calls, tokens and cost per step and model")
        run.data["summary"] = {"total_usd": float(costs.iloc[-1]["cost_usd"])}

    table = pd.DataFrame(rows)
    print(table[table["section"].isin(["templated", "unanswerable", "faithfulness", "completeness", "miss_control"])
                & table["group"].eq("all")].drop(columns=["section", "group", "note"]).to_string(index=False))
    print(agreement.drop(columns=["note"]).to_string(index=False))
    print(costs.drop(columns=["basis", "phase"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
