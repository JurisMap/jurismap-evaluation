"""Phase 4a, A1: the silver labels reviewed by a second model (the author, 2026-10-07).

A model of another family than the system under test (default: judge A, Grok 4.20 reasoning) checks
100 decisions per label type: the 60 of the author's audit and 40 more (config/a1_labels.yaml, `review`).

- Legal basis: the reviewer sees `acordao_base` and the parsed provisions and says whether the list
  is right (what is missing, what is spurious).
- Infringed provision: it sees the ementa, the report and the vote and says whether the text states
  that the party infringed the provision. Long texts are cut; a provision it does not find is asked
  again with the passages of the full text that cite the act.
- Penalty type: it lists the penalty types from `acordao_penalidade` without seeing the label.
- Infraction category: it labels the decision with the labelling prompt of scripts/41, without
  seeing any label; its category is compared with Gemini's and, where it exists, Mistral's.
- Controls: labels with a planted error (an article dropped or changed; a provision of an act the
  decision never cites) are mixed in, so the share of errors the reviewer catches is measured.

Every response is cached; the manifest holds the call log. Only public CRSNSP text is sent.

Outputs (never overwritten):
  data/labels/a1_label_review.parquet    every verdict
  results/a1_label_review.json           accuracy as judged by the reviewer, agreement with the author's
                                         marks and between the category labellers, controls caught
  results/a1_label_review.xlsx           the 100 decisions with every opinion side by side, and one
                                         row per label the reviewer disputes, for the author to decide
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from jm_eval.io import latest_artefact_path, load_config, new_artefact_path, resolve
from jm_eval.llm import get_model
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import labels as L
from jm_eval.retrieval import review as R
from jm_eval.stats import wilson_interval

SCRIPT = "49_label_review"
MAX_FORMAT_RETRIES = 2
CORPUS_COLUMNS = ["acordao_hash", "acordao_titulo", "acordao_url", "acordao_ementa", "acordao_base", "acordao_penalidade",
                  "acordao_relatorio", "acordao_voto_relator"]


def _labeller():
    spec = importlib.util.spec_from_file_location("infraction_category", Path(__file__).with_name("41_infraction_category.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ask(model, prompt: str, validate, task: str, calls: list) -> dict | None:
    """One checked answer; an unusable answer is asked again, up to `MAX_FORMAT_RETRIES` times."""
    for attempt in range(MAX_FORMAT_RETRIES + 1):
        extra = {"review": R.REVIEW_VERSION, "task": task, **({"format_retry": attempt} if attempt else {})}
        try:
            result = model.complete(prompt, system=R.SYSTEM, json_output=True, cache_extra=extra)
        except ValueError:                           # not JSON
            continue
        answer, problem = validate(result.parsed)
        calls.append({"task": task, "attempt": attempt, "cached": result.cached, "input_tokens": result.input_tokens,
                      "output_tokens": result.output_tokens, "cost_usd": result.cost_usd, "problem": problem})
        if answer:
            return answer
    return None


def run_review(model, labeller, taxonomy: dict, labels: pd.DataFrame, corpus: pd.DataFrame, sample: list[str],
               controls: pd.DataFrame, workers: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """(verdicts on the sample, verdicts on the controls, call log)."""
    lab, cor = labels.set_index("acordao_hash"), corpus.set_index("acordao_hash")
    calls: list = []

    def base_task(h: str, provisions: list[str]) -> dict:
        prompt = R.base_prompt(cor.at[h, "acordao_base"], provisions, cor.at[h, "acordao_penalidade"])
        return ask(model, prompt, R.validate_base, "base", calls) or {}

    def infringed_task(h: str, provision: str) -> dict:
        prompt = R.infringed_prompt(provision, cor.at[h, "acordao_ementa"], cor.at[h, "acordao_relatorio"],
                                    cor.at[h, "acordao_voto_relator"])
        answer = ask(model, prompt, R.validate_infringed, "infringed", calls) or {}
        if answer.get("verdict") in ("no", "not_in_text"):
            # the prompt cuts long texts: look again at the passages of the full text that cite the act
            passages = R.act_passages(provision, cor.at[h, "acordao_relatorio"], cor.at[h, "acordao_voto_relator"])
            if passages:
                second = ask(model, R.passages_prompt(provision, passages), R.validate_passages, "infringed_passages", calls)
                if second:
                    answer = {**second, "second_look": True}
        return answer

    def category_task(h: str) -> dict:
        row = {"acordao_hash": h, "acordao_ementa": cor.at[h, "acordao_ementa"], "acordao_relatorio": cor.at[h, "acordao_relatorio"]}
        record = labeller.label_one(model, taxonomy, row)
        calls.extend({"task": "category", **c} for c in record["calls"])
        return record

    def review_one(h: str) -> dict:
        provisions = list(lab.at[h, "base_provisions"])
        primary = lab.at[h, "text_primary"]
        base = base_task(h, provisions)
        infringed = infringed_task(h, primary) if isinstance(primary, str) else {}
        category = category_task(h)
        label_penalty = list(lab.at[h, "penalty_types"])
        return {
            "acordao_hash": h, "label_base": provisions, "base_verdict": base.get("base_verdict"),
            "base_missing": base.get("base_missing", []), "base_spurious": base.get("base_spurious", []),
            "base_comment": base.get("base_comment", ""),
            "label_penalty": label_penalty, "reviewer_penalty": base.get("penalty_types"),
            "penalty_agree": None if base.get("penalty_types") is None else base["penalty_types"] == label_penalty,
            "label_infringed": primary if isinstance(primary, str) else None,
            "infringed_verdict": infringed.get("verdict"), "infringed_evidence": infringed.get("evidence", ""),
            "infringed_comment": infringed.get("comment", ""), "infringed_second_look": bool(infringed.get("second_look")),
            "reviewer_category": category.get("primary"), "reviewer_category_secondary": category.get("secondary", []),
        }

    def control_one(row) -> dict:
        if row.kind == "base":
            answer = base_task(row.acordao_hash, list(row.label))
            verdict, caught = answer.get("base_verdict"), answer.get("base_verdict") == "wrong"
        else:
            answer = infringed_task(row.acordao_hash, row.label)
            verdict, caught = answer.get("verdict"), answer.get("verdict") == "no"
        return {"acordao_hash": row.acordao_hash, "kind": row.kind, "planted": row.planted, "verdict": verdict,
                "caught": bool(caught) if verdict else None}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = pd.DataFrame(list(pool.map(review_one, sample)))
        control_verdicts = pd.DataFrame(list(pool.map(control_one, controls.itertuples(index=False))),
                                        columns=["acordao_hash", "kind", "planted", "verdict", "caught"])
    return verdicts, control_verdicts, calls


def _share(correct: int, n: int) -> dict:
    low, high = wilson_interval(correct, n)
    return {"n": n, "correct": correct, "share": round(correct / n, 4) if n else None,
            "wilson_95": [round(low, 4), round(high, 4)] if n else None}


def summarise(verdicts: pd.DataFrame, control_verdicts: pd.DataFrame, hugo: pd.DataFrame | None) -> dict:
    """Accuracy as the reviewer judges it, controls caught, and agreement with the author and between labellers."""
    base = verdicts[verdicts["base_verdict"].notna() & verdicts["label_base"].str.len().gt(0)]
    penalty = verdicts[verdicts["penalty_agree"].notna()]
    infringed = verdicts[verdicts["infringed_verdict"].isin(["yes", "no"])]
    out = {
        "decisions": len(verdicts),
        "unusable_answers": {"base": int(verdicts["base_verdict"].isna().sum()),
                             "infringed": int((verdicts["label_infringed"].notna() & verdicts["infringed_verdict"].isna()).sum()),
                             "category": int(verdicts["reviewer_category"].isna().sum())},
        "legal_basis": _share(int((base["base_verdict"] == "correct").sum()), len(base)),
        "penalty_type": _share(int(penalty["penalty_agree"].sum()), len(penalty)),
        "infringed_provision": {**_share(int((infringed["infringed_verdict"] == "yes").sum()), len(infringed)),
                                "not_in_text": int((verdicts["infringed_verdict"] == "not_in_text").sum())},
        "controls": {kind: {"planted": int(len(sub)), "caught": int(sub["caught"].fillna(False).sum())}
                     for kind, sub in control_verdicts.groupby("kind")},
        "category": {f"reviewer vs {column.removeprefix('category_')}": R.agreement(verdicts["reviewer_category"], verdicts[column])
                     for column in ("category_gemini", "category_mistral") if column in verdicts and verdicts[column].notna().any()},
    }
    if hugo is not None and len(hugo):
        merged = verdicts.merge(hugo, on="acordao_hash", how="inner")
        same_label = merged["label_unchanged"]
        pairs = {"legal_basis": ("ok_base", merged["base_verdict"].map({"correct": 1, "wrong": 0}), same_label),
                 "penalty_type": ("ok_penalty", merged["penalty_agree"].map({True: 1, False: 0}), merged["ok_penalty"].notna()),
                 "infringed_provision": ("ok_infringed", merged["infringed_verdict"].map({"yes": 1, "no": 0}), same_label)}
        agreement = {}
        for name, (column, reviewer, usable) in pairs.items():
            both = usable & merged[column].isin([0, 1]) & reviewer.notna() & merged[name_label(name)].apply(has_label)
            agreement[name] = {"n": int(both.sum()), "same": int((merged.loc[both, column] == reviewer[both]).sum()),
                               "hugo_correct_reviewer_wrong": int(((merged.loc[both, column] == 1) & (reviewer[both] == 0)).sum()),
                               "hugo_wrong_reviewer_correct": int(((merged.loc[both, column] == 0) & (reviewer[both] == 1)).sum())}
        out["agreement_with_hugo"] = agreement
        if "hugo_category" in merged and merged["hugo_category"].notna().any():
            out["category"]["reviewer vs hugo"] = R.agreement(merged["reviewer_category"], merged["hugo_category"])
            out["category"]["gemini vs hugo"] = R.agreement(merged["category_gemini"], merged["hugo_category"])
    return out


def name_label(name: str) -> str:
    return {"legal_basis": "label_base", "penalty_type": "label_penalty", "infringed_provision": "label_infringed"}[name]


def has_label(value) -> bool:
    """A label is present: a non-empty string or list (missing values are None or NaN)."""
    try:
        return len(value) > 0
    except TypeError:
        return False


def hugo_marks(sheet: pd.DataFrame, audited_labels: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """the author's marks per decision, and whether the label the author marked is still the current one."""
    old, new = audited_labels.set_index("acordao_hash"), current.set_index("acordao_hash")
    rows = []
    for r in sheet.itertuples(index=False):
        h = r.acordao_hash
        unchanged = list(old.at[h, "base_provisions"]) == list(new.at[h, "base_provisions"]) and \
            (old.at[h, "text_primary"] == new.at[h, "text_primary"] or (pd.isna(old.at[h, "text_primary"]) and pd.isna(new.at[h, "text_primary"])))
        row = {"acordao_hash": h, "audit_id": r.audit_id, "ok_base": r.ok_base, "ok_penalty": r.ok_penalty,
               "ok_infringed": r.ok_infringed, "label_unchanged": bool(unchanged)}
        if "ok_category" in sheet.columns:
            ok, fixed, labelled = r.ok_category, r.correct_category, str(r.label_category).split(":")[0]
            row["hugo_category"] = labelled if ok == 1 else (str(fixed).strip() if ok == 0 and isinstance(fixed, str) else None)
        rows.append(row)
    return pd.DataFrame(rows)


def disputes(table: pd.DataFrame) -> pd.DataFrame:
    """One row per label that the reviewer disputes (or that the category labellers split on)."""
    rows = []
    for r in table.itertuples(index=False):
        if r.base_verdict == "wrong":
            rows.append((r.acordao_hash, "legal basis", "; ".join(r.label_base),
                         f"missing: {'; '.join(r.base_missing) or '-'} | spurious: {'; '.join(r.base_spurious) or '-'}", r.base_comment))
        if r.penalty_agree is False:
            rows.append((r.acordao_hash, "penalty type", " + ".join(r.label_penalty), " + ".join(r.reviewer_penalty), ""))
        if r.infringed_verdict == "no":
            rows.append((r.acordao_hash, "infringed provision", r.label_infringed, "not stated as infringed", r.infringed_comment))
        opinions = {name: getattr(r, name, None) for name in ("category_gemini", "category_mistral", "reviewer_category")}
        given = {k: v for k, v in opinions.items() if isinstance(v, str)}
        if len(set(given.values())) > 1:
            rows.append((r.acordao_hash, "infraction category", given.get("category_gemini", ""),
                         " | ".join(f"{k.replace('category_', '').replace('reviewer_category', 'reviewer')}: {v}" for k, v in given.items()), ""))
    out = pd.DataFrame(rows, columns=["acordao_hash", "label_type", "label", "reviewer", "comment"])
    out["hugo_decision"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewer", default="judge_grok", help="entry of `judges` in config/models.yaml")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    cfg, paths = L.settings(), load_config("paths")
    review_cfg = cfg["review"]
    results, labels_dir = resolve(paths["results"]), resolve(paths["data"]["labels"])
    labeller = _labeller()
    taxonomy = labeller.load_taxonomy()
    model = get_model("judges", args.reviewer)
    with RunManifest(SCRIPT, results, models={"reviewer": model.model},
                     params={"review": review_cfg, "review_version": R.REVIEW_VERSION, "workers": args.workers,
                             "taxonomy_version": taxonomy["version"]}) as run:
        labels_path = latest_artefact_path(labels_dir / "a1_silver_labels.parquet")
        labels = pd.read_parquet(labels_path)
        run.add_input(labels_path)
        corpus = pd.read_parquet(resolve(cfg["corpus"]), columns=CORPUS_COLUMNS)
        audit_path = latest_artefact_path(results / "labels_audit.xlsx")
        sheet = pd.read_excel(audit_path, sheet_name="audit")
        run.add_input(audit_path)
        excluded = set(cfg["development"]["ids"]) | set(cfg["audit"].get("excluded_ids") or [])
        sample = R.review_sample(labels, sheet["acordao_hash"].tolist(), excluded, review_cfg["n"], review_cfg["seed"])
        controls = R.controls(labels, sample, excluded, review_cfg["controls"]["legal_basis"],
                              review_cfg["controls"]["infringed_provision"], review_cfg["seed"])

        verdicts, control_verdicts, calls = run_review(model, labeller, taxonomy, labels, corpus, sample, controls, args.workers)
        for column, name in (("category_gemini", "a1_infraction_category.parquet"),
                             ("category_mistral", "a1_infraction_category_mistral.parquet")):
            path = latest_artefact_path(labels_dir / name)
            if path.exists():
                other = pd.read_parquet(path, columns=["acordao_hash", "primary"]).rename(columns={"primary": column})
                verdicts = verdicts.merge(other, on="acordao_hash", how="left")
        hugo = hugo_marks(sheet, pd.read_parquet(labels_dir / review_cfg["audited_labels"]), labels)
        summary = summarise(verdicts, control_verdicts, hugo)
        corpus_categories = [pd.read_parquet(p, columns=["acordao_hash", "primary"]).set_index("acordao_hash")["primary"]
                             for p in (latest_artefact_path(labels_dir / "a1_infraction_category.parquet"),
                                       latest_artefact_path(labels_dir / "a1_infraction_category_mistral.parquet")) if p.exists()]
        if len(corpus_categories) == 2:              # the two labellings of the whole corpus
            gemini, mistral = corpus_categories[0].align(corpus_categories[1], join="inner")
            summary["category"]["gemini vs mistral (corpus)"] = R.agreement(gemini, mistral)

        verdicts_path = new_artefact_path(labels_dir / "a1_label_review.parquet")
        verdicts.to_parquet(verdicts_path, index=False)
        run.add_output(verdicts_path, f"A1 silver labels: verdicts of the reviewing model ({model.model})")
        summary_path = new_artefact_path(results / "a1_label_review.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(summary_path, "A1 silver labels: review by a second model, summary")

        table = verdicts.merge(hugo.drop(columns=["label_unchanged"]), on="acordao_hash", how="left")
        fields = corpus[["acordao_hash", "acordao_titulo", "acordao_url", "acordao_base", "acordao_penalidade", "acordao_ementa"]]
        table = fields.merge(table, on="acordao_hash", how="right")
        sheet_path = new_artefact_path(results / "a1_label_review.xlsx")
        join = lambda v: "; ".join(v) if isinstance(v, (list, tuple)) or hasattr(v, "tolist") else v  # noqa: E731
        with pd.ExcelWriter(sheet_path, engine="openpyxl") as writer:
            disputes(verdicts).merge(fields, on="acordao_hash", how="left").to_excel(writer, sheet_name="to_decide", index=False)
            table.map(join).to_excel(writer, sheet_name="review", index=False)
            control_verdicts.to_excel(writer, sheet_name="controls", index=False)
        run.add_output(sheet_path, "A1 silver labels: every opinion side by side; disputed labels for the author to decide")

        paid = [c for c in calls if not c["cached"]]
        run.data["summary"] = {**summary, "calls": len(calls), "calls_not_cached": len(paid),
                               "input_tokens": sum(c["input_tokens"] or 0 for c in paid),
                               "output_tokens": sum(c["output_tokens"] or 0 for c in paid),
                               "cost_usd": round(sum(c["cost_usd"] or 0 for c in paid), 4)}
        run.data["calls"] = calls
        print(json.dumps(run.data["summary"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
