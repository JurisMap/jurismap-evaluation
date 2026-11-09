"""Phase 6, A3 (task 5): faithfulness of the answers to the open questions, claim by claim.

The chatbot answers from the full text of 10 decisions (tens of thousands of tokens), which is
more than the judges' per-minute token quotas allow. The judges therefore see claims and a few
evidence passages, never the whole context:

1. Claims: Mistral Medium 3 reads the answer alone and lists its atomic claims.
2. Evidence, on this machine: the 10 decisions are split into passages of about 150 words, ranked
   for each claim by reciprocal-rank fusion of BGE-M3 and BM25. A claim that names decisions gets
   the best passages of those decisions (4 of one, 2 of each of two or three); any other claim
   gets the 4 best passages of the whole set.
3. Verification: a local NLI model labels every claim; GPT-5.4 and Mistral Medium 3 label batches
   of up to 5 claims as supported, contradicted or not found. Gemini 3.8 Flash labels the batches
   of the first run of each question and is reported separately (same family as the system).
4. Retrieval-miss control: a random 20% (at least 30) of the claims that a judge labelled not found
   are judged again by GPT-5.4 against the full text of the decision they come from: the decision the
   claim names (each of up to three), else the decision of its best passage.

`--setup-only` downloads the NLI model once into data/cache/hf. `--dev` runs the development set.
Every prompt stays under 8K tokens; the manifest holds the call log.

Outputs (never overwritten):
  data/v2/a3/claims.parquet     claims, evidence and every label
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from jm_eval.chatbot import evidence as E
from jm_eval.chatbot import judging as J
from jm_eval.chatbot import metrics as M
from jm_eval.chatbot import nli as N
from jm_eval.chatbot import store
from jm_eval.io import load_config, new_artefact_path
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import dense_local as D

SCRIPT = "64_a3_faithfulness"
BATCH_TOKEN_BUDGET = 7000          # estimated tokens of a verification prompt (the limit is 8,000)
RANK = {"supported": 2, "contradicted": 1, "not_found": 0}


def decompose(asker: J.Asker, answers: pd.DataFrame) -> pd.DataFrame:
    def one(a: dict) -> list[dict]:
        result = asker.ask(J.DECOMPOSE_PROMPT.format(answer=a["answer"]), J.validate_claims, item=a["answer_id"])
        if result is None:
            raise RuntimeError(f"no usable claims for {a['answer_id']}")
        return [{"answer_id": a["answer_id"], "question_id": a["question_id"], "set_id": a["set_id"], "run": a["run"],
                 "claim_id": f"{a['answer_id']}/c{i}", "claim": c["claim"], "processes": c["processes"]}
                for i, c in enumerate(result["claims"], 1)]

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [r for rs in pool.map(one, answers.to_dict("records")) for r in rs]
    return pd.DataFrame(rows, columns=["answer_id", "question_id", "set_id", "run", "claim_id", "claim", "processes"])


def retrieve(claims: pd.DataFrame, questions: dict, corpus: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Adds to each claim its evidence passages and the decision it most likely comes from."""
    model = D.load_model(spec["dense_model"], spec["dense_max_seq_length"], offline=True)
    by_hash = corpus.set_index("acordao_hash", drop=False)
    evidence, cited_all, likely, stats = {}, {}, {}, {}
    for set_id, group in claims.groupby("set_id", sort=True):
        members = by_hash.loc[questions[group["question_id"].iloc[0]]["case_hashes"]]
        passages = E.build_passages(members, spec["passage_words"], spec["passage_overlap_words"])
        passage_vectors, _ = D.encode(model, passages["text"].tolist())
        claim_vectors, _ = D.encode(model, group["claim"].tolist())
        dense = claim_vectors @ passage_vectors.T
        bm25 = E.bm25_query_scores(passages["text"].tolist(), group["claim"].tolist())
        stats[set_id] = {"passages": len(passages), "claims": len(group)}
        for i, c in enumerate(group.to_dict("records")):
            cited = E.cited_hashes(list(c["processes"]), c["claim"], members)
            chosen = E.select_evidence(dense[i], bm25[i], passages, cited, top=spec["top_passages"],
                                       named={int(k): v for k, v in spec["named_passages"].items()},
                                       beyond=spec["named_passages_beyond"], limit=spec["max_passages"], rrf_k=spec["rrf_k"])
            evidence[c["claim_id"]] = passages["text"].iloc[chosen].tolist()
            cited_all[c["claim_id"]] = cited
            likely[c["claim_id"]] = E.most_likely_decision(chosen, passages, cited)
    out = claims.copy()
    out["evidence"] = out["claim_id"].map(evidence)
    out["cited_hashes"] = out["claim_id"].map(cited_all)
    out["likely_hash"] = out["claim_id"].map(likely)
    return out, stats


def nli_labels(claims: pd.DataFrame, spec: dict) -> pd.DataFrame:
    nli = N.load(spec["nli_model"], offline=True)
    premises = [p for ev in claims["evidence"] for p in ev]
    hypotheses = [c for c, ev in zip(claims["claim"], claims["evidence"]) for _ in ev]
    probabilities = N.probabilities(nli, premises, hypotheses)
    out, start = claims.copy(), 0
    labels, entail, contradict = [], [], []
    for ev in claims["evidence"]:
        block = probabilities[start:start + len(ev)]
        start += len(ev)
        labels.append(N.claim_label(block, spec["nli_threshold"]))
        entail.append(float(block[:, 0].max()) if len(block) else np.nan)
        contradict.append(float(block[:, 2].max()) if len(block) else np.nan)
    out["nli"], out["nli_entailment_max"], out["nli_contradiction_max"] = labels, entail, contradict
    return out


def item_text(number: int, claim: str, passages: list[str]) -> str:
    listed = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return f"ITEM {number}\nCLAIM: {claim}\nEVIDENCE:\n{listed}"


def batches(claims: pd.DataFrame, size: int) -> list[list[dict]]:
    """Claims of one answer in order, up to `size` per batch and under the token budget."""
    out = []
    for _, group in claims.groupby("answer_id", sort=False):
        current: list[dict] = []
        for c in group.to_dict("records"):
            trial = current + [c]
            prompt = J.VERIFY_PROMPT.format(items="\n\n".join(item_text(i, x["claim"], list(x["evidence"])) for i, x in enumerate(trial, 1)))
            if current and (len(trial) > size or J.estimated_tokens(prompt) > BATCH_TOKEN_BUDGET):
                out.append(current)
                current = [c]
            else:
                current = trial
        if current:
            out.append(current)
    return out


def verify(asker: J.Asker, claim_batches: list[list[dict]], workers: int = 3) -> dict[str, str | None]:
    def one(batch: list[dict]) -> dict[str, str | None]:
        prompt = J.VERIFY_PROMPT.format(items="\n\n".join(item_text(i, c["claim"], list(c["evidence"])) for i, c in enumerate(batch, 1)))
        result = asker.ask(prompt, lambda p: J.validate_verdicts(p, len(batch)), item=batch[0]["claim_id"])
        verdicts = result["verdicts"] if result else [{"label": None}] * len(batch)
        return {c["claim_id"]: v["label"] for c, v in zip(batch, verdicts)}

    labels: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(one, claim_batches):
            labels.update(part)
    return labels


def full_text_parts(row: dict, part_chars: int) -> list[str]:
    """Everything the chatbot received of one decision, in parts of at most `part_chars` characters."""
    header = "; ".join(f"{label}: {J.field_text(row[field])}" for label, field in E.HEADER_FIELDS)
    body = "\n".join([f"TÍTULO: {J.field_text(row['acordao_titulo'])}", header]
                     + [f"{label.upper()}:\n{J.field_text(row[field])}" for label, field in E.LONG_FIELDS])
    return [body[i:i + part_chars] for i in range(0, len(body), part_chars)] or [""]


MAX_CONTROL_DECISIONS = 3


def combine_control(labels: list[str | None]) -> str | None:
    """A claim about several decisions: supported only if each supports it; contradicted if any contradicts it."""
    if any(label is None for label in labels):
        return None
    if "contradicted" in labels:
        return "contradicted"
    return "supported" if all(label == "supported" for label in labels) else "not_found"


def miss_control(asker: J.Asker, claims: pd.DataFrame, corpus: pd.DataFrame, spec: dict, verifiers: list[str]) -> tuple[pd.DataFrame, int]:
    """Re-judges a sample of the not-found claims against the full text of the decisions they come from.

    The decision is the one the claim names (up to three: the claim must hold for each), else the
    decision of its best passage. A long decision is judged in parts and the best verdict counts.
    """
    pool_ids = sorted(claims.loc[(claims[verifiers] == "not_found").any(axis=1), "claim_id"])
    n = min(len(pool_ids), max(spec["minimum"], int(np.ceil(spec["share"] * len(pool_ids)))))
    rng = np.random.default_rng(spec["seed"])
    sample = sorted(rng.choice(pool_ids, size=n, replace=False)) if n else []
    by_hash = corpus.set_index("acordao_hash", drop=False)
    rows = claims[claims["claim_id"].isin(sample)].to_dict("records")

    def one_decision(c: dict, h: str, several: bool) -> tuple[str | None, str]:
        parts = full_text_parts(by_hash.loc[h].to_dict(), spec["part_chars"])
        best, quote = None, ""
        for k, part in enumerate(parts, 1):
            note = f" (part {k} of {len(parts)} of a long decision)" if len(parts) > 1 else ""
            prompt = J.FULL_TEXT_PROMPT.format(claim=c["claim"], decision=part, part_note=note,
                                               scope_note=J.SEVERAL_DECISIONS_NOTE if several else "")
            result = asker.ask(prompt, J.validate_label, item=f"{c['claim_id']}|{h}|part{k}")
            if result and (best is None or RANK[result["label"]] > RANK[best]):
                best, quote = result["label"], result["quote"]
            if best == "supported":
                break
        return best, quote

    def one(c: dict) -> dict:
        decisions = list(c["cited_hashes"])[:MAX_CONTROL_DECISIONS] or [c["likely_hash"]]
        verdicts = [one_decision(c, h, len(decisions) > 1) for h in decisions]
        return {"claim_id": c["claim_id"], "control_hashes": decisions,
                "full_text_label": combine_control([label for label, _ in verdicts]),
                "full_text_quote": " | ".join(quote for _, quote in verdicts if quote)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        out = list(pool.map(one, rows))
    return pd.DataFrame(out, columns=["claim_id", "control_hashes", "full_text_label", "full_text_quote"]), len(pool_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--setup-only", action="store_true", help="download the NLI model into data/cache/hf and stop")
    ap.add_argument("--without-separate", action="store_true", help="development only: skip the same-family judge")
    ap.add_argument("--candidate", help="Phase 6.1: a strategy of config/a3_candidates.yaml, on the development sets")
    ap.add_argument("--setting", help="Phase 7: a context size of config/a5_context_size.yaml, on the evaluated questions")
    ap.add_argument("--judges", nargs="+", help="only these verifiers of the config (a later run adds the others from the cache)")
    ap.add_argument("--skip-control", action="store_true", help="leave the retrieval-miss control to a later run")
    ap.add_argument("--reuse-labels", action="store_true",
                    help="keep the judges' labels of the newest claims table for the claims whose evidence is unchanged")
    ap.add_argument("--dry-run", action="store_true", help="with --reuse-labels: count the claims to judge again and stop")
    args = ap.parse_args()
    args.dev = args.dev or bool(args.candidate)
    args.without_separate = args.without_separate or bool(args.candidate)     # the selection uses the two judges only
    store.use_candidate(args.candidate)
    store.use_setting(args.setting)
    cfg = load_config("a3_chatbot")
    spec = cfg["faithfulness"]
    if args.setup_only:
        N.load(spec["nli_model"], offline=False, device="cpu")
        print(f"{spec['nli_model']} is in data/cache/hf")
        return 0
    work, judges = store.work_dir(args.dev), load_config("models")["judges"]
    verifiers, separate = spec["verifiers"], [] if args.without_separate and args.dev else spec["separate"]
    if args.setting:                         # Phase 7: the judges of config/a5_context_size.yaml
        verifiers, separate = load_config("a5_context_size")["judges"]["verifiers"], load_config("a5_context_size")["judges"]["separate"]
    if args.judges:
        verifiers, separate = [v for v in verifiers if v in args.judges], [s for s in separate if s in args.judges]
    models = {"decomposition": judges[spec["decomposition_model"]]["model"], "dense": spec["dense_model"],
              "nli": spec["nli_model"], **{name: judges[name]["model"] for name in verifiers + separate},
              "miss_control": judges[spec["miss_control"]["model"]]["model"]}
    with RunManifest(SCRIPT, work, models=models, params={"dev": args.dev, "candidate": args.candidate, "setting": args.setting,
                                                          "judges": verifiers + separate,
                                                          "skip_control": args.skip_control, "faithfulness": spec,
                                                          "prompts_version": J.PROMPTS_VERSION,
                                                          "batch_token_budget": BATCH_TOKEN_BUDGET},
                     seeds={"miss_control": spec["miss_control"]["seed"]}) as run:
        questions = {q["question_id"]: q for q in store.load_questions(args.dev)}
        answers = store.load_table("answers.parquet", args.dev)
        answers = answers[answers["type"] == "open"]
        corpus = store.load_corpus()

        calls: list = []
        run.data["calls"] = calls
        limits = load_config("a5_context_size")["prompt_limits"]["decomposition"] if args.setting else {}
        decomposer = J.Asker(spec["decomposition_model"], "decomposition", dev=args.dev, log=calls, **limits)
        claims = decompose(decomposer, answers)
        claims, retrieval = retrieve(claims, questions, corpus, spec)

        kept = pd.DataFrame()
        to_judge = claims
        if args.reuse_labels:                       # after a change of the evidence retrieval: judge again only what it changed
            previous = store.load_table("claims.parquet", args.dev).set_index("claim_id")
            same = [c in previous.index and list(previous.at[c, "evidence"]) == list(e) for c, e in zip(claims["claim_id"], claims["evidence"])]
            kept = previous.loc[claims.loc[same, "claim_id"], [n for n in verifiers + separate if n in previous.columns]]
            to_judge = claims[[not s for s in same]]
            print(json.dumps({"claims": len(claims), "evidence_unchanged": int(sum(same)), "to_judge_again": len(to_judge),
                              "of_which_first_runs": int((to_judge["run"] == 1).sum())}))
            if args.dry_run:
                return 0
        claims = nli_labels(claims, spec)
        claim_batches = batches(to_judge, spec["batch_claims"])
        askers = {name: J.Asker(name, "verification", dev=args.dev, log=calls) for name in verifiers + separate}
        # the same-family judge labels only the claims of the runs named in `separate_runs` (all runs if unset)
        runs = spec.get("separate_runs")
        first_runs = [b for b in claim_batches if b[0]["run"] in runs] if runs else claim_batches
        with ThreadPoolExecutor(max_workers=len(askers)) as pool:
            futures = {name: pool.submit(verify, asker, first_runs if name in separate else claim_batches)
                       for name, asker in askers.items()}
        for name, future in futures.items():
            labels = future.result()
            if name in kept:
                labels = {**kept[name].to_dict(), **labels}
            claims[name] = claims["claim_id"].map(labels)
        claims["majority"] = [M.majority(list(row)) for row in claims[["nli", *verifiers]].itertuples(index=False)]

        controller = J.Asker(spec["miss_control"]["model"], "miss_control", dev=args.dev, log=calls)
        if args.skip_control:
            control = pd.DataFrame(columns=["claim_id", "control_hashes", "full_text_label", "full_text_quote"])
            pool_size = int((claims[verifiers] == "not_found").any(axis=1).sum())
        else:
            control, pool_size = miss_control(controller, claims, corpus, spec["miss_control"], verifiers)
        claims = claims.merge(control, on="claim_id", how="left")

        path = new_artefact_path(work / "claims.parquet")
        claims.to_parquet(path, index=False)
        run.add_output(path, "A3: claims, evidence and labels" + (" (development set)" if args.dev else ""))

        rate = M.miss_rate(control)
        summary = {"answers": int(answers["answer_id"].nunique()), "claims": len(claims),
                   "claims_per_answer": round(len(claims) / max(1, answers["answer_id"].nunique()), 1),
                   "evidence_passages_per_claim": round(float(claims["evidence"].map(len).mean()), 2) if len(claims) else None,
                   "retrieval": retrieval, "verification_batches": len(claim_batches),
                   "not_found_by_a_judge": pool_size, "miss_control": rate,
                   "verifiers": {name: {**M.faithfulness(claims, name), **M.corrected(claims, name, control, rate["miss_rate"])}
                                 for name in ["nli", *verifiers, "majority", *separate]},
                   "calls": {f"{step}/{role}": J.calls_summary([c for c in calls if c["step"] == step and c["role"] == role])
                             for step, role in sorted({(c["step"], c["role"]) for c in calls})},
                   "max_input_tokens": max((c["input_tokens"] or 0 for c in calls), default=0)}
        run.data["summary"] = summary
        print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
