"""Phase 1 step 1a: match the v1 decisions to the frozen v2 corpus (the comparison set).

acordao_hash is a content hash and it changed when production was re-scraped, so v1 and v2 hashes
never coincide. Decisions are matched instead by, in order:
  1. url      the acordão URL, normalised (lower case, no trailing "/view" or "/");
  2. process  a process number from the v1 title equals one in the v2 title or URL, the v2 session
              number is within ±1 of v1's, and exactly one unmatched v2 decision qualifies.
Every match is 1:1. The text diagnostics (word-set Jaccard of the rapporteur reports, same outcome)
are reported, not used for matching: v2 re-extracts the report text with an LLM, so it rarely is
identical to v1's.

Inputs:  data/v2/corpus.parquet (Phase 0; v1 ∩ joblib decisions with v1 fields),
         data/v2/corpus_v2_frozen_<date>.parquet (10_freeze_v2_corpus.py)
Outputs: data/v2/v1_v2_id_map.csv    one row per Phase 0 decision (v2 hash empty if unmatched)
         data/v2/shared_ids.txt      v2 acordao_hash of the matched decisions, sorted
         results/v1_v2_match_summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.manifest import RunManifest

SCRIPT = "11_match_v1_v2"
URL_PROC_RE = re.compile(r"(\d{4,5})-(\d{6,7})-(\d{2,4})-(\d{2})")


def _find_processes():
    spec = importlib.util.spec_from_file_location("export", Path(__file__).with_name("00_export_corpus.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.find_processes


find_processes = _find_processes()


def norm_url(u) -> str:
    return re.sub(r"(/view)?/*$", "", str(u or "").strip().lower())


def words(text) -> set[str]:
    return set(re.findall(r"\w+", str(text or "").lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


def v2_processes(row) -> set[str]:
    procs = {p["digits"] for p in find_processes(row["acordao_titulo"])}
    procs |= {"".join(m) for m in URL_PROC_RE.findall(str(row["acordao_url"] or ""))}
    return procs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", required=True, help="data/v2/corpus_v2_frozen_<date>.parquet")
    args = ap.parse_args()

    paths = load_config("paths")
    out_dir = resolve(paths["data"]["v2"])
    results_dir = resolve(paths["results"])
    p0_path, fz_path = out_dir / "corpus.parquet", resolve(args.frozen)

    with RunManifest(SCRIPT, out_dir, params={"frozen": str(fz_path), "session_tolerance": 1}) as run:
        run.add_input(p0_path)
        run.add_input(fz_path)
        p0 = pd.read_parquet(p0_path)
        fz = pd.read_parquet(fz_path)

        fz_by_url = {norm_url(u): h for u, h in zip(fz["acordao_url"], fz["acordao_hash"])}
        assert len(fz_by_url) == len(fz), "normalised v2 URLs are not unique"
        fz_procs = {h: v2_processes(r) for h, r in zip(fz["acordao_hash"], fz.to_dict("records"))}
        fz_sess = dict(zip(fz["acordao_hash"], fz["sessao_nr_int"]))

        taken: set[str] = set()
        rows = []
        for r in p0.to_dict("records"):
            v2, method = fz_by_url.get(norm_url(r["acordao_url"])), "url"
            if v2 is None or v2 in taken:
                v2, method = None, None
                v1_procs = {p["digits"] for p in find_processes(r["acordao_titulo"])}
                if r.get("process_number_digits"):
                    v1_procs.add(r["process_number_digits"])
                cands = [h for h, ps in fz_procs.items()
                         if h not in taken and ps & v1_procs and pd.notna(fz_sess[h])
                         and abs(int(fz_sess[h]) - int(r["sessao_nr_int"])) <= 1]
                if len(cands) == 1:
                    v2, method = cands[0], "process"
            if v2:
                taken.add(v2)
            rows.append({"v1_acordao_hash": r["acordao_hash"], "v2_acordao_hash": v2, "method": method,
                         "v1_sessao_nr_int": r["sessao_nr_int"]})

        m = pd.DataFrame(rows)
        fz_idx = fz.set_index("acordao_hash")
        p0_idx = p0.set_index("acordao_hash")
        matched = m[m["v2_acordao_hash"].notna()].copy()
        matched["v2_sessao_nr_int"] = matched["v2_acordao_hash"].map(fz_idx["sessao_nr_int"])
        matched["relatorio_jaccard"] = [
            jaccard(words(p0_idx.at[a, "acordao_relatorio"]), words(fz_idx.at[b, "acordao_relatorio"]))
            for a, b in zip(matched["v1_acordao_hash"], matched["v2_acordao_hash"])]
        matched["relatorio_identical"] = [
            str(p0_idx.at[a, "acordao_relatorio"]).strip() == str(fz_idx.at[b, "acordao_relatorio"]).strip()
            for a, b in zip(matched["v1_acordao_hash"], matched["v2_acordao_hash"])]
        matched["same_outcome"] = [
            p0_idx.at[a, "acordao_resultado_curated"] == fz_idx.at[b, "acordao_resultado"]
            for a, b in zip(matched["v1_acordao_hash"], matched["v2_acordao_hash"])]
        m = m.merge(matched.drop(columns=["v1_sessao_nr_int", "method", "v2_acordao_hash"]),
                    on="v1_acordao_hash", how="left")
        assert m["v2_acordao_hash"].dropna().is_unique

        map_path = new_artefact_path(out_dir / "v1_v2_id_map.csv")
        m.to_csv(map_path, index=False)
        run.add_output(map_path, "v1 (Phase 0) -> v2 frozen acordao_hash map")
        shared = sorted(m["v2_acordao_hash"].dropna())
        shared_path = new_artefact_path(out_dir / "shared_ids.txt")
        shared_path.write_text("\n".join(shared) + "\n", encoding="utf-8")
        run.add_output(shared_path, "v2 acordao_hash of the decisions shared with v1 (comparison set)")

        j = matched["relatorio_jaccard"]
        unmatched = m[m["v2_acordao_hash"].isna()]
        summary = {
            "phase0_decisions": len(p0),
            "frozen_decisions": len(fz),
            "matched": len(shared),
            "matched_by": dict(Counter(matched["method"])),
            "unmatched": len(unmatched),
            "unmatched_by_v1_session": {str(int(k)): int(v) for k, v in
                                        sorted(Counter(unmatched["v1_sessao_nr_int"]).items())},
            "session_changed": int((matched["v1_sessao_nr_int"] != matched["v2_sessao_nr_int"]).sum()),
            "relatorio_identical": int(matched["relatorio_identical"].sum()),
            "relatorio_jaccard": {"min": round(float(j.min()), 3), "p05": round(float(j.quantile(.05)), 3),
                                  "median": round(float(j.median()), 3), "below_0.5": int((j < .5).sum())},
            "same_outcome": int(matched["same_outcome"].sum()),
            "outcome_changes": {f"{a} -> {b}": int(n) for (a, b), n in Counter(
                (p0_idx.at[x, "acordao_resultado_curated"], fz_idx.at[y, "acordao_resultado"])
                for x, y, s in zip(matched["v1_acordao_hash"], matched["v2_acordao_hash"],
                                   matched["same_outcome"]) if not s).most_common()},
            "frozen_decisions_in_v1_session_range": int(fz["sessao_nr_int"].between(
                p0["sessao_nr_int"].min(), p0["sessao_nr_int"].max()).sum()),
        }
        run.data["summary"] = summary
        summary_path = new_artefact_path(results_dir / "v1_v2_match_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        run.add_output(summary_path, "aggregate v1/v2 matching counts")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
