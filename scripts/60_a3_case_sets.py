"""Phase 6, A3 (task 1): the case sets of the chatbot evaluation.

Six sets of 10 decisions of version 2's corpus, built as the app builds a selection from v2.4 on:
a seed decision plays the user's case and the set is its 10 most similar decisions by the retrieval
vectors, without the seed and the other decisions of its process. One seed is drawn at random in
each of the six most frequent legal-basis keys of the Phase 4a labels. A further set, from the next
key, is the development set on which prompts and scoring rules are tried; it is never reported.

The decisions of these sets (seeds included) are not to be used as Phase 8 test notices.

Outputs (never overwritten):
  data/v2/a3/case_sets.json     seeds, members (`case_hashes`) and similarities
  results/a3_case_sets.csv      one row per set: stratum, what its decisions share, sizes
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from jm_eval.chatbot import casesets, store
from jm_eval.io import new_artefact_path
from jm_eval.manifest import RunManifest
from jm_eval.retrieval import corpus as C

SCRIPT = "60_a3_case_sets"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = casesets.settings()
    spec = cfg["case_sets"]
    with RunManifest(SCRIPT, store.results_dir(), params={"case_sets": spec}, seeds={"case_sets": spec["seed"]}) as run:
        labels = C.load_labels()
        vectors = C.method_vectors(spec["vector_method"]).astype(np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        corpus = store.load_corpus()
        assert labels["acordao_hash"].tolist() == corpus["acordao_hash"].tolist()
        sets = casesets.build_sets(labels, vectors, n_sets=spec["n_sets"], size=spec["size"], seed=spec["seed"],
                                   development_sets=spec["development_sets"])
        members = [h for s in sets for h in s["case_hashes"]]
        assert len(members) == len(set(members)) and not set(members) & {s["seed_hash"] for s in sets}

        sets_path = new_artefact_path(store.work_dir() / "case_sets.json")
        sets_path.write_text(json.dumps({"vector_method": spec["vector_method"], "sets": sets}, indent=1), encoding="utf-8")
        run.add_output(sets_path, "A3: case sets (seeds, members, similarities)")
        table = casesets.describe(sets, labels, corpus)
        table_path = new_artefact_path(store.results_dir() / "a3_case_sets.csv")
        table.to_csv(table_path, index=False)
        run.add_output(table_path, "A3: case sets, one row per set")
        run.data["summary"] = {"sets": len(sets), "decisions": len(members)}
        print(table.drop(columns=["seed_hash"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
