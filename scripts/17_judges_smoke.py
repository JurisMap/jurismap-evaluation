"""Phase 1: smoke test of the three LLM judges on one faithfulness case (fictitious, not SUSEP data).

The answer has 4 claims: 2 supported by the context and 2 invented. A correct judge marks claims
1 and 3 supported, 2 and 4 unsupported (faithfulness 0.5).
Output: results/judges_smoke.json (verdicts, tokens, cost, latency) + run manifest.
"""

from __future__ import annotations

import json

from jm_eval.io import load_config, new_artefact_path, resolve
from jm_eval.llm import get_judge
from jm_eval.manifest import RunManifest

SCRIPT = "17_judges_smoke"
CONTEXT = (
    "Acórdão CRSNSP (fictício). A Seguradora Exemplo S.A. recorreu de multa de R$ 40.000,00 aplicada pela "
    "SUSEP por atraso no envio de informações periódicas (art. 88 do Decreto-Lei nº 73/1966). O Conselho, "
    "por unanimidade, deu parcial provimento ao recurso para reduzir a multa a R$ 20.000,00, reconhecendo "
    "a primariedade da recorrente."
)
CLAIMS = [
    "A multa original foi de R$ 40.000,00.",
    "A recorrente alegou prescrição da pretensão punitiva.",
    "O recurso teve provimento parcial e a multa foi reduzida para R$ 20.000,00.",
    "A decisão foi tomada por maioria, com dois votos vencidos.",
]
EXPECTED = [True, False, True, False]
PROMPT = (
    "Você avalia a fidelidade de uma resposta a um contexto. Para cada afirmação, diga se ela é "
    "sustentada pelo contexto (true) ou não (false). Responda apenas com JSON no formato "
    '{{"claims": [{{"id": 1, "supported": true}}, ...], "faithfulness": <fração sustentada>}}.\n\n'
    "Contexto:\n{context}\n\nAfirmações:\n{claims}"
)


def main() -> int:
    paths = load_config("paths")
    judges = ["judge_a", "judge_b", "judge_gemini"]
    models = {j: load_config("models")["judges"][j]["model"] for j in judges}
    prompt = PROMPT.format(context=CONTEXT, claims="\n".join(f"{i}. {c}" for i, c in enumerate(CLAIMS, 1)))
    out_dir = resolve(paths["data"]["root"]) / "judges_smoke"
    results = {}
    with RunManifest(SCRIPT, out_dir, models=models, params={"expected": EXPECTED}) as run:
        for name in judges:
            r = get_judge(name).complete(prompt, json_output=True)
            verdicts = [bool(c["supported"]) for c in sorted(r.parsed["claims"], key=lambda c: c["id"])]
            results[name] = {"model": r.model, "verdicts": verdicts, "correct": verdicts == EXPECTED,
                             "faithfulness": r.parsed.get("faithfulness"), "input_tokens": r.input_tokens,
                             "output_tokens": r.output_tokens, "reasoning_tokens": r.reasoning_tokens,
                             "cost_usd": r.cost_usd, "latency_s": r.latency_s, "cached": r.cached,
                             "credentials": r.credentials}
        run.data["summary"] = results
    path = new_artefact_path(resolve(paths["results"]) / "judges_smoke.json")
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(r["correct"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
