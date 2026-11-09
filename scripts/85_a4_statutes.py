"""Phase 8, A4 (task 4): official texts of the main normative acts, for the statute-validity check.

For each act of config/a4_drafting.yaml `statutes.acts`: the federal acts are downloaded from Planalto
(the compiled HTML), the CNSP resolutions and SUSEP circulars from SUSEP's normative database (the
record whose title is the act's and whose year is the act's year; every attachment marked "Versão
Original" or "Versão Compilada" whose file name carries the act's number). Each file is stored with
its URL, date and sha256; a file already downloaded is not fetched again. The article numbers of each
act are then read from its texts.

Outputs:
  data/v2/a4/statutes/<act>/...            the downloaded files and meta.json (URL, date, hash)
  results/a4_statutes.json                 per act: sources, article numbers found, last article
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone

import requests

from jm_eval.drafting import notices as N
from jm_eval.io import load_config, new_artefact_path, resolve, sha256_file
from jm_eval.manifest import RunManifest

SCRIPT = "85_a4_statutes"
# "Art. 2º", "Art . 2º" (Planalto), "Art. 1.338" (thousands), "Art. 4º-A"
ARTICLE = re.compile(r"\bArt(?:igo)?s?\s*\.?\s*(\d{1,3}\.\d{3}|\d{1,4})\s*(?:[º°o]|\.)?(?:\s*-\s*([A-Z])\b)?")


def article_numbers(text: str) -> set[str]:
    """Article numbers of a legal text: "Art. 2º" -> "2", "Art. 4º-A" -> "4-A", "Art. 1.338" -> "1338"."""
    return {f"{n}-{s}" if s else n
            for n, s in ((str(int(m.group(1).replace(".", ""))), m.group(2)) for m in ARTICLE.finditer(text))}


def html_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="ignore") if b"charset=utf-8" in raw[:3000].lower() else raw.decode("latin-1", errors="ignore")
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


def pdf_text(path) -> str:
    import pypdf

    return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(path)).pages)


def fetch(session: requests.Session, url: str, dest, params: dict | None = None) -> dict:
    if not dest.exists():
        r = session.get(url, params=params, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        retrieved = None
    return {"file": dest.name, "url": url + (("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""),
            "retrieved_utc": retrieved, "sha256": sha256_file(dest), "bytes": dest.stat().st_size}


def susep_files(session: requests.Session, api: str, key: str, title: str, folder) -> list[dict]:
    number, year = re.search(r"(\d+)/(\d{4})$", key).groups()
    r = session.get(api, params={"router": "search", "exp": f'"{title}"', "limit": 20, "expand": "anexos"}, timeout=60)
    r.raise_for_status()
    records = [x for x in r.json()["data"] if x.get("titulo", "").strip() == title and str(x.get("datadoc", "")).endswith(year)]
    if len(records) != 1:
        raise ValueError(f"{key}: {len(records)} records titled {title!r} in {year}")
    out = []
    for a in records[0].get("anexos") or []:
        name = (a.get("arquivo") or "").strip()
        if (a.get("descricao") or "").strip() in ("Versão Original", "Versão Compilada") and re.search(rf"(?<!\d)0*{number}(?!\d)", name):
            meta = fetch(session, api, folder / name, {"router": f"upload/{a['cod_anexo']}"})
            out.append({**meta, "version": a["descricao"].strip(), "susep_record": records[0]["codigo"]})
    if not out:
        raise ValueError(f"{key}: no original or compiled text among the attachments")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    spec = N.settings()["statutes"]
    root = resolve(load_config("paths")["data"]["v2"]) / "a4" / "statutes"
    session = requests.Session()
    session.headers["User-Agent"] = spec["user_agent"]
    table, problems = {}, {}
    with RunManifest(SCRIPT, root.parent, params={"statutes": spec}) as run:
        for key, act in spec["acts"].items():
            folder = root / key.replace("/", "_")
            folder.mkdir(parents=True, exist_ok=True)
            try:
                if "planalto" in act:
                    files = [{**fetch(session, act["planalto"], folder / act["planalto"].rsplit("/", 1)[-1]), "version": "Planalto"}]
                else:
                    files = susep_files(session, spec["susep_api"], key, act["susep_title"], folder)
            except Exception as e:                      # reported, not fatal: the act's citations become manual review
                problems[key] = str(e)
                print(f"{key}: {e}")
                continue
            meta_path = folder / "meta.json"
            old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            for f in files:                              # keep the first retrieval date of a file
                f["retrieved_utc"] = f["retrieved_utc"] or next((o["retrieved_utc"] for o in old.get("files", []) if o["file"] == f["file"]), None)
            meta_path.write_text(json.dumps({"act": key, "name": act["name"], "files": files}, indent=1, ensure_ascii=False), encoding="utf-8")
            articles: set[str] = set()
            for f in files:
                path = folder / f["file"]
                text = pdf_text(path) if path.suffix.lower() == ".pdf" else html_text(path.read_bytes())
                f["articles"] = len(article_numbers(text))
                articles |= article_numbers(text)
            numbers = [int(a.split("-")[0]) for a in articles]
            table[key] = {"name": act["name"], "files": files, "articles": sorted(articles, key=lambda a: (int(a.split("-")[0]), a)),
                          "last_article": max(numbers) if numbers else None}
            print(f"{key}: {len(files)} file(s), {len(articles)} articles, last {table[key]['last_article']}", flush=True)
        out = new_artefact_path(resolve(load_config("paths")["results"]) / "a4_statutes.json")
        out.write_text(json.dumps({"acts": table, "problems": problems}, indent=1, ensure_ascii=False), encoding="utf-8")
        run.add_output(out, "A4: article numbers of the main normative acts, from their official texts")
        run.data["summary"] = {"acts": len(table), "problems": problems}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
