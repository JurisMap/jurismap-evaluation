"""Download the NLTK data the lexical retrieval methods use (Portuguese stop words, RSLP stemmer).

They go to the directory config/a1_methods.yaml names (`nltk_data`, default data/cache/nltk_data).
`scripts/44_a1_local_embeddings.py --setup-only` does the same and also downloads the local embedding models.
"""

import nltk

from jm_eval.retrieval.lexical import nltk_data_dir

if __name__ == "__main__":
    target = nltk_data_dir()
    target.mkdir(parents=True, exist_ok=True)
    for package in ("stopwords", "rslp"):
        if not nltk.download(package, download_dir=str(target), quiet=True):
            raise SystemExit(f"could not download NLTK package {package!r}")
    print(f"NLTK data in {target}")
