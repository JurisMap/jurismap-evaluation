"""Check the data archive against data/MANIFEST.csv (sha256 and size of every file).

Run from the repository root after extracting the data archive there:
    python scripts/verify_data.py            # every file listed must be present and match
    python scripts/verify_data.py --partial  # only check the files that are present
Exits 1 if a file is missing (without --partial) or differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partial", action="store_true", help="skip files that are not present")
    args = ap.parse_args()
    with open(ROOT / "data" / "MANIFEST.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing, differ, ok = [], [], 0
    for row in rows:
        p = ROOT / row["path"]
        if not p.exists():
            missing.append(row["path"])
        elif p.stat().st_size != int(row["bytes"]) or sha256(p) != row["sha256"]:
            differ.append(row["path"])
        else:
            ok += 1
    print(f"{ok} of {len(rows)} files match; {len(missing)} missing; {len(differ)} differ")
    for p in differ:
        print(f"  differs: {p}")
    if missing and not args.partial:
        for p in missing[:20]:
            print(f"  missing: {p}")
    sys.exit(1 if differ or (missing and not args.partial) else 0)


if __name__ == "__main__":
    main()
