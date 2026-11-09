"""Phase 0: copy (never move) the SUSEP gold-standard files into data/private/.

The files stay on this machine (ground rule 1): data/private/ is git-ignored and
the pre-commit hook rejects it. Their hashes go to data/private/SHA256SUMS and
the run manifest in data/private/, not to the tracked data/MANIFEST.csv.
"""

from __future__ import annotations

import shutil
import sys

from jm_eval.io import load_config, resolve, sha256_file
from jm_eval.manifest import RunManifest


def main() -> int:
    paths = load_config("paths")
    src_dir = resolve(paths["paper_folder"])
    dst_dir = resolve(paths["data"]["private"])
    dst_dir.mkdir(parents=True, exist_ok=True)
    with RunManifest("01_stage_private", dst_dir, record_outputs=False) as run:
        sums = []
        for name in paths["susep_sources"]:
            src, dst = src_dir / name, dst_dir / name
            src_sha = sha256_file(src)
            run.add_input(src, sha256=src_sha)
            if dst.exists():
                if sha256_file(dst) != src_sha:
                    raise SystemExit(f"{dst} exists with different content; not overwriting")
            else:
                shutil.copy2(src, dst)
            if sha256_file(dst) != src_sha:
                raise SystemExit(f"hash mismatch after copy: {name}")
            run.add_output(dst)
            sums.append(f"{src_sha}  {name}")
            print(f"staged {name} ({dst.stat().st_size:,} bytes)")
        (dst_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
