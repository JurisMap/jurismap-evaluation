"""Pre-commit hook: refuse to commit private data (SUSEP files, expert answers).

Blocks any path under data/private/ and any file whose name matches the SUSEP
gold-standard files (similarity_experts*, experts_evaluations*, data.csv),
wherever it is in the tree. Ground rule 1 of the implementation plan.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import PurePosixPath

BLOCKED_DIRS = ("data/private",)
BLOCKED_NAMES = ("similarity_experts*", "experts_evaluations*", "data.csv",
                 "*gemini-key*.json",  # service-account keys
                 # expert questionnaire and B3 files holding the experts' answers (identifiable)
                 "avaliação do jurismap*", "avaliacao do jurismap*", "b3_coding_*", "b3_disagreements*",
                 "b3_review*", "b3_llm_*", "b3_check_*", "b3_translations*", "b3_quotes_selection*")


def is_blocked(path: str) -> bool:
    p = PurePosixPath(path.replace("\\", "/"))
    posix = p.as_posix().lstrip("./")
    if any(posix == d or posix.startswith(d + "/") for d in BLOCKED_DIRS):
        return True
    name = p.name.lower()
    return any(fnmatch.fnmatch(name, pat.lower()) for pat in BLOCKED_NAMES)


def main(argv: list[str]) -> int:
    blocked = [p for p in argv if is_blocked(p)]
    for p in blocked:
        print(f"BLOCKED (private data, ground rule 1): {p}", file=sys.stderr)
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
