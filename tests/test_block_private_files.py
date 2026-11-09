import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hook", ROOT / "scripts/hooks/block_private_files.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def test_blocks_private_dir_and_susep_names():
    for p in ["data/private/anything.txt", "data/private/sub/x.parquet",
              "similarity_experts_evaluation.xlsm", "results/experts_evaluations_mean.xlsx",
              "data.csv", "some/dir/DATA.CSV", "tests/fixtures/similarity_experts_copy.csv"]:
        assert hook.is_blocked(p), p
    assert hook.main(["README.md", "data/private/x"]) == 1


def test_allows_normal_files():
    for p in ["README.md", "results/a1_retrieval_main.csv", "data/MANIFEST.csv",
              "src/jm_eval/io.py", "data_private_notes.md", "metadata.csv"]:
        assert not hook.is_blocked(p), p
    assert hook.main(["README.md"]) == 0
