import json

from jm_eval.manifest import RunManifest


def test_manifest_contents(tmp_path):
    out = tmp_path / "x.txt"
    with RunManifest("t", tmp_path, models={"m": "x"}, record_outputs=False) as run:
        out.write_text("hi")
        run.add_output(out)
    m = json.loads(run.path.read_text())
    for key in ["git", "models", "config_hash", "seeds", "started_utc", "ended_utc", "packages", "outputs"]:
        assert key in m
    assert m["status"] == "ok"
    assert m["seeds"]["default"] == 20260926
    assert m["git"]["eval"]["sha"] is None or len(m["git"]["eval"]["sha"]) == 40
    assert any(p.startswith("pandas==") for p in m["packages"])


def test_manifest_written_on_failure_and_not_overwritten(tmp_path):
    try:
        with RunManifest("t", tmp_path, record_outputs=False):
            raise ValueError("boom")
    except ValueError:
        pass
    with RunManifest("t", tmp_path, record_outputs=False):
        pass
    files = sorted(p.name for p in tmp_path.glob("t*.manifest.json"))
    assert files == ["t.manifest.json", "t_2.manifest.json"]
    assert json.loads((tmp_path / "t.manifest.json").read_text())["status"] == "failed"
