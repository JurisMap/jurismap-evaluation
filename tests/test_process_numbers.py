import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export", ROOT / "scripts/00_export_corpus.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def pf(title, rel=""):
    return export.process_fields({"acordao_titulo": title, "acordao_relatorio": rel, "acordao_voto_relator": ""})


def test_standard_and_space_separated():
    assert pf("Processo 15414.604792/2016-77")["process_number_digits"] == "15414604792201677"
    assert pf("Processo  15414.604258  2016-61")["process_number"] == "15414.604258/2016-61"


def test_apensos_primary_first():
    f = pf("Processo 15414.629444/2019-55 - Apenso Processo Susep 15414.629448/2019-32")
    assert f["process_number"] == "15414.629444/2019-55"
    assert f["process_numbers_in_title"] == ["15414629444201955", "15414629448201932"]


def test_recurso_and_fallback_to_relatorio():
    f = pf("Recurso 7142", rel="... processo nº 15414.100720/2012-21 ...")
    assert f["recurso_number"] == "7142"
    assert f["process_number_source"] == "acordao_relatorio"
    assert pf("Recurso 7142")["process_number"] is None


def test_invalid_formats_flagged():
    assert pf("Processo 15414.000494/98-24")["process_number_valid_format"] is False
    assert pf("Processo 15414.604792/2016-77")["process_number_valid_format"] is True
