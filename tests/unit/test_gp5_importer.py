"""GP5 importer tests."""

import io
import json
from pathlib import Path

import guitarpro

from tab_export.gp5_exporter import document_to_gp5_bytes
from tab_export.gp5_importer import document_to_reference_json, gp5_to_document, gp5_to_reference_json
from tab_schema.alignment import align_documents
from tests.unit.test_tab_export import _intro_document


def test_gp5_importer_roundtrip_intro():
    source = _intro_document()
    imported = gp5_to_document(io.BytesIO(document_to_gp5_bytes(source)), max_measures=1)
    assert imported.meta.title == "Enter Sandman"
    assert imported.meta.artist == "Metallica"
    assert imported.meta.bpm == 123

    alignment = align_documents(source, imported, window_ms=250)
    assert alignment.pitch_f1 >= 0.95
    assert alignment.fret_match_rate >= 0.95


def test_gp5_to_reference_json_matches_intro(tmp_path: Path):
    source = _intro_document()
    gp5_path = tmp_path / "intro.gp5"
    gp5_path.write_bytes(document_to_gp5_bytes(source))

    payload = gp5_to_reference_json(gp5_path, max_measures=1)
    reference_path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "enter_sandman" / "reference_intro.json"
    )
    manual = json.loads(reference_path.read_text())

    assert payload["artist"] == "Metallica"
    assert len(payload["notes"]) == len(manual["notes"])
    assert payload["notes"][0]["fret"] == manual["notes"][0]["fret"]


def test_document_to_reference_json_flatten():
    doc = _intro_document()
    payload = document_to_reference_json(doc, source="test")
    assert payload["source"] == "test"
    assert len(payload["notes"]) == len(doc.all_notes())
