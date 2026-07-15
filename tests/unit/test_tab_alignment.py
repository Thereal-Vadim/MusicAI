"""Tab alignment metrics tests."""

import json
from pathlib import Path

from tab_schema.alignment import align_files, align_tab_notes

REFERENCE = Path(__file__).resolve().parents[2] / "benchmarks" / "enter_sandman" / "reference_intro.json"


def test_perfect_match_f1():
    result = align_files(REFERENCE, REFERENCE)
    assert result.overall_f1 >= 0.99
    assert result.pitch_f1 >= 0.99
    assert result.timing_accuracy >= 0.99
    assert all(a.status == "match" for a in result.alignments if a.status != "extra_pred")


def test_timing_miss_detected(tmp_path):
    data = json.loads(REFERENCE.read_text())
    notes = data["notes"]
    shifted = []
    for note in notes:
        copy = dict(note)
        copy["start_ms"] = float(note["start_ms"]) + 120
        shifted.append(copy)

    predicted = tmp_path / "shifted.json"
    predicted.write_text(json.dumps({"notes": shifted}))
    result = align_files(REFERENCE, predicted, timing_tolerance_ms=80, normalize_timing=False)
    assert result.pitch_f1 >= 0.99
    assert result.timing_accuracy < 0.5


def test_empty_predicted():
    result = align_tab_notes(json.loads(REFERENCE.read_text())["notes"], [])
    assert result.overall_f1 == 0.0
    assert result.matched == 0
