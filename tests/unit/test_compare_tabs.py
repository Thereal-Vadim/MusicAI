"""Compare tabs unit tests."""

import json
from pathlib import Path

from benchmarks.enter_sandman.compare_tabs import compare_files

REFERENCE = Path(__file__).resolve().parents[1] / "enter_sandman" / "reference_intro.json"


def test_compare_perfect_match(tmp_path: Path):
    predicted = tmp_path / "draft.json"
    predicted.write_text(REFERENCE.read_text())
    result = compare_files(REFERENCE, predicted)
    assert result.overall_similarity >= 0.99
    assert result.pitch_f1 >= 0.99


def test_compare_empty_predicted(tmp_path: Path):
    predicted = tmp_path / "empty.json"
    predicted.write_text(json.dumps({"notes": []}))
    result = compare_files(REFERENCE, predicted)
    assert result.overall_similarity == 0.0
    assert result.matched == 0
