"""Reference vs Songsterr scoring tests."""

from pathlib import Path

from judge.judge import MusicTheoryJudge, note_from_raw
from tab_schema.reference import apply_reference_scoring, resolve_reference_profile

REFERENCE = Path(__file__).resolve().parents[2] / "benchmarks" / "enter_sandman" / "reference_intro.json"


def test_resolve_enter_sandman_profile():
    profile = resolve_reference_profile("Enter Sandman", "Metallica")
    assert profile is not None
    assert profile.path == REFERENCE


def test_reference_match_adds_bonus():
    profile = resolve_reference_profile("Enter Sandman", "Metallica")
    assert profile is not None
    note = note_from_raw(40, 0, 120, 6, 0, audio_confidence=0.9)
    judged = MusicTheoryJudge().judge([note], bpm=123).notes[0]
    before = judged.confidence.overall
    scored, summary = apply_reference_scoring([judged], profile)
    assert "reference_match" in scored[0].flags
    assert scored[0].confidence.overall > before
    assert summary["reference_matched"] == 1


def test_reference_mismatch_applies_penalty():
    profile = resolve_reference_profile("Enter Sandman", "Metallica")
    assert profile is not None
    note = note_from_raw(40, 0, 120, 6, 5, audio_confidence=0.9)  # wrong fret vs ref fret 0
    judged = MusicTheoryJudge().judge([note], bpm=123).notes[0]
    before = judged.confidence.overall
    scored, _ = apply_reference_scoring([judged], profile)
    assert "reference_mismatch" in scored[0].flags
    assert scored[0].confidence.overall < before
