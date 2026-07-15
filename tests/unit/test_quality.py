"""Quality metrics tests."""

from tab_schema.models import JudgeResult, NoteConfidence, TabNote
from tab_schema.quality import compute_quality_metrics, is_high_confidence_note


def _note(overall: float, snapped: bool = False, audio: float = 0.9) -> TabNote:
    return TabNote(
        id="n1",
        pitch="E4",
        start_ms=0,
        duration_ms=200,
        string=1,
        fret=0,
        confidence=NoteConfidence(audio=audio, overall=overall, judge=0.9),
        judge=JudgeResult(snapped=snapped, in_scale=True, in_chord=True),
    )


def test_high_confidence_allows_judge_correction():
    assert is_high_confidence_note(_note(0.96, snapped=True)) is True
    assert is_high_confidence_note(_note(0.96, snapped=False)) is True


def test_reference_mismatch_blocks_high_confidence():
    note = _note(0.96, snapped=False)
    note.flags.append("reference_mismatch")
    assert is_high_confidence_note(note) is False


def test_quality_metrics_counts():
    notes = [
        _note(0.96, snapped=False),
        _note(0.96, snapped=True),
        _note(0.55, snapped=False, audio=0.4),
    ]
    q = compute_quality_metrics(notes, key_confidence=0.85)
    assert q.notes_total == 3
    assert q.high_confidence_count == 2
    assert q.snapped_count == 1
    assert q.high_confidence_pct == round(2 / 3, 4)
