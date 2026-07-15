"""Fingering optimizer tests."""

from judge.judge import note_from_raw
from musicai_worker.fingering.optimizer import assign_fingering, midi_to_positions, optimize_sequence


def test_midi_to_positions_open_string():
    positions = midi_to_positions(64)  # E4 on high e string
    assert any(p.string == 1 and p.fret == 0 for p in positions)


def test_assign_fingering_returns_tab_notes():
    raw = [(64, 0.0, 250.0, 0.8), (67, 300.0, 250.0, 0.7)]
    notes = assign_fingering(raw)
    assert len(notes) == 2
    assert all(1 <= n.string <= 6 for n in notes)


def test_optimize_sequence_preserves_count():
    notes = [
        note_from_raw(64, 0, 200, 1, 0, 0.8),
        note_from_raw(67, 250, 200, 1, 3, 0.8),
        note_from_raw(69, 500, 200, 1, 5, 0.8),
    ]
    optimized = optimize_sequence(notes)
    assert len(optimized) == 3
