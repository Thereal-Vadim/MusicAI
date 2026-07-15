"""Music Theory Judge tests."""

from judge.judge import JudgeConfig, MusicTheoryJudge, note_from_raw
from judge.rules import detect_key, scale_pitch_classes, snap_midi_to_pitch_classes


def test_detect_key_c_major():
    pitch_classes = [0, 2, 4, 5, 7, 9, 11] * 2
    key = detect_key(pitch_classes)
    assert key.root == "C"
    assert key.mode == "major"


def test_chromatic_note_snapped_gets_confidence_bonus():
    judge = MusicTheoryJudge()
    notes = [
        note_from_raw(60, 0, 200, 2, 0, audio_confidence=0.9),
        note_from_raw(64, 250, 200, 2, 0, audio_confidence=0.9),
        note_from_raw(67, 500, 200, 1, 0, audio_confidence=0.9),
        note_from_raw(66, 750, 200, 2, 1, audio_confidence=0.9),  # F# in C major
    ]
    result = judge.judge(notes, bpm=120)
    chromatic = next(n for n in result.notes if n.start_ms == 750)
    assert chromatic.judge.snapped is True
    assert chromatic.confidence.overall >= 0.85


def test_snap_on_low_confidence_out_of_scale():
    judge = MusicTheoryJudge()
    notes = [
        note_from_raw(60, 0, 200, 2, 0, audio_confidence=0.9),
        note_from_raw(64, 250, 200, 2, 0, audio_confidence=0.9),
        note_from_raw(67, 500, 200, 1, 0, audio_confidence=0.9),
        note_from_raw(66, 750, 200, 2, 1, audio_confidence=0.2),  # F#4
    ]
    result = judge.judge(notes, bpm=120)
    note = next(n for n in result.notes if n.start_ms == 750)
    assert note.judge.snapped is True
    assert note.original_pitch is not None
    assert note.confidence.judge >= 0.88


def test_in_scale_notes_with_low_audio_can_reach_high_confidence_after_judge():
    judge = MusicTheoryJudge()
    notes = [
        note_from_raw(60, 0, 200, 2, 0, audio_confidence=0.42),
        note_from_raw(64, 250, 200, 2, 0, audio_confidence=0.38),
        note_from_raw(67, 500, 200, 1, 0, audio_confidence=0.41),
    ]
    result = judge.judge(notes, bpm=120)
    assert all(n.confidence.overall >= 0.55 for n in result.notes)


def test_trusted_notes_can_reach_high_confidence():
    judge = MusicTheoryJudge()
    notes = [
        note_from_raw(60, 0, 200, 2, 0, audio_confidence=0.95, vision_confidence=0.9),
        note_from_raw(64, 250, 200, 2, 0, audio_confidence=0.93, vision_confidence=0.88),
        note_from_raw(67, 500, 200, 1, 0, audio_confidence=0.96, vision_confidence=0.91),
    ]
    result = judge.judge(notes, bpm=120)
    assert all(not n.judge.snapped for n in result.notes)
    assert result.notes[2].confidence.overall >= 0.95
    assert result.notes[0].confidence.overall >= 0.94


def test_snap_midi_to_nearest_scale_tone():
    c_major = scale_pitch_classes("C", "major")
    snapped = snap_midi_to_pitch_classes(61, c_major)
    assert snapped in {60, 62}
