"""Music Theory Judge tests."""

from judge.judge import JudgeConfig, MusicTheoryJudge, note_from_raw
from judge.rules import detect_key, scale_pitch_classes, snap_midi_to_scale


def test_detect_key_c_major():
    pitch_classes = [0, 2, 4, 5, 7, 9, 11] * 2
    key = detect_key(pitch_classes)
    assert key.root == "C"
    assert key.mode == "major"


def test_chromatic_note_flagged_not_deleted():
    judge = MusicTheoryJudge(JudgeConfig(snap_audio_confidence_threshold=0.0))
    notes = [
        note_from_raw(60, 0, 200, 2, 0, audio_confidence=0.9),
        note_from_raw(64, 250, 200, 2, 0, audio_confidence=0.9),
        note_from_raw(67, 500, 200, 1, 0, audio_confidence=0.9),
        note_from_raw(66, 750, 200, 2, 1, audio_confidence=0.9),  # F# in C major
    ]
    result = judge.judge(notes, bpm=120)
    chromatic = next(n for n in result.notes if n.start_ms == 750)
    assert "chromatic_note" in chromatic.judge.flags


def test_snap_on_low_confidence_out_of_scale():
    judge = MusicTheoryJudge(JudgeConfig(snap_audio_confidence_threshold=0.9))
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


def test_snap_midi_to_nearest_scale_tone():
    c_major = scale_pitch_classes("C", "major")
    snapped = snap_midi_to_scale(61, c_major)
    assert snapped in {60, 62}
