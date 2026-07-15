"""Judge config and playability tests."""

from judge.judge import JudgeConfig, MusicTheoryJudge, note_from_raw
from judge.rules import detect_key_with_fallback, validate_simultaneous_notes


def test_judge_config_from_yaml():
    cfg = JudgeConfig.from_yaml()
    assert cfg.snap_audio_confidence_threshold == 0.5
    assert cfg.max_simultaneous_notes == 4


def test_detect_key_with_fallback_c_major():
    pitch_classes = [0, 2, 4, 5, 7, 9, 11] * 2
    key = detect_key_with_fallback(pitch_classes, use_music21=True)
    assert key.root in {"C", "A", "G"}
    assert key.mode in {"major", "minor"}


def test_playability_flags_too_many_notes():
    judge = MusicTheoryJudge(JudgeConfig(max_simultaneous_notes=2, use_music21=False))
    notes = [
        note_from_raw(60, 0, 200, 6, 0, 0.9),
        note_from_raw(64, 0, 200, 5, 0, 0.9),
        note_from_raw(67, 0, 200, 4, 0, 0.9),
    ]
    result = judge.judge(notes, bpm=120)
    assert any("too_many_simultaneous_notes" in n.judge.flags for n in result.notes)


def test_judged_result_report():
    judge = MusicTheoryJudge()
    notes = [note_from_raw(60, 0, 200, 6, 0, 0.9)]
    result = judge.judge(notes, bpm=120)
    report = result.to_report()
    assert "key" in report
    assert "stats" in report


def test_validate_simultaneous_notes():
    assert validate_simultaneous_notes(4, max_notes=4) is True
    assert validate_simultaneous_notes(5, max_notes=4) is False
