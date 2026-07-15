"""Unit tests for AST label → guitar timbre mapping."""

from inference.adapters.audio_classifier_adapter import (
    MIDI_ACOUSTIC_STEEL,
    MIDI_DISTORTION,
    MIDI_ELECTRIC_CLEAN,
    MIDI_OVERDRIVE,
    map_audioset_labels_to_timbre,
)


def test_map_acoustic_guitar():
    timbre, label, conf = map_audioset_labels_to_timbre(
        [("Acoustic guitar", 0.42), ("Music", 0.3)]
    )
    assert timbre.midi_program == MIDI_ACOUSTIC_STEEL
    assert timbre.type == "Acoustic Guitar"
    assert "acoustic" in label.lower()
    assert conf == 0.42


def test_map_distortion_over_generic_music():
    timbre, label, conf = map_audioset_labels_to_timbre(
        [("Music", 0.55), ("Distortion", 0.31), ("Electric guitar", 0.2)]
    )
    assert timbre.midi_program == MIDI_DISTORTION
    assert "distortion" in label.lower()
    assert conf == 0.31


def test_map_overdrive_electric():
    timbre, _, _ = map_audioset_labels_to_timbre(
        [("Electric guitar", 0.4), ("Rock music", 0.2)]
    )
    assert timbre.midi_program == MIDI_OVERDRIVE


def test_map_fallback_clean_when_no_guitar():
    timbre, label, conf = map_audioset_labels_to_timbre(
        [("Speech", 0.9), ("Silence", 0.05)]
    )
    assert timbre.midi_program == MIDI_ELECTRIC_CLEAN
    assert timbre.type == "Electric Guitar (clean)"
    assert conf == 0.0 or label in {"Speech", "fallback", "Silence"}


def test_map_empty_labels_fallback():
    timbre, label, conf = map_audioset_labels_to_timbre([])
    assert timbre.midi_program == MIDI_ELECTRIC_CLEAN
    assert label == "fallback"
    assert conf == 0.0
