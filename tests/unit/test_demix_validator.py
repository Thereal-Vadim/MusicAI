from __future__ import annotations

import numpy as np

from musicai_worker.demix_validator import validate_guitar_demix


def test_validate_demix_passes_independent_voices() -> None:
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    solo = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rhythm = (0.5 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    report = validate_guitar_demix(solo, rhythm, sr, target_part="rhythm")
    assert report.leakage_score < 0.5
    assert report.solo_playability_score > 0
    assert report.rhythm_playability_score > 0


def test_validate_demix_detects_high_leakage() -> None:
    sr = 22050
    y = np.random.default_rng(0).normal(0, 0.1, sr).astype(np.float32)
    report = validate_guitar_demix(y, y.copy(), sr, target_part="solo")
    assert report.leakage_score >= 0.99
