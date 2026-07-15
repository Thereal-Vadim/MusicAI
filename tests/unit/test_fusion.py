"""Fusion scorer tests."""

from inference.schemas.model_io import HandFrame
from judge.judge import note_from_raw
from musicai_worker.fusion.scorer import FusionScorer


def test_fusion_flags_audio_vision_mismatch():
    scorer = FusionScorer()
    notes = [note_from_raw(64, 0, 200, 1, 0, 0.8)]
    frames = [HandFrame(timestamp_ms=0, fret_zone=7, visibility_score=0.9)]
    fused = scorer.fuse_notes(notes, frames, audio_only=False)
    assert "audio_vision_mismatch" in fused[0].flags


def test_fusion_audio_only_fallback():
    scorer = FusionScorer()
    notes = [note_from_raw(64, 0, 200, 1, 0, 0.8)]
    fused = scorer.fuse_notes(notes, [], audio_only=True)
    assert fused[0].confidence.overall == fused[0].confidence.audio
