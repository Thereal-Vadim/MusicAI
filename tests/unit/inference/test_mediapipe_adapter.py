"""MediaPipe Tasks API adapter tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.adapters.mediapipe_adapter import MediaPipeAdapter, hand_frame_from_result
from inference.schemas.model_io import VisionInput


def test_healthcheck_requires_tasks_api():
    adapter = MediaPipeAdapter()
    assert adapter.healthcheck() in {True, False}


def test_hand_frame_from_empty_result():
    frame = hand_frame_from_result(SimpleNamespace(hand_landmarks=[]), 100.0)
    assert frame.timestamp_ms == 100.0
    assert frame.landmarks == []
    assert frame.fret_zone is None
    assert frame.visibility_score == 0.0


def test_hand_frame_from_landmarks():
    hand = [SimpleNamespace(x=0.1, y=0.2, z=0.0, visibility=0.8) for _ in range(21)]
    hand[8] = SimpleNamespace(x=0.6, y=0.3, z=0.0, visibility=0.9)
    result = SimpleNamespace(hand_landmarks=[hand])
    frame = hand_frame_from_result(result, 250.0)
    assert len(frame.landmarks) == 21
    assert frame.fret_zone is not None
    assert frame.visibility_score == 0.8


@pytest.mark.asyncio
async def test_predict_missing_video(tmp_path: Path):
    adapter = MediaPipeAdapter()
    out = await adapter.predict(VisionInput(video=tmp_path / "missing.mp4"))
    assert out.fallback_audio_only is True
    assert out.frames == []


def test_resolve_model_uses_explicit_path(tmp_path: Path):
    from inference.adapters.mediapipe_model import resolve_hand_landmarker_model

    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"fake")
    assert resolve_hand_landmarker_model(model) == model
