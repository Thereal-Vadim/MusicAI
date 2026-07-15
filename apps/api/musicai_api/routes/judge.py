"""Judge API routes."""

from __future__ import annotations

from fastapi import APIRouter

from judge.judge import JudgeConfig

router = APIRouter(prefix="/v1/judge", tags=["judge"])


@router.get("/config")
async def judge_config() -> dict[str, object]:
    cfg = JudgeConfig.from_yaml()
    return {
        "snap_audio_confidence_threshold": cfg.snap_audio_confidence_threshold,
        "beats_per_measure": cfg.beats_per_measure,
        "min_note_duration_ms": cfg.min_note_duration_ms,
        "max_simultaneous_notes": cfg.max_simultaneous_notes,
        "max_chord_span": cfg.max_chord_span,
        "use_music21": cfg.use_music21,
    }
