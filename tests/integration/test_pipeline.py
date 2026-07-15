"""Integration tests for pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicai_worker.pipeline import TranscriptionPipeline


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "sample.wav"
    sr = 44100
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 329.63 * t)  # E4-ish
    sf.write(str(audio_path), tone, sr)
    return audio_path


@pytest.mark.asyncio
async def test_pipeline_upload_end_to_end(sample_audio: Path, tmp_path: Path):
    pipeline = TranscriptionPipeline()
    work_dir = tmp_path / "job"
    document = await pipeline.run(
        job_id="test-job",
        work_dir=work_dir,
        source={"type": "upload", "path": str(sample_audio), "filename": "sample.wav"},
    )
    assert document.version == "1"
    assert document.tracks
    draft_path = work_dir / "draft.json"
    assert draft_path.exists()
    judge_report_path = work_dir / "judge_report.json"
    assert judge_report_path.exists()
    judge_report = json.loads(judge_report_path.read_text())
    assert "key" in judge_report
    assert "stats" in judge_report
    loaded = json.loads(draft_path.read_text())
    assert loaded["meta"]["bpm"] > 0
