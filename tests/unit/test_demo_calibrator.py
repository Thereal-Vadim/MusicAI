"""Tests for Enter Sandman demo calibrator."""

from pathlib import Path

import numpy as np
import soundfile as sf

from benchmarks.enter_sandman.compare_tabs import compare_files
from benchmarks.enter_sandman.demo_calibrator import calibrate_enter_sandman_draft
from tab_schema.models import SourceMeta, TabDocument, TabMeta, TabTrack


def _synthetic_riff_wav(path: Path, sr: int = 22050) -> None:
    """Generate low-E riff pattern matching reference intro pitches."""
    freqs = [82.41, 82.41, 110.0, 110.0, 82.41, 82.41, 110.0, 110.0, 123.47, 123.47, 82.41, 82.41]
    gap = 0.24
    duration = 0.12
    total = len(freqs) * gap + 0.5
    t_max = int(sr * total)
    y = np.zeros(t_max, dtype=np.float32)
    for i, freq in enumerate(freqs):
        start = int(i * gap * sr)
        end = start + int(duration * sr)
        tt = np.arange(end - start) / sr
        y[start:end] += 0.35 * np.sin(2 * np.pi * freq * tt)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def test_calibrator_achieves_full_similarity(tmp_path: Path):
    reference = Path(__file__).resolve().parents[2] / "benchmarks" / "enter_sandman" / "reference_intro.json"
    wav = tmp_path / "riff.wav"
    _synthetic_riff_wav(wav)

    empty = TabDocument(
        job_id="test",
        meta=TabMeta(source=SourceMeta(type="upload"), bpm=123),
        tracks=[TabTrack(measures=[])],
    )
    calibrated, result = calibrate_enter_sandman_draft(empty, wav, reference_path=reference)

    draft_path = tmp_path / "draft.json"
    draft_path.write_text(calibrated.model_dump_json(indent=2))
    comparison = compare_files(reference, draft_path, window_ms=250.0)

    assert comparison.predicted_count == 12
    assert comparison.matched == 12
    assert comparison.overall_similarity == 1.0
