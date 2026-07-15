"""Run benchmark spike against manifest samples."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from inference.registry import ModelRegistry
from inference.schemas.model_io import BpmInput, SeparateInput, TranscribeInput, VisionInput
from judge.judge import MusicTheoryJudge
from musicai_worker.fingering.optimizer import assign_fingering
from musicai_worker.fusion.scorer import FusionScorer


def ensure_synthetic_sample(path: Path, seconds: float = 2.0, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    sr = 44100
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    tone = 0.15 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), tone, sr)


async def run_benchmark(manifest_path: Path, report_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    registry = ModelRegistry.from_config()
    fusion = FusionScorer()
    judge = MusicTheoryJudge()

    results: list[dict] = []
    for sample in manifest["samples"]:
        sample_path = Path(sample["path"])
        if sample_path.suffix in {".wav", ".mp3"}:
            ensure_synthetic_sample(sample_path)

        started = time.perf_counter()
        notes_count = 0
        judge_snapped = 0
        vision_fallback = True

        if sample_path.suffix in {".wav", ".mp3"} and sample_path.exists():
            demucs = registry.get("demucs/htdemucs_6s")
            basic_pitch = registry.get("basic-pitch/v1")
            bpm_adapter = registry.get("librosa/beat")

            sep = await demucs.predict(SeparateInput(audio=sample_path, stem="guitar"))
            tx = await basic_pitch.predict(TranscribeInput(audio=sep.stem_path))
            bpm = await bpm_adapter.predict(BpmInput(audio=sample_path))

            tuples = fusion.raw_to_tab_notes(tx.notes)
            tab_notes = assign_fingering(tuples)
            judged = judge.judge(tab_notes, bpm=bpm.bpm)
            notes_count = len(judged.notes)
            judge_snapped = judged.stats.snapped_notes
        elif sample["type"] == "youtube":
            vision = registry.get("mediapipe/hands")
            out = await vision.predict(VisionInput(video=sample_path if sample_path.exists() else None))
            vision_fallback = out.fallback_audio_only

        elapsed = time.perf_counter() - started
        results.append(
            {
                "id": sample["id"],
                "notes": notes_count,
                "judge_snapped": judge_snapped,
                "vision_fallback": vision_fallback,
                "elapsed_sec": round(elapsed, 2),
            }
        )

    report = {
        "run_id": str(uuid.uuid4()),
        "results": results,
        "recommendation": "go" if all(r["elapsed_sec"] < manifest["metrics"]["pipeline_p95_sec_max"] for r in results) else "review",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/manifest.json")
    parser.add_argument("--report", default="benchmarks/reports/latest.json")
    args = parser.parse_args()
    report = asyncio.run(run_benchmark(Path(args.manifest), Path(args.report)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
