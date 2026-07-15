"""Worker pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference.registry import ModelRegistry
from inference.schemas.model_io import BpmInput, SeparateInput, TranscribeInput, VisionInput
from judge.judge import JudgeConfig, MusicTheoryJudge
from musicai_worker.fingering.optimizer import assign_fingering, optimize_sequence
from musicai_worker.fusion.scorer import FusionScorer
from tab_schema.models import SourceMeta, TabDocument, TabMeasure, TabMeta, TabTrack

PIPELINE_STAGES = [
    "queued",
    "ingest",
    "separate",
    "transcribe",
    "vision",
    "fusion",
    "judge",
    "draft",
    "done",
    "failed",
]


class TranscriptionPipeline:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry.from_config()
        self.fusion = FusionScorer()
        self.judge = MusicTheoryJudge(JudgeConfig())

    async def run(
        self,
        job_id: str,
        work_dir: Path,
        source: dict[str, Any],
        tuning: list[str] | None = None,
        progress_callback: Any = None,
    ) -> TabDocument:
        async def report(stage: str, detail: str = "") -> None:
            if progress_callback:
                await progress_callback(stage, detail)

        work_dir.mkdir(parents=True, exist_ok=True)
        tuning = tuning or ["E2", "A2", "D3", "G3", "B3", "E4"]

        await report("ingest", "Normalizing media")
        audio_path = work_dir / "input.wav"
        video_path: Path | None = None
        source_meta = SourceMeta(type=source["type"])

        if source["type"] == "upload":
            from musicai_worker.ingest import normalize_audio

            normalize_audio(Path(source["path"]), audio_path)
            source_meta.filename = source.get("filename")
        else:
            from musicai_worker.ingest import download_youtube

            audio_path, video_path, youtube_id = download_youtube(source["url"], work_dir)
            source_meta.url = source["url"]
            source_meta.youtube_id = youtube_id
            if video_path and video_path.exists():
                pass
            else:
                video_path = work_dir / "video.mp4" if (work_dir / "video.mp4").exists() else None

        await report("separate", "Running Demucs guitar stem separation")
        demucs = self.registry.get("demucs/htdemucs_6s")
        separate_result = await demucs.predict(
            SeparateInput(audio=audio_path, stem="guitar", output_dir=work_dir / "stems")
        )
        stem_path = separate_result.stem_path

        await report("transcribe", "Running Basic Pitch transcription")
        basic_pitch = self.registry.get("basic-pitch/v1")
        bpm_adapter = self.registry.get("librosa/beat")
        bpm_result = await bpm_adapter.predict(BpmInput(audio=audio_path))
        transcribe_result = await basic_pitch.predict(TranscribeInput(audio=stem_path))

        raw_tuples = self.fusion.raw_to_tab_notes(transcribe_result.notes)
        tab_notes = assign_fingering(raw_tuples)
        tab_notes = optimize_sequence(tab_notes)

        await report("vision", "Analyzing hand position on fretboard")
        mediapipe = self.registry.get("mediapipe/hands")
        vision_result = await mediapipe.predict(
            VisionInput(video=video_path, audio=audio_path)
        )

        await report("fusion", "Merging audio and vision signals")
        fused_notes = self.fusion.fuse_notes(
            tab_notes,
            vision_result.frames,
            audio_only=vision_result.fallback_audio_only,
        )

        await report("judge", "Running music theory validation")
        judged = self.judge.judge(fused_notes, bpm=bpm_result.bpm)

        await report("draft", "Building TabDocument")
        document = self._build_document(
            job_id=job_id,
            source_meta=source_meta,
            bpm=bpm_result.bpm,
            key=judged.key.root,
            mode=judged.key.mode,
            tuning=tuning,
            notes=judged.notes,
        )

        draft_path = work_dir / "draft.json"
        draft_path.write_text(document.model_dump_json(indent=2))
        await report("done", str(draft_path))
        return document

    def _build_document(
        self,
        job_id: str,
        source_meta: SourceMeta,
        bpm: float,
        key: str,
        mode: str,
        tuning: list[str],
        notes: list,
    ) -> TabDocument:
        ms_per_measure = (60_000 / bpm) * 4
        measures_map: dict[int, TabMeasure] = {}

        for note in notes:
            idx = int(note.start_ms // ms_per_measure)
            if idx not in measures_map:
                measures_map[idx] = TabMeasure(
                    index=idx,
                    start_ms=idx * ms_per_measure,
                    confidence=note.confidence.overall,
                )
            measures_map[idx].notes.append(note)

        measures = [measures_map[i] for i in sorted(measures_map.keys())]
        overall = sum(n.confidence.overall for n in notes) / len(notes) if notes else 0.0

        return TabDocument(
            job_id=job_id,
            meta=TabMeta(
                bpm=bpm,
                key=key,
                mode=mode,
                tuning=tuning,
                source=source_meta,
                overall_confidence=overall,
            ),
            tracks=[TabTrack(measures=measures)],
        )

    @staticmethod
    def load_draft(path: Path) -> TabDocument:
        return TabDocument.model_validate_json(path.read_text())

    @staticmethod
    def save_draft(path: Path, document: TabDocument) -> None:
        path.write_text(document.model_dump_json(indent=2))
