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
from musicai_worker.logging_setup import get_logger, log_artifact, setup_logging, stage_timer
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
        self.judge = MusicTheoryJudge(JudgeConfig.from_yaml())

    async def run(
        self,
        job_id: str,
        work_dir: Path,
        source: dict[str, Any],
        tuning: list[str] | None = None,
        progress_callback: Any = None,
    ) -> TabDocument:
        log_dir = work_dir / "logs"
        setup_logging(job_id=job_id, log_dir=log_dir)
        log = get_logger("pipeline", job_id=job_id, stage="pipeline")

        log.info("Pipeline start work_dir=%s source_type=%s", work_dir, source.get("type"))
        log.info("Registry models=%s", self.registry.list_models())
        log.info("Registry health=%s", self.registry.healthcheck_all())
        log.info("Registry config=%s", self.registry.runtime_config())
        log_artifact(log_dir, "registry_config.json", self.registry.runtime_config())

        async def report(stage: str, detail: str = "") -> None:
            stage_log = get_logger("pipeline", job_id=job_id, stage=stage)
            stage_log.info("progress detail=%s", detail or stage)
            if progress_callback:
                await progress_callback(stage, detail)

        work_dir.mkdir(parents=True, exist_ok=True)
        tuning = tuning or ["E2", "A2", "D3", "G3", "B3", "E4"]
        log.info("Tuning=%s", tuning)

        await report("ingest", "Normalizing media")
        audio_path = work_dir / "input.wav"
        video_path: Path | None = None
        source_meta = SourceMeta(type=source["type"])

        with stage_timer(get_logger("pipeline", job_id=job_id, stage="ingest"), "ingest"):
            if source["type"] == "upload":
                from musicai_worker.ingest import normalize_audio

                normalize_audio(Path(source["path"]), audio_path)
                source_meta.filename = source.get("filename")
                log.info("Upload normalized path=%s", audio_path)
            else:
                from musicai_worker.ingest import download_youtube

                audio_path, video_path, youtube_id = download_youtube(
                source["url"], work_dir, allow_placeholder=False
            )
                source_meta.url = source["url"]
                source_meta.youtube_id = youtube_id
                log.info(
                    "YouTube ingested youtube_id=%s audio=%s video=%s",
                    youtube_id,
                    audio_path,
                    video_path,
                )
                if not (video_path and video_path.exists()):
                    video_path = work_dir / "video.mp4" if (work_dir / "video.mp4").exists() else None

        await report("separate", "Running Demucs guitar stem separation")
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="separate"), "separate"):
            demucs = self.registry.get("demucs/htdemucs_6s")
            log.info("Demucs adapter=%s", demucs.describe())
            separate_result = await demucs.predict(
                SeparateInput(audio=audio_path, stem="guitar", output_dir=work_dir / "stems")
            )
            stem_path = separate_result.stem_path
            log.info("Guitar stem path=%s model_id=%s", stem_path, separate_result.model_id)
            log_artifact(log_dir, "separate_output.json", separate_result.model_dump(mode="json"))

        await report("transcribe", "Running Basic Pitch transcription")
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="transcribe"), "transcribe"):
            basic_pitch = self.registry.get("basic-pitch/v1")
            bpm_adapter = self.registry.get("librosa/beat")
            log.info("BasicPitch adapter=%s", basic_pitch.describe())
            bpm_result = await bpm_adapter.predict(BpmInput(audio=audio_path))
            log.info("BPM=%.2f beats=%d model=%s", bpm_result.bpm, len(bpm_result.beat_times_sec), bpm_result.model_id)
            transcribe_result = await basic_pitch.predict(TranscribeInput(audio=stem_path))
            log.info(
                "Transcribed raw_notes=%d model=%s",
                len(transcribe_result.notes),
                transcribe_result.model_id,
            )
            log_artifact(
                log_dir,
                "transcribe_notes.json",
                [n.model_dump() for n in transcribe_result.notes[:200]],
            )

            raw_tuples = self.fusion.raw_to_tab_notes(transcribe_result.notes)
            tab_notes = assign_fingering(raw_tuples)
            tab_notes = optimize_sequence(tab_notes)
            log.info("After fingering notes=%d", len(tab_notes))

        await report("vision", "Analyzing hand position on fretboard")
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="vision"), "vision"):
            mediapipe = self.registry.get("mediapipe/hands")
            log.info("MediaPipe adapter=%s", mediapipe.describe())
            vision_result = await mediapipe.predict(
                VisionInput(video=video_path, audio=audio_path)
            )
            log.info(
                "Vision frames=%d fallback_audio_only=%s model=%s",
                len(vision_result.frames),
                vision_result.fallback_audio_only,
                vision_result.model_id,
            )
            log_artifact(
                log_dir,
                "vision_frames.json",
                [f.model_dump() for f in vision_result.frames[:50]],
            )

        await report("fusion", "Merging audio and vision signals")
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="fusion"), "fusion"):
            fused_notes = self.fusion.fuse_notes(
                tab_notes,
                vision_result.frames,
                audio_only=vision_result.fallback_audio_only,
            )
            conflict_count = sum(1 for n in fused_notes if "audio_vision_mismatch" in n.flags)
            log.info("Fusion notes=%d conflicts=%d", len(fused_notes), conflict_count)

        await report("judge", "Running music theory validation")
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="judge"), "judge"):
            judged = self.judge.judge(fused_notes, bpm=bpm_result.bpm)
            judge_report_path = work_dir / "judge_report.json"
            judge_report_path.write_text(json.dumps(judged.to_report(), indent=2))
            log.info(
                "Judge key=%s %s snapped=%d flagged=%d chromatic=%d",
                judged.key.root,
                judged.key.mode,
                judged.stats.snapped_notes,
                judged.stats.flagged_notes,
                judged.stats.chromatic_notes,
            )

        await report("draft", "Building TabDocument")
        document = self._build_document(
            job_id=job_id,
            source_meta=source_meta,
            bpm=bpm_result.bpm,
            key=judged.key.root,
            mode=judged.key.mode,
            tuning=tuning,
            notes=judged.notes,
            chords=judged.chords,
        )

        draft_path = work_dir / "draft.json"
        draft_path.write_text(document.model_dump_json(indent=2))
        log.info(
            "Draft saved path=%s measures=%d overall_confidence=%.3f",
            draft_path,
            sum(len(t.measures) for t in document.tracks),
            document.meta.overall_confidence,
        )
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
        chords: list | None = None,
    ) -> TabDocument:
        ms_per_measure = (60_000 / bpm) * 4 if bpm > 0 else 2000.0
        measures_map: dict[int, TabMeasure] = {}
        chord_by_measure = {c.measure_index: c.symbol for c in (chords or [])}

        for note in notes:
            idx = int(note.start_ms // ms_per_measure)
            if idx not in measures_map:
                measure_confidence = note.confidence.overall
                measures_map[idx] = TabMeasure(
                    index=idx,
                    start_ms=idx * ms_per_measure,
                    confidence=measure_confidence,
                    chord=chord_by_measure.get(idx),
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
