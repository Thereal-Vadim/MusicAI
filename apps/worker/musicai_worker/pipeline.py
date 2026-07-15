"""Worker pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference.pipeline_routing import load_pipeline_routing
from inference.registry import ModelRegistry
from inference.schemas.model_io import BpmInput, SeparateInput, TranscribeInput, VisionInput
from judge.judge import JudgeConfig, MusicTheoryJudge
from musicai_worker.demix_validator import validate_guitar_demix
from musicai_worker.fingering.aco_optimizer import ACOConfig, optimize_sequence_aco
from musicai_worker.fingering.optimizer import assign_fingering, optimize_sequence
from musicai_worker.fusion.scorer import FusionScorer
from musicai_worker.guitar_isolation import GUITAR_PART_LABELS, GuitarPart, isolate_guitar_part
from musicai_worker.logging_setup import get_logger, log_artifact, setup_logging, stage_timer
from musicai_worker.stage_runners import log_routing_summary, run_coarse_separation, run_guitar_demix, select_fingering_optimizer
from tab_schema.models import SourceMeta, TabDocument, TabMeasure, TabMeta, TabTrack
from tab_schema.quality import compute_quality_metrics
from tab_schema.reference import (
    apply_reference_scoring,
    merge_reference_into_quality,
    resolve_reference_profile,
)

PIPELINE_STAGES = [
    "queued",
    "ingest",
    "separate",
    "guitar_demix",
    "demix_validate",
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
        self.routing = load_pipeline_routing()
        self.fusion = FusionScorer()
        self.judge = MusicTheoryJudge(JudgeConfig.from_yaml())

    async def run(
        self,
        job_id: str,
        work_dir: Path,
        source: dict[str, Any],
        tuning: list[str] | None = None,
        guitar_part: GuitarPart = "combined",
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
        log_artifact(log_dir, "pipeline_routing.json", log_routing_summary(self.routing))

        async def report(
            stage: str,
            detail: str = "",
            *,
            sub_progress: float = 0.0,
            finished: bool = False,
            stage_duration_sec: float | None = None,
        ) -> None:
            stage_log = get_logger("pipeline", job_id=job_id, stage=stage)
            stage_log.info("progress detail=%s", detail or stage)
            if progress_callback:
                await progress_callback(
                    stage,
                    detail,
                    sub_progress=sub_progress,
                    finished=finished,
                    stage_duration_sec=stage_duration_sec,
                )

        work_dir.mkdir(parents=True, exist_ok=True)
        tuning = tuning or ["E2", "A2", "D3", "G3", "B3", "E4"]
        log.info("Tuning=%s guitar_part=%s", tuning, guitar_part)

        await report("ingest", "Normalizing media", sub_progress=0.0)
        audio_path = work_dir / "input.wav"
        video_path: Path | None = None
        source_meta = SourceMeta(type=source["type"])

        with stage_timer(get_logger("pipeline", job_id=job_id, stage="ingest"), "ingest") as ingest_metrics:
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
        await report(
            "ingest",
            "Media ready",
            sub_progress=1.0,
            stage_duration_sec=ingest_metrics.get("elapsed_sec"),
        )

        part_label = GUITAR_PART_LABELS.get(guitar_part, "Guitar")
        await report("separate", "Coarse separation: vocals / bass / drums / guitar", sub_progress=0.0)
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="separate"), "separate") as separate_metrics:
            coarse = await run_coarse_separation(
                self.registry,
                self.routing.coarse_separation,
                SeparateInput(
                    audio=audio_path,
                    stem="guitar",
                    guitar_part=guitar_part,
                    mode="multi_stem",
                    output_dir=work_dir / "stems",
                ),
            )
            separate_result = coarse.output
            guitar_raw = separate_result.stem_path
            log.info(
                "Coarse separation backend=%s method=%s stems=%s guitar=%s",
                coarse.backend_id,
                separate_result.isolation_method,
                list(separate_result.coarse_stems.keys()),
                guitar_raw,
            )
            log_artifact(
                log_dir,
                "coarse_stems.json",
                {k: str(v) for k, v in separate_result.coarse_stems.items()},
            )
        await report(
            "separate",
            f"Coarse stems ready ({len(separate_result.coarse_stems)} tracks)",
            sub_progress=1.0,
            stage_duration_sec=separate_metrics.get("elapsed_sec"),
        )

        demix_run = None
        validation_report = None
        stem_path = guitar_raw
        isolation_method = separate_result.isolation_method or "coarse_multi"

        if guitar_part != "combined":
            await report("guitar_demix", "Guitar de-mix (Wave-U-Net → CASA fallback)", sub_progress=0.0)
            with stage_timer(
                get_logger("pipeline", job_id=job_id, stage="guitar_demix"), "guitar_demix"
            ) as demix_metrics:
                demix_run = await run_guitar_demix(
                    self.registry,
                    self.routing.guitar_demix,
                    guitar_stem=guitar_raw,
                    output_dir=work_dir / "stems" / "demix",
                    mix_path=audio_path,
                )
                log_artifact(
                    log_dir,
                    "guitar_demix.json",
                    {**demix_run.diagnostics, "backend_id": demix_run.backend_id, "method": demix_run.method},
                )
            await report(
                "guitar_demix",
                f"Demix solo/rhythm ({demix_run.method})",
                sub_progress=1.0,
                stage_duration_sec=demix_metrics.get("elapsed_sec"),
            )

            await report("demix_validate", "Playability validation (zero-cost)", sub_progress=0.0)
            with stage_timer(
                get_logger("pipeline", job_id=job_id, stage="demix_validate"), "demix_validate"
            ) as validate_metrics:
                import librosa

                solo_y, sr = librosa.load(str(demix_run.solo), sr=44100, mono=True)
                rhythm_y, _ = librosa.load(str(demix_run.rhythm), sr=44100, mono=True)
                validation_report = validate_guitar_demix(
                    solo_y,
                    rhythm_y,
                    sr,
                    target_part=guitar_part,
                )

                if validation_report.passed:
                    stem_path = demix_run.solo if guitar_part == "solo" else demix_run.rhythm
                    isolation_method = f"{coarse.backend_id}+{demix_run.backend_id}+validated"
                else:
                    log.warning(
                        "Demix validation failed leakage=%.2f merged_poly=%d — HPSS fallback",
                        validation_report.leakage_score,
                        validation_report.merged_max_polyphony,
                    )
                    stem_path = isolate_guitar_part(
                        guitar_raw,
                        guitar_part,
                        work_dir / "stems" / "parts",
                        mix_path=audio_path,
                    )
                    isolation_method = f"{coarse.backend_id}+{demix_run.backend_id}+fallback_hpss"
                    from dataclasses import replace

                    validation_report = replace(validation_report, used_fallback=True)

                log_artifact(log_dir, "demix_validation.json", validation_report.to_dict())
            await report(
                "demix_validate",
                f"Validation {'OK' if validation_report.passed else 'fallback'} · {part_label}",
                sub_progress=1.0,
                stage_duration_sec=validate_metrics.get("elapsed_sec"),
            )
        else:
            stem_path = guitar_raw

        log.info(
            "Transcription stem=%s method=%s part=%s",
            stem_path,
            isolation_method,
            guitar_part,
        )
        log_artifact(
            log_dir,
            "separate_output.json",
            {
                **separate_result.model_dump(mode="json"),
                "guitar_part": guitar_part,
                "transcription_stem": str(stem_path),
                "isolation_method": isolation_method,
                "demix_validation": validation_report.to_dict() if validation_report else None,
            },
        )
        from musicai_worker.stem_manifest import write_stems_manifest

        write_stems_manifest(
            work_dir,
            guitar_part=guitar_part,
            demucs_stem=guitar_raw,
            transcription_stem=stem_path,
            input_audio=audio_path,
            coarse_stems=separate_result.coarse_stems,
            solo_demix=demix_run.solo if demix_run else None,
            rhythm_demix=demix_run.rhythm if demix_run else None,
        )

        await report("transcribe", "Running Basic Pitch transcription", sub_progress=0.0)
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="transcribe"), "transcribe") as transcribe_metrics:
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
            if select_fingering_optimizer(self.routing) == "aco":
                from dataclasses import asdict

                aco_cfg = ACOConfig(**asdict(self.routing.fingering.aco))
                tab_notes = optimize_sequence_aco(tab_notes, aco_cfg)
                log.info("Fingering optimizer=aco ants=%d iterations=%d", aco_cfg.n_ants, aco_cfg.n_iterations)
            else:
                tab_notes = optimize_sequence(tab_notes)
                log.info("Fingering optimizer=dp")
            log.info("After fingering notes=%d", len(tab_notes))
        await report(
            "transcribe",
            f"Transcribed {len(tab_notes)} notes @ {bpm_result.bpm:.0f} BPM",
            sub_progress=1.0,
            stage_duration_sec=transcribe_metrics.get("elapsed_sec"),
        )

        await report("vision", "Analyzing hand position on fretboard", sub_progress=0.0)
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="vision"), "vision") as vision_metrics:
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
        await report(
            "vision",
            f"Vision frames={len(vision_result.frames)}",
            sub_progress=1.0,
            stage_duration_sec=vision_metrics.get("elapsed_sec"),
        )

        await report("fusion", "Merging audio and vision signals", sub_progress=0.0)
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="fusion"), "fusion") as fusion_metrics:
            fused_notes = self.fusion.fuse_notes(
                tab_notes,
                vision_result.frames,
                audio_only=vision_result.fallback_audio_only,
            )
            conflict_count = sum(1 for n in fused_notes if "audio_vision_mismatch" in n.flags)
            log.info("Fusion notes=%d conflicts=%d", len(fused_notes), conflict_count)
        await report(
            "fusion",
            f"Fusion complete, {conflict_count} conflicts",
            sub_progress=1.0,
            stage_duration_sec=fusion_metrics.get("elapsed_sec"),
        )

        await report("judge", "Running music theory validation", sub_progress=0.0)
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="judge"), "judge") as judge_metrics:
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
        await report(
            "judge",
            f"Judge: {judged.key.root} {judged.key.mode}, snapped={judged.stats.snapped_notes}",
            sub_progress=1.0,
            stage_duration_sec=judge_metrics.get("elapsed_sec"),
        )

        await report("draft", "Building TabDocument", sub_progress=0.0)
        document = self._build_document(
            job_id=job_id,
            source_meta=source_meta,
            bpm=bpm_result.bpm,
            key=judged.key.root,
            mode=judged.key.mode,
            tuning=tuning,
            notes=judged.notes,
            chords=judged.chords,
            key_confidence=judged.key.confidence,
            title=source.get("title"),
            artist=source.get("artist"),
            guitar_part=guitar_part,
        )

        draft_path = work_dir / "draft.json"
        draft_path.write_text(document.model_dump_json(indent=2))
        log.info(
            "Draft saved path=%s measures=%d overall_confidence=%.3f",
            draft_path,
            sum(len(t.measures) for t in document.tracks),
            document.meta.overall_confidence,
        )
        await report(
            "draft",
            f"Draft saved ({sum(len(t.measures) for t in document.tracks)} measures)",
            sub_progress=1.0,
        )
        await report(
            "done",
            f"Pipeline complete — confidence {document.meta.overall_confidence * 100:.0f}%",
            sub_progress=1.0,
            finished=True,
        )
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
        key_confidence: float = 0.0,
        title: str | None = None,
        artist: str | None = None,
        guitar_part: GuitarPart = "combined",
    ) -> TabDocument:
        notes_list = list(notes)
        ref_summary: dict[str, object] = {}
        profile = resolve_reference_profile(title, artist)
        if profile:
            notes_list, ref_summary = apply_reference_scoring(notes_list, profile)

        ms_per_measure = (60_000 / bpm) * 4 if bpm > 0 else 2000.0
        measures_map: dict[int, TabMeasure] = {}
        chord_by_measure = {c.measure_index: c.symbol for c in (chords or [])}

        for note in notes_list:
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
        overall = sum(n.confidence.overall for n in notes_list) / len(notes_list) if notes_list else 0.0
        quality = compute_quality_metrics(notes_list, key_confidence=key_confidence)
        if ref_summary:
            quality = merge_reference_into_quality(quality, ref_summary)

        return TabDocument(
            job_id=job_id,
            meta=TabMeta(
                title=title,
                artist=artist,
                bpm=bpm,
                key=key,
                mode=mode,
                tuning=tuning,
                guitar_part=guitar_part,
                source=source_meta,
                overall_confidence=overall,
                quality=quality,
            ),
            tracks=[TabTrack(name=GUITAR_PART_LABELS.get(guitar_part, "Guitar"), measures=measures)],
        )

    @staticmethod
    def load_draft(path: Path) -> TabDocument:
        return TabDocument.model_validate_json(path.read_text())

    @staticmethod
    def save_draft(path: Path, document: TabDocument) -> None:
        path.write_text(document.model_dump_json(indent=2))
