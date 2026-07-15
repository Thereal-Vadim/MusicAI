"""Worker pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference.pipeline_routing import load_pipeline_routing
from inference.preflight import assert_guitar_demix_available
from inference.registry import ModelRegistry
from inference.schemas.model_io import BpmInput, SeparateInput, TranscribeInput, VisionInput
from judge.judge import JudgeConfig, MusicTheoryJudge
from musicai_worker.demix_validator import validate_guitar_demix
from musicai_worker.fingering.aco_optimizer import ACOConfig, optimize_sequence_aco
from musicai_worker.fingering.optimizer import assign_fingering, optimize_sequence
from musicai_worker.fusion.scorer import FusionScorer
from musicai_worker.guitar_isolation import GUITAR_PART_LABELS, GuitarPart
from musicai_worker.logging_setup import get_logger, log_artifact, setup_logging, stage_timer
from musicai_worker.stage_runners import (
    log_routing_summary,
    run_audio_cleanup,
    run_coarse_separation,
    run_guitar_demix,
    run_timbre_classify,
    select_fingering_optimizer,
)
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
    "audio_cleanup",
    "timbre_classify",
    "transcribe",
    "vision",
    "fusion",
    "judge",
    "draft",
    "done",
    "failed",
]

DEFAULT_TIMBRE_TYPE = "Electric Guitar (clean)"
DEFAULT_MIDI_PROGRAM = 27


def _track_display_name(timbre_type: str, role: str) -> str:
    if role == "solo":
        return f"{timbre_type} (Lead)"
    if role == "rhythm":
        return f"{timbre_type} (Rhythm)"
    return timbre_type


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

        assert_guitar_demix_available(self.registry, guitar_part, self.routing.guitar_demix)

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
            guitar_coarse = separate_result.stem_path
            log.info(
                "Coarse separation backend=%s method=%s stems=%s guitar=%s",
                coarse.backend_id,
                separate_result.isolation_method,
                list(separate_result.coarse_stems.keys()),
                guitar_coarse,
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
        cleanup_run = None
        stem_path = guitar_coarse
        isolation_method = separate_result.isolation_method or "coarse_multi"

        if guitar_part != "combined":
            await report("guitar_demix", "Guitar de-mix (solo / rhythm)", sub_progress=0.0)
            with stage_timer(
                get_logger("pipeline", job_id=job_id, stage="guitar_demix"), "guitar_demix"
            ) as demix_metrics:
                demix_run = await run_guitar_demix(
                    self.registry,
                    self.routing.guitar_demix,
                    guitar_stem=guitar_coarse,
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

                if not validation_report.passed:
                    log_artifact(log_dir, "demix_validation.json", validation_report.to_dict())
                    raise RuntimeError(
                        "Wave-U-Net demix failed playability validation "
                        f"(leakage={validation_report.leakage_score:.2f}, "
                        f"merged_poly={validation_report.merged_max_polyphony}). "
                        "No HPSS heuristic fallback — fix WAVE_UNET_WEIGHTS or use guitar_part=combined."
                    )

                stem_path = demix_run.solo if guitar_part == "solo" else demix_run.rhythm
                isolation_method = f"{coarse.backend_id}+{demix_run.backend_id}+validated"
                log_artifact(log_dir, "demix_validation.json", validation_report.to_dict())
            await report(
                "demix_validate",
                f"Validation OK · {part_label}",
                sub_progress=1.0,
                stage_duration_sec=validate_metrics.get("elapsed_sec"),
            )

        # Build the list of stems to clean / classify / transcribe.
        # Dual path when demix validated: both solo + rhythm.
        stem_jobs: list[dict[str, Any]] = []
        if demix_run is not None and validation_report is not None and validation_report.passed:
            stem_jobs = [
                {"role": "solo", "source": demix_run.solo, "out_name": "cleaned_di_solo.wav"},
                {"role": "rhythm", "source": demix_run.rhythm, "out_name": "cleaned_di_rhythm.wav"},
            ]
        else:
            role = guitar_part if guitar_part in ("solo", "rhythm", "combined") else "combined"
            stem_jobs = [
                {"role": role, "source": stem_path, "out_name": "cleaned_di_guitar.wav"},
            ]

        await report("audio_cleanup", "DI cleanup: dereverb + noise gate", sub_progress=0.0)
        cleanup_runs: dict[str, Any] = {}
        with stage_timer(
            get_logger("pipeline", job_id=job_id, stage="audio_cleanup"), "audio_cleanup"
        ) as cleanup_metrics:
            for job in stem_jobs:
                cleanup_run = await run_audio_cleanup(
                    self.registry,
                    self.routing.audio_cleanup,
                    audio=job["source"],
                    output_path=work_dir / "stems" / job["out_name"],
                )
                cleaned = cleanup_run.audio if cleanup_run is not None else job["source"]
                job["cleaned"] = cleaned
                if cleanup_run is not None:
                    cleanup_runs[job["role"]] = cleanup_run
                    log.info(
                        "Audio cleanup role=%s backend=%s method=%s",
                        job["role"],
                        cleanup_run.backend_id,
                        cleanup_run.method,
                    )
            if cleanup_runs:
                isolation_method = f"{isolation_method}+{next(iter(cleanup_runs.values())).backend_id}"
                log_artifact(
                    log_dir,
                    "audio_cleanup.json",
                    {
                        role: {
                            "backend_id": run.backend_id,
                            "method": run.method,
                            "path": str(run.audio),
                            **run.diagnostics,
                        }
                        for role, run in cleanup_runs.items()
                    },
                )
            else:
                log.info("Audio cleanup skipped; using demix/coarse stems as-is")
                for job in stem_jobs:
                    job["cleaned"] = job["source"]
        primary_cleanup = cleanup_runs.get(guitar_part) or next(iter(cleanup_runs.values()), None)
        stem_path = next(
            (j["cleaned"] for j in stem_jobs if j["role"] == guitar_part),
            stem_jobs[0]["cleaned"],
        )
        await report(
            "audio_cleanup",
            (
                f"DI cleanup {len(cleanup_runs)} stem(s)"
                if cleanup_runs
                else "Audio cleanup skipped"
            ),
            sub_progress=1.0,
            stage_duration_sec=cleanup_metrics.get("elapsed_sec"),
        )

        await report("timbre_classify", "Classifying guitar timbre (AST)", sub_progress=0.0)
        with stage_timer(
            get_logger("pipeline", job_id=job_id, stage="timbre_classify"), "timbre_classify"
        ) as classify_metrics:
            for job in stem_jobs:
                timbre = await run_timbre_classify(
                    self.registry,
                    self.routing.timbre_classify,
                    audio=job["cleaned"],
                )
                if timbre is None:
                    job["timbre_type"] = DEFAULT_TIMBRE_TYPE
                    job["midi_program"] = DEFAULT_MIDI_PROGRAM
                    job["timbre_label"] = "fallback"
                    job["timbre_confidence"] = 0.0
                else:
                    job["timbre_type"] = timbre.type
                    job["midi_program"] = timbre.midi_program
                    job["timbre_label"] = timbre.label
                    job["timbre_confidence"] = timbre.confidence
                    log.info(
                        "Timbre role=%s type=%s midi=%d label=%s conf=%.3f",
                        job["role"],
                        timbre.type,
                        timbre.midi_program,
                        timbre.label,
                        timbre.confidence,
                    )
            log_artifact(
                log_dir,
                "timbre_classify.json",
                {
                    j["role"]: {
                        "type": j["timbre_type"],
                        "midi_program": j["midi_program"],
                        "label": j["timbre_label"],
                        "confidence": j["timbre_confidence"],
                    }
                    for j in stem_jobs
                },
            )
        await report(
            "timbre_classify",
            ", ".join(f"{j['role']}={j['timbre_type']}" for j in stem_jobs),
            sub_progress=1.0,
            stage_duration_sec=classify_metrics.get("elapsed_sec"),
        )

        log.info(
            "Transcription stems=%s method=%s part=%s",
            [(j["role"], str(j["cleaned"])) for j in stem_jobs],
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
                "track_roles": [j["role"] for j in stem_jobs],
            },
        )
        from musicai_worker.stem_manifest import write_stems_manifest

        write_stems_manifest(
            work_dir,
            guitar_part=guitar_part,
            demucs_stem=guitar_coarse,
            transcription_stem=stem_path,
            input_audio=audio_path,
            coarse_stems=separate_result.coarse_stems,
            solo_demix=demix_run.solo if demix_run else None,
            rhythm_demix=demix_run.rhythm if demix_run else None,
            dereverb_stem=primary_cleanup.audio if primary_cleanup else None,
        )

        await report("transcribe", "Running Basic Pitch transcription", sub_progress=0.0)
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="transcribe"), "transcribe") as transcribe_metrics:
            basic_pitch = self.registry.get("basic-pitch/v1")
            bpm_adapter = self.registry.get("librosa/beat")
            log.info("BasicPitch adapter=%s", basic_pitch.describe())
            bpm_result = await bpm_adapter.predict(BpmInput(audio=audio_path))
            log.info(
                "BPM=%.2f beats=%d model=%s",
                bpm_result.bpm,
                len(bpm_result.beat_times_sec),
                bpm_result.model_id,
            )

            use_aco = select_fingering_optimizer(self.routing) == "aco"
            aco_cfg = None
            if use_aco:
                from dataclasses import asdict

                aco_cfg = ACOConfig(**asdict(self.routing.fingering.aco))

            for job in stem_jobs:
                transcribe_result = await basic_pitch.predict(TranscribeInput(audio=job["cleaned"]))
                log.info(
                    "Transcribed role=%s raw_notes=%d model=%s",
                    job["role"],
                    len(transcribe_result.notes),
                    transcribe_result.model_id,
                )
                raw_tuples = self.fusion.raw_to_tab_notes(transcribe_result.notes)
                tab_notes = assign_fingering(raw_tuples)
                if use_aco and aco_cfg is not None:
                    tab_notes = optimize_sequence_aco(tab_notes, aco_cfg)
                else:
                    tab_notes = optimize_sequence(tab_notes)
                job["tab_notes"] = tab_notes
                log.info("After fingering role=%s notes=%d", job["role"], len(tab_notes))

            log_artifact(
                log_dir,
                "transcribe_notes.json",
                {
                    j["role"]: [n.model_dump() for n in j["tab_notes"][:100]]
                    for j in stem_jobs
                },
            )
        total_notes = sum(len(j["tab_notes"]) for j in stem_jobs)
        await report(
            "transcribe",
            f"Transcribed {total_notes} notes across {len(stem_jobs)} track(s) @ {bpm_result.bpm:.0f} BPM",
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
            conflict_count = 0
            for job in stem_jobs:
                # Apply vision fusion to lead/solo (or the only track); rhythm stays audio-only.
                apply_vision = job["role"] in ("solo", "combined") or len(stem_jobs) == 1
                fused_notes = self.fusion.fuse_notes(
                    job["tab_notes"],
                    vision_result.frames if apply_vision else [],
                    audio_only=vision_result.fallback_audio_only or not apply_vision,
                )
                job["fused_notes"] = fused_notes
                conflict_count += sum(1 for n in fused_notes if "audio_vision_mismatch" in n.flags)
            log.info("Fusion tracks=%d conflicts=%d", len(stem_jobs), conflict_count)
        await report(
            "fusion",
            f"Fusion complete, {conflict_count} conflicts",
            sub_progress=1.0,
            stage_duration_sec=fusion_metrics.get("elapsed_sec"),
        )

        await report("judge", "Running music theory validation", sub_progress=0.0)
        with stage_timer(get_logger("pipeline", job_id=job_id, stage="judge"), "judge") as judge_metrics:
            primary_judged = None
            for job in stem_jobs:
                judged = self.judge.judge(job["fused_notes"], bpm=bpm_result.bpm)
                job["judged"] = judged
            # Prefer solo/lead as the document-level key/meta source.
            for preferred in ("solo", "combined", guitar_part, stem_jobs[0]["role"]):
                match = next((j for j in stem_jobs if j["role"] == preferred), None)
                if match is not None:
                    primary_judged = match["judged"]
                    break
            if primary_judged is None:
                primary_judged = stem_jobs[0]["judged"]
            judge_report_path = work_dir / "judge_report.json"
            judge_report_path.write_text(
                json.dumps(
                    {
                        j["role"]: j["judged"].to_report()
                        for j in stem_jobs
                    },
                    indent=2,
                )
            )
            log.info(
                "Judge key=%s %s tracks=%d",
                primary_judged.key.root,
                primary_judged.key.mode,
                len(stem_jobs),
            )
        await report(
            "judge",
            f"Judge: {primary_judged.key.root} {primary_judged.key.mode}",
            sub_progress=1.0,
            stage_duration_sec=judge_metrics.get("elapsed_sec"),
        )

        await report("draft", "Building TabDocument", sub_progress=0.0)
        track_specs = [
            {
                "role": j["role"],
                "name": _track_display_name(j["timbre_type"], j["role"]),
                "midi_program": j["midi_program"],
                "notes": j["judged"].notes,
                "chords": j["judged"].chords if j["role"] in ("solo", "combined") else [],
            }
            for j in stem_jobs
        ]
        # Prefer Lead first in multi-track exports.
        track_specs.sort(key=lambda t: 0 if t["role"] == "solo" else 1 if t["role"] == "rhythm" else 2)

        document = self._build_document(
            job_id=job_id,
            source_meta=source_meta,
            bpm=bpm_result.bpm,
            key=primary_judged.key.root,
            mode=primary_judged.key.mode,
            tuning=tuning,
            track_specs=track_specs,
            key_confidence=primary_judged.key.confidence,
            title=source.get("title"),
            artist=source.get("artist"),
            guitar_part=guitar_part,
        )

        draft_path = work_dir / "draft.json"
        draft_path.write_text(document.model_dump_json(indent=2))
        log.info(
            "Draft saved path=%s tracks=%d measures=%d overall_confidence=%.3f",
            draft_path,
            len(document.tracks),
            sum(len(t.measures) for t in document.tracks),
            document.meta.overall_confidence,
        )
        await report(
            "draft",
            f"Draft saved ({len(document.tracks)} track(s), {sum(len(t.measures) for t in document.tracks)} measures)",
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
        track_specs: list[dict[str, Any]],
        key_confidence: float = 0.0,
        title: str | None = None,
        artist: str | None = None,
        guitar_part: GuitarPart = "combined",
    ) -> TabDocument:
        tracks: list[TabTrack] = []
        all_notes_flat: list = []
        ref_summary: dict[str, object] = {}
        profile = resolve_reference_profile(title, artist)
        ms_per_measure = (60_000 / bpm) * 4 if bpm > 0 else 2000.0
        primary_role = "solo" if any(s.get("role") == "solo" for s in track_specs) else (
            "combined" if any(s.get("role") == "combined" for s in track_specs) else guitar_part
        )

        for spec in track_specs:
            notes_list = list(spec.get("notes") or [])
            if profile and spec.get("role") == primary_role:
                notes_list, ref_summary = apply_reference_scoring(notes_list, profile)

            measures_map: dict[int, TabMeasure] = {}
            chords = spec.get("chords") or []
            chord_by_measure = {c.measure_index: c.symbol for c in chords}

            for note in notes_list:
                idx = int(note.start_ms // ms_per_measure)
                if idx not in measures_map:
                    measures_map[idx] = TabMeasure(
                        index=idx,
                        start_ms=idx * ms_per_measure,
                        confidence=note.confidence.overall,
                        chord=chord_by_measure.get(idx),
                    )
                measures_map[idx].notes.append(note)

            role = spec.get("role")
            tracks.append(
                TabTrack(
                    name=spec.get("name") or GUITAR_PART_LABELS.get(guitar_part, "Guitar"),
                    midi_program=spec.get("midi_program"),
                    role=role if role in ("solo", "rhythm", "combined") else None,
                    measures=[measures_map[i] for i in sorted(measures_map.keys())],
                )
            )
            all_notes_flat.extend(notes_list)

        overall = (
            sum(n.confidence.overall for n in all_notes_flat) / len(all_notes_flat)
            if all_notes_flat
            else 0.0
        )
        quality = compute_quality_metrics(all_notes_flat, key_confidence=key_confidence)
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
            tracks=tracks,
        )

    @staticmethod
    def load_draft(path: Path) -> TabDocument:
        return TabDocument.model_validate_json(path.read_text())

    @staticmethod
    def save_draft(path: Path, document: TabDocument) -> None:
        path.write_text(document.model_dump_json(indent=2))
