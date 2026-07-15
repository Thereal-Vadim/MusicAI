"""Stage runners with registry-backed fallback chains."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.pipeline_routing import (
    AudioCleanupRouting,
    CoarseSeparationRouting,
    PipelineRouting,
    StageRouting,
    TimbreClassifyRouting,
)
from inference.registry import ModelRegistry
from inference.schemas.model_io import (
    DereverbInput,
    GuitarDemixInput,
    SeparateInput,
    SeparateOutput,
    TimbreClassifyInput,
    TimbreClassifyOutput,
)

log = logging.getLogger("musicai.stage_runners")

ENSEMBLE_BACKEND_ID = "ensemble/roformer_plus_demucs"


@dataclass(frozen=True)
class CoarseSeparationResult:
    output: SeparateOutput
    backend_id: str


@dataclass(frozen=True)
class AudioCleanupRunResult:
    audio: Path
    method: str
    backend_id: str
    diagnostics: dict[str, float]


# Backwards-compatible alias
DereverbRunResult = AudioCleanupRunResult


@dataclass(frozen=True)
class GuitarDemixRunResult:
    solo: Path
    rhythm: Path
    method: str
    backend_id: str
    diagnostics: dict[str, float]


def _adapter_healthy(registry: ModelRegistry, model_id: str) -> bool:
    try:
        adapter = registry.get(model_id)
    except KeyError:
        return False
    return adapter.healthcheck()


def _ensemble_steps_healthy(
    registry: ModelRegistry,
    routing: CoarseSeparationRouting,
) -> bool:
    if not routing.ensemble:
        return False
    return _adapter_healthy(registry, routing.ensemble.step_1) and _adapter_healthy(
        registry, routing.ensemble.step_2
    )


async def _run_single_coarse_backend(
    registry: ModelRegistry,
    backend_id: str,
    separate_input: SeparateInput,
) -> CoarseSeparationResult:
    adapter = registry.get(backend_id)
    if not adapter.healthcheck():
        raise RuntimeError(f"Backend {backend_id} is unhealthy")

    output = await adapter.predict(separate_input)
    log.info("Coarse separation via %s method=%s", backend_id, output.isolation_method)
    return CoarseSeparationResult(output=output, backend_id=backend_id)


async def _run_coarse_ensemble(
    registry: ModelRegistry,
    routing: CoarseSeparationRouting,
    separate_input: SeparateInput,
) -> CoarseSeparationResult:
    if not routing.ensemble:
        raise RuntimeError("Ensemble steps are not configured")

    step_1_id = routing.ensemble.step_1
    step_2_id = routing.ensemble.step_2
    output_dir = separate_input.output_dir or separate_input.audio.parent / "stems"
    work_root = output_dir / "ensemble" / separate_input.audio.stem
    standard_dir = work_root / "standard"
    standard_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Ensemble coarse separation step_1=%s step_2=%s audio=%s",
        step_1_id,
        step_2_id,
        separate_input.audio.name,
    )

    step1_out = (
        await _run_single_coarse_backend(
            registry,
            step_1_id,
            separate_input.model_copy(
                update={"output_dir": work_root / "step1_roformer"},
            ),
        )
    ).output

    vocals = step1_out.coarse_stems.get("vocals")
    instrumental = step1_out.coarse_stems.get("guitar")
    if vocals is None or instrumental is None:
        raise FileNotFoundError(
            f"RoFormer step missing vocals/instrumental stems: {list(step1_out.coarse_stems.keys())}"
        )

    step2_out = (
        await _run_single_coarse_backend(
            registry,
            step_2_id,
            SeparateInput(
                audio=instrumental,
                stem="guitar",
                guitar_part=separate_input.guitar_part,
                mode="multi_stem",
                output_dir=work_root / "step2_demucs",
            ),
        )
    ).output

    coarse_stems: dict[str, Path] = {}
    shutil.copy2(vocals, standard_dir / "vocals.wav")
    coarse_stems["vocals"] = standard_dir / "vocals.wav"

    for stem_id in ("bass", "drums", "guitar"):
        stem_path = step2_out.coarse_stems.get(stem_id)
        if stem_path and stem_path.exists() and stem_path.stat().st_size > 0:
            dest = standard_dir / f"{stem_id}.wav"
            shutil.copy2(stem_path, dest)
            coarse_stems[stem_id] = dest

    guitar_path = coarse_stems.get("guitar")
    if guitar_path is None:
        raise FileNotFoundError(
            f"Ensemble Demucs step did not produce guitar stem: {list(step2_out.coarse_stems.keys())}"
        )

    return CoarseSeparationResult(
        output=SeparateOutput(
            stem_path=guitar_path,
            model_id=ENSEMBLE_BACKEND_ID,
            guitar_part=separate_input.guitar_part,
            coarse_stems=coarse_stems,
            isolation_method="ensemble_roformer_demucs",
        ),
        backend_id=ENSEMBLE_BACKEND_ID,
    )


async def run_coarse_separation(
    registry: ModelRegistry,
    routing: CoarseSeparationRouting,
    separate_input: SeparateInput,
) -> CoarseSeparationResult:
    last_error: Exception | None = None

    if routing.is_ensemble:
        if _ensemble_steps_healthy(registry, routing):
            try:
                return await _run_coarse_ensemble(registry, routing, separate_input)
            except Exception as exc:
                log.warning("Ensemble coarse separation failed: %s", exc)
                last_error = exc
        else:
            log.info(
                "Ensemble steps unhealthy (step_1=%s step_2=%s), using fallbacks",
                routing.ensemble.step_1 if routing.ensemble else "?",
                routing.ensemble.step_2 if routing.ensemble else "?",
            )

    for backend_id in routing.chain:
        if backend_id.startswith("ensemble/"):
            continue
        try:
            return await _run_single_coarse_backend(registry, backend_id, separate_input)
        except KeyError:
            log.debug("Coarse backend %s not registered, skipping", backend_id)
            continue
        except Exception as exc:
            log.warning("Coarse backend %s failed: %s", backend_id, exc)
            last_error = exc

    raise RuntimeError(
        f"No coarse separation backend available (primary={routing.primary}, fallbacks={routing.fallbacks})"
    ) from last_error


async def run_audio_cleanup(
    registry: ModelRegistry,
    routing: AudioCleanupRouting,
    *,
    audio: Path,
    output_path: Path,
) -> AudioCleanupRunResult | None:
    """Spectral dereverb + noise gate. Returns None when stage is disabled."""
    if not routing.enabled:
        log.info("Audio cleanup stage disabled; using raw transcription stem")
        return None

    last_error: Exception | None = None
    for backend_id in routing.chain:
        try:
            adapter = registry.get(backend_id)
        except KeyError:
            log.debug("Audio cleanup backend %s not registered, skipping", backend_id)
            continue

        if not adapter.healthcheck():
            log.info("Audio cleanup backend %s unhealthy", backend_id)
            continue

        try:
            output = await adapter.predict(
                DereverbInput(audio=audio, output_path=output_path)
            )
            return AudioCleanupRunResult(
                audio=output.audio_path,
                method=output.method,
                backend_id=backend_id,
                diagnostics=output.diagnostics,
            )
        except Exception as exc:
            log.warning("Audio cleanup backend %s failed: %s", backend_id, exc)
            last_error = exc

    if last_error:
        log.warning("Audio cleanup unavailable, continuing with raw stem: %s", last_error)
    return None


# Backwards-compatible alias
run_dereverb = run_audio_cleanup


async def run_guitar_demix(
    registry: ModelRegistry,
    routing: StageRouting,
    *,
    guitar_stem: Path,
    output_dir: Path,
    mix_path: Path | None,
) -> GuitarDemixRunResult:
    last_error: Exception | None = None

    for backend_id in routing.chain:
        try:
            adapter = registry.get(backend_id)
        except KeyError:
            log.debug("Demix backend %s not registered, skipping", backend_id)
            continue

        if not adapter.healthcheck():
            log.info("Demix backend %s unhealthy", backend_id)
            continue

        try:
            output = await adapter.predict(
                GuitarDemixInput(
                    guitar_stem=guitar_stem,
                    output_dir=output_dir,
                    mix_path=mix_path,
                )
            )
            return GuitarDemixRunResult(
                solo=output.solo_path,
                rhythm=output.rhythm_path,
                method=output.method,
                backend_id=backend_id,
                diagnostics=output.diagnostics,
            )
        except Exception as exc:
            log.warning("Demix backend %s failed: %s", backend_id, exc)
            last_error = exc

    raise RuntimeError(
        f"Wave-U-Net demix unavailable (primary={routing.primary}). "
        "Configure WAVE_UNET_WEIGHTS in .env"
    ) from last_error


@dataclass(frozen=True)
class TimbreClassifyRunResult:
    type: str
    midi_program: int
    label: str
    confidence: float
    backend_id: str
    top_labels: list[dict[str, Any]]


async def run_timbre_classify(
    registry: ModelRegistry,
    routing: TimbreClassifyRouting,
    *,
    audio: Path,
) -> TimbreClassifyRunResult | None:
    """AST guitar timbre classification. Returns None when disabled or unavailable."""
    if not routing.enabled:
        log.info("Timbre classify stage disabled")
        return None

    last_error: Exception | None = None
    for backend_id in routing.chain:
        try:
            adapter = registry.get(backend_id)
        except KeyError:
            log.debug("Timbre classify backend %s not registered, skipping", backend_id)
            continue

        if not adapter.healthcheck():
            log.info("Timbre classify backend %s unhealthy", backend_id)
            continue

        try:
            output: TimbreClassifyOutput = await adapter.predict(TimbreClassifyInput(audio=audio))
            return TimbreClassifyRunResult(
                type=output.type,
                midi_program=output.midi_program,
                label=output.label,
                confidence=output.confidence,
                backend_id=backend_id,
                top_labels=list(output.top_labels),
            )
        except Exception as exc:
            log.warning("Timbre classify backend %s failed: %s", backend_id, exc)
            last_error = exc

    if last_error:
        log.warning("Timbre classify unavailable, using Clean fallback: %s", last_error)
    return None


def select_fingering_optimizer(routing: PipelineRouting) -> str:
    return routing.fingering.optimizer


def log_routing_summary(routing: PipelineRouting) -> dict[str, Any]:
    return routing.to_dict()
