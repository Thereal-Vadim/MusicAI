"""Tests for modular pipeline routing and ensemble cascade."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference.adapters.bs_roformer_adapter import BSRoFormerAdapter
from inference.adapters.wave_unet_adapter import WaveUNetAdapter
from inference.pipeline_routing import load_pipeline_routing
from inference.registry import ModelRegistry
from inference.schemas.model_io import SeparateInput, SeparateOutput
from musicai_worker.stage_runners import (
    ENSEMBLE_BACKEND_ID,
    _ensemble_steps_healthy,
    run_coarse_separation,
)


def test_pipeline_routing_loads_ensemble_cascade():
    routing = load_pipeline_routing()
    assert routing.coarse_separation.primary == "ensemble/roformer_plus_demucs"
    assert routing.coarse_separation.is_ensemble is True
    assert routing.coarse_separation.ensemble is not None
    assert routing.coarse_separation.ensemble.step_1 == "bs-roformer/vocals-isolation"
    assert routing.coarse_separation.ensemble.step_2 == "demucs/htdemucs_6s"
    assert routing.coarse_separation.chain == ("demucs/htdemucs_6s",)
    assert routing.guitar_demix.primary == "wave-unet/guitar-demix"
    assert routing.guitar_demix.fallbacks == ()
    assert routing.audio_cleanup.enabled is True
    assert routing.audio_cleanup.primary == "spectral/dereverb"
    assert routing.timbre_classify.enabled is True
    assert routing.timbre_classify.primary == "audio-classifier/ast-audioset"


def test_registry_includes_new_adapters():
    registry = ModelRegistry.from_config()
    models = registry.list_models()
    assert "bs-roformer/guitar-4stem" in models
    assert "bs-roformer/vocals-isolation" in models
    assert "wave-unet/guitar-demix" in models
    assert "hpss/guitar-demix" not in models
    assert "spectral/dereverb" in models
    assert "audio-classifier/ast-audioset" in models


def test_bs_roformer_unhealthy_without_checkpoint():
    adapter = BSRoFormerAdapter(checkpoint_path=Path("/nonexistent/model.ckpt"))
    assert adapter.healthcheck() is False


def test_wave_unet_unhealthy_without_weights():
    adapter = WaveUNetAdapter(weights_path=Path("/nonexistent/weights.pt"))
    assert adapter.healthcheck() is False


def test_coarse_routing_skips_unhealthy_ensemble_steps():
    registry = ModelRegistry.from_config()
    routing = load_pipeline_routing()
    assert _ensemble_steps_healthy(registry, routing.coarse_separation) is False
    assert registry.get("demucs/htdemucs_6s").healthcheck() is True


@pytest.mark.asyncio
async def test_ensemble_merge_produces_four_stems(tmp_path: Path) -> None:
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"mix")
    vocals = tmp_path / "vocals.wav"
    instrumental = tmp_path / "instrumental.wav"
    bass = tmp_path / "bass.wav"
    drums = tmp_path / "drums.wav"
    guitar = tmp_path / "guitar.wav"
    for path, payload in [
        (vocals, b"voc"),
        (instrumental, b"inst"),
        (bass, b"bas"),
        (drums, b"dru"),
        (guitar, b"gtr"),
    ]:
        path.write_bytes(payload)

    roformer = MagicMock()
    roformer.healthcheck.return_value = True
    roformer.predict = AsyncMock(
        return_value=SeparateOutput(
            stem_path=instrumental,
            model_id="bs-roformer/vocals-isolation",
            coarse_stems={"vocals": vocals, "guitar": instrumental},
            isolation_method="bs_roformer_4stem",
        )
    )

    demucs = MagicMock()
    demucs.healthcheck.return_value = True
    demucs.predict = AsyncMock(
        return_value=SeparateOutput(
            stem_path=guitar,
            model_id="demucs/htdemucs_6s",
            coarse_stems={"bass": bass, "drums": drums, "guitar": guitar},
            isolation_method="demucs_multi",
        )
    )

    registry = MagicMock()
    registry.get.side_effect = lambda model_id: {
        "bs-roformer/vocals-isolation": roformer,
        "demucs/htdemucs_6s": demucs,
    }[model_id]

    routing = load_pipeline_routing()
    result = await run_coarse_separation(
        registry,
        routing.coarse_separation,
        SeparateInput(audio=audio, output_dir=tmp_path / "stems", mode="multi_stem"),
    )

    assert result.backend_id == ENSEMBLE_BACKEND_ID
    assert result.output.isolation_method == "ensemble_roformer_demucs"
    assert set(result.output.coarse_stems.keys()) == {"vocals", "bass", "drums", "guitar"}
    assert result.output.stem_path.name == "guitar.wav"
    assert (tmp_path / "stems" / "ensemble" / "mix" / "standard" / "guitar.wav").exists()
    step2_input = demucs.predict.await_args.args[0]
    assert step2_input.audio == instrumental
