"""Expanded inference registry tests."""

from inference.adapters.base import BaseModelAdapter
from inference.registry import ModelRegistry
from inference.settings import InferenceSettings


def test_registry_loads_models():
    registry = ModelRegistry.from_config()
    models = registry.list_models()
    assert "demucs/htdemucs_6s" in models
    assert "basic-pitch/v1" in models
    assert "mediapipe/hands" in models
    assert "librosa/beat" in models
    assert "bs-roformer/guitar-4stem" in models
    assert "wave-unet/guitar-demix" in models
    assert "hpss/guitar-demix" in models


def test_registry_healthcheck():
    registry = ModelRegistry.from_config()
    health = registry.healthcheck_all()
    assert isinstance(health, dict)
    assert "demucs/htdemucs_6s" in health


def test_adapters_implement_base_protocol():
    registry = ModelRegistry.from_config()
    for model_id in registry.list_models():
        adapter = registry.get(model_id)
        assert isinstance(adapter, BaseModelAdapter)
        assert adapter.model_id == model_id


def test_runtime_config_exposes_settings():
    settings = InferenceSettings(
        inference_runtime="local",
        demucs_model="htdemucs_6s",
        demucs_device="cpu",
    )
    registry = ModelRegistry.from_config(settings=settings)
    config = registry.runtime_config()
    assert config["runtime"] == "local"
    assert config["demucs_model"] == "htdemucs_6s"
    assert config["demucs_device"] == "cpu"


def test_describe_all_includes_health():
    registry = ModelRegistry.from_config()
    descriptions = registry.describe_all()
    assert len(descriptions) >= 6
    assert all("healthy" in item for item in descriptions)
