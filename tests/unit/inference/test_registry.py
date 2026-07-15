"""Inference adapter tests."""

from inference.registry import ModelRegistry


def test_registry_loads_models():
    registry = ModelRegistry.from_config()
    models = registry.list_models()
    assert "demucs/htdemucs_6s" in models
    assert "basic-pitch/v1" in models
    assert "mediapipe/hands" in models
    assert "librosa/beat" in models


def test_registry_healthcheck():
    registry = ModelRegistry.from_config()
    health = registry.healthcheck_all()
    assert isinstance(health, dict)
    assert "demucs/htdemucs_6s" in health
