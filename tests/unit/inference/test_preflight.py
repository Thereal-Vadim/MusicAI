"""Preflight checks for guitar demix availability."""

from unittest.mock import MagicMock

import pytest

from inference.pipeline_routing import StageRouting, load_pipeline_routing
from inference.preflight import (
    GuitarDemixUnavailableError,
    assert_guitar_demix_available,
    guitar_demix_required,
    guitar_demix_unavailable_reason,
)
from inference.registry import ModelRegistry


def test_guitar_demix_required_for_solo_rhythm_only():
    assert guitar_demix_required("solo") is True
    assert guitar_demix_required("rhythm") is True
    assert guitar_demix_required("combined") is False


def test_guitar_demix_primary_is_wave_unet():
    routing = load_pipeline_routing()
    assert routing.guitar_demix.primary == "wave-unet/guitar-demix"
    assert routing.guitar_demix.fallbacks == ()


def test_assert_guitar_demix_skips_combined():
    registry = ModelRegistry.from_config()
    assert_guitar_demix_available(registry, "combined")


def test_assert_guitar_demix_fails_when_wave_unet_unhealthy():
    registry = MagicMock()
    adapter = MagicMock()
    adapter.healthcheck.return_value = False
    registry.get.return_value = adapter

    routing = StageRouting(primary="wave-unet/guitar-demix", fallbacks=())
    reason = guitar_demix_unavailable_reason(registry, routing)
    assert reason is not None
    assert "WAVE_UNET_WEIGHTS" in reason

    with pytest.raises(GuitarDemixUnavailableError):
        assert_guitar_demix_available(registry, "solo", routing)
