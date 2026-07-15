"""Preflight checks for guitar demix availability."""

import pytest

from inference.pipeline_routing import load_pipeline_routing
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


def test_guitar_demix_available_with_hpss_primary():
    registry = ModelRegistry.from_config()
    routing = load_pipeline_routing()
    assert routing.guitar_demix.primary == "hpss/guitar-demix"
    assert guitar_demix_unavailable_reason(registry, routing.guitar_demix) is None


def test_assert_guitar_demix_skips_combined():
    registry = ModelRegistry.from_config()
    assert_guitar_demix_available(registry, "combined")


def test_assert_guitar_demix_allows_rhythm_with_hpss():
    registry = ModelRegistry.from_config()
    routing = load_pipeline_routing()
    assert_guitar_demix_available(registry, "rhythm", routing.guitar_demix)
