"""Preflight checks before starting ML pipeline stages."""

from __future__ import annotations

from inference.pipeline_routing import StageRouting
from inference.registry import ModelRegistry

GUITAR_DEMIX_PARTS = frozenset({"solo", "rhythm"})


class GuitarDemixUnavailableError(RuntimeError):
    """Raised when solo/rhythm demix is requested but no demix backend is ready."""


def guitar_demix_required(guitar_part: str) -> bool:
    return guitar_part in GUITAR_DEMIX_PARTS


def _demix_chain_healthy(registry: ModelRegistry, routing: StageRouting) -> bool:
    for backend_id in routing.chain:
        try:
            adapter = registry.get(backend_id)
        except KeyError:
            continue
        if adapter.healthcheck():
            return True
    return False


def guitar_demix_unavailable_reason(
    registry: ModelRegistry,
    routing: StageRouting | None = None,
) -> str | None:
    """Return a user-facing reason if no demix backend can run, else None."""
    if routing is None:
        from inference.pipeline_routing import load_pipeline_routing

        routing = load_pipeline_routing().guitar_demix

    if _demix_chain_healthy(registry, routing):
        return None

    return (
        "No guitar demix backend is available. "
        "HPSS demix should always be healthy; check models.yaml and restart the API."
    )


def assert_guitar_demix_available(
    registry: ModelRegistry,
    guitar_part: str,
    routing: StageRouting | None = None,
) -> None:
    if not guitar_demix_required(guitar_part):
        return
    reason = guitar_demix_unavailable_reason(registry, routing)
    if reason:
        raise GuitarDemixUnavailableError(reason)
