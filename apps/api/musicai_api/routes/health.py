"""Health and inference status routes."""

from __future__ import annotations

from fastapi import APIRouter
from inference.registry import ModelRegistry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/inference/status")
async def inference_status() -> dict[str, object]:
    registry = ModelRegistry.from_config()
    return {
        "models": registry.list_models(),
        "health": registry.healthcheck_all(),
    }
