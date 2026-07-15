"""Inference configuration and status API."""

from __future__ import annotations

from fastapi import APIRouter

from inference.registry import ModelRegistry

router = APIRouter(prefix="/v1/inference", tags=["inference"])


@router.get("/status")
async def inference_status() -> dict[str, object]:
    registry = ModelRegistry.from_config()
    return {
        "models": registry.list_models(),
        "health": registry.healthcheck_all(),
        "adapters": registry.describe_all(),
    }


@router.get("/config")
async def inference_config() -> dict[str, object]:
    registry = ModelRegistry.from_config()
    return registry.runtime_config()
