"""Cloud inference adapter stubs (post-MVP)."""

from __future__ import annotations

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import SeparateInput, SeparateOutput


class ReplicateDemucsAdapter(BaseModelAdapter):
    def __init__(self, model_id: str = "replicate/demucs", api_token: str | None = None) -> None:
        self.model_id = model_id
        self.api_token = api_token
        self.runtime = "cloud"

    def healthcheck(self) -> bool:
        return bool(self.api_token)

    async def predict(self, input_data: SeparateInput) -> SeparateOutput:
        raise NotImplementedError(
            "Cloud inference is not implemented in MVP. Set INFERENCE_RUNTIME=local."
        )
