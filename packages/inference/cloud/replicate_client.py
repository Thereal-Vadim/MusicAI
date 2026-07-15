"""Cloud inference adapter stubs (post-MVP)."""

from __future__ import annotations

from inference.schemas.model_io import SeparateInput, SeparateOutput


class ReplicateDemucsAdapter:
    model_id = "replicate/demucs"
    runtime = "cloud"

    def __init__(self, api_token: str | None = None) -> None:
        self.api_token = api_token

    def healthcheck(self) -> bool:
        return bool(self.api_token)

    async def predict(self, input_data: SeparateInput) -> SeparateOutput:
        raise NotImplementedError(
            "Cloud inference is not implemented in MVP. Set INFERENCE_RUNTIME=local."
        )
