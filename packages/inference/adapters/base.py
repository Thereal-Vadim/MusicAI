"""Base model adapter protocol."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    model_id: str
    runtime: Literal["local", "cloud"]

    async def predict(self, input_data: object) -> object:
        ...

    def healthcheck(self) -> bool:
        ...
