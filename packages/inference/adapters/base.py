"""Base model adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal


class BaseModelAdapter(ABC):
    model_id: str
    runtime: Literal["local", "cloud"]

    @abstractmethod
    async def predict(self, input_data: Any) -> Any:
        ...

    @abstractmethod
    def healthcheck(self) -> bool:
        ...

    def describe(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "runtime": self.runtime,
            "healthy": self.healthcheck(),
        }
