"""HPSS + stereo guitar demix adapter (solo vs rhythm heuristic separation)."""

from __future__ import annotations

import asyncio
import logging

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import GuitarDemixInput, GuitarDemixOutput
from musicai_worker.guitar_isolation import demix_guitar_stems

log = logging.getLogger("musicai.hpss_demix")


class HpssDemixAdapter(BaseModelAdapter):
    """
    Heuristic guitar demix: HPSS attributes + stereo panning + Wiener soft masks.

    Works without trained weights. Best on stereo mixes where rhythm is panned L/R.
    """

    def __init__(self, model_id: str = "hpss/guitar-demix") -> None:
        self.model_id = model_id
        self.runtime = "local"

    def healthcheck(self) -> bool:
        return True

    async def predict(self, input_data: GuitarDemixInput) -> GuitarDemixOutput:
        return await asyncio.to_thread(self._demix, input_data)

    def _demix(self, input_data: GuitarDemixInput) -> GuitarDemixOutput:
        solo_path, rhythm_path, diagnostics = demix_guitar_stems(
            input_data.guitar_stem,
            input_data.output_dir,
            mix_path=input_data.mix_path,
        )
        log.info("HPSS demix complete stereo=%s", diagnostics.get("stereo_used"))
        return GuitarDemixOutput(
            solo_path=solo_path,
            rhythm_path=rhythm_path,
            model_id=self.model_id,
            method="hpss_stereo_v2",
            diagnostics=diagnostics,
        )

    def describe(self) -> dict[str, object]:
        base = super().describe()
        base["backend"] = "hpss"
        base["requires_stereo_mix"] = False
        base["stereo_improves_quality"] = True
        return base
