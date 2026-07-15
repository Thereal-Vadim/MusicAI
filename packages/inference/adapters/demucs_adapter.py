"""Demucs source separation adapter."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import SeparateInput, SeparateOutput

COARSE_STEMS = ("drums", "bass", "vocals", "guitar")


class DemucsAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_id: str = "demucs/htdemucs_6s",
        model_name: str = "htdemucs_6s",
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.device = device
        self.runtime = "local"

    def healthcheck(self) -> bool:
        try:
            import demucs  # noqa: F401

            return True
        except ImportError:
            return shutil.which("python") is not None

    async def predict(self, input_data: SeparateInput) -> SeparateOutput:
        return await asyncio.to_thread(self._separate, input_data)

    def _separate(self, input_data: SeparateInput) -> SeparateOutput:
        output_dir = input_data.output_dir or input_data.audio.parent / "stems"
        output_dir.mkdir(parents=True, exist_ok=True)
        log = logging.getLogger("musicai.demucs")

        try:
            cmd = [
                sys.executable,
                "-m",
                "demucs",
                "-n",
                self.model_name,
            ]
            if input_data.mode == "two_stems":
                cmd.extend(["--two-stems", input_data.stem])
            cmd.extend(
                [
                    "-d",
                    self.device,
                    "-o",
                    str(output_dir),
                    str(input_data.audio),
                ]
            )

            log.info("Running command: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            stem_dir = output_dir / self.model_name / input_data.audio.stem
            coarse_stems: dict[str, Path] = {}
            for name in COARSE_STEMS:
                candidate = stem_dir / f"{name}.wav"
                if candidate.exists() and candidate.stat().st_size > 0:
                    coarse_stems[name] = candidate

            guitar_path = coarse_stems.get("guitar")
            if guitar_path is None:
                guitar_path = stem_dir / f"{input_data.stem}.wav"
            if not guitar_path.exists():
                raise FileNotFoundError(f"Guitar stem not found under {stem_dir}")

            return SeparateOutput(
                stem_path=guitar_path,
                model_id=self.model_id,
                guitar_part=input_data.guitar_part,
                coarse_stems=coarse_stems,
                isolation_method="demucs_multi" if input_data.mode == "multi_stem" else "demucs_two_stems",
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            log.warning("Demucs failed (%s), using audio fallback", exc)
            fallback = output_dir / f"{input_data.stem}_fallback.wav"
            shutil.copy2(input_data.audio, fallback)
            return SeparateOutput(
                stem_path=fallback,
                model_id=f"{self.model_id}+fallback",
                guitar_part=input_data.guitar_part,
                coarse_stems={},
                isolation_method="fallback",
            )
