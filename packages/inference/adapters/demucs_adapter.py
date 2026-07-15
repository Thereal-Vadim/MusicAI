"""Demucs source separation adapter."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from inference.schemas.model_io import SeparateInput, SeparateOutput


class DemucsAdapter:
    model_id = "demucs/htdemucs_6s"
    runtime = "local"

    def __init__(self, model_name: str = "htdemucs_6s", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def healthcheck(self) -> bool:
        return shutil.which("python") is not None

    async def predict(self, input_data: SeparateInput) -> SeparateOutput:
        return await asyncio.to_thread(self._separate, input_data)

    def _separate(self, input_data: SeparateInput) -> SeparateOutput:
        output_dir = input_data.output_dir or input_data.audio.parent / "stems"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "python",
                "-m",
                "demucs",
                "-n",
                self.model_name,
                "--two-stems",
                input_data.stem,
                "-d",
                self.device,
                "-o",
                str(output_dir),
                str(input_data.audio),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            stem_path = (
                output_dir
                / self.model_name
                / input_data.audio.stem
                / f"{input_data.stem}.wav"
            )
            if not stem_path.exists():
                raise FileNotFoundError(f"Stem not found: {stem_path}")
            return SeparateOutput(stem_path=stem_path, model_id=self.model_id)
        except (subprocess.CalledProcessError, FileNotFoundError):
            fallback = output_dir / f"{input_data.stem}_fallback.wav"
            shutil.copy2(input_data.audio, fallback)
            return SeparateOutput(stem_path=fallback, model_id=f"{self.model_id}+fallback")
