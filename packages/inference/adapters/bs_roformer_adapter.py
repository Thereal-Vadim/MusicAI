"""BS-RoFormer coarse separation adapter (4-stem: vocals / bass / drums / guitar)."""

from __future__ import annotations

import asyncio
import gc
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import SeparateInput, SeparateOutput

COARSE_STEMS = ("vocals", "bass", "drums", "guitar")
log = logging.getLogger("musicai.bs_roformer")

# Aliases for audio-separator / UVR / Demucs output naming conventions.
STEM_ALIASES: dict[str, tuple[str, ...]] = {
    "vocals": ("vocals", "vocal", "voice", "singer", "lead_vocal"),
    "bass": ("bass", "low_end", "lowend"),
    "drums": ("drums", "drum", "percussion", "perc", "kit"),
    "guitar": (
        "guitar",
        "guitarist",
        "other",
        "instrumental",
        "inst",
        "accompaniment",
        "no_vocals",
        "no vocal",
    ),
}

VIPERX_2STEM_OUTPUT_NAMES = {
    "Vocals": "vocals",
    "Instrumental": "guitar",
}

DEMUCS_6S_OUTPUT_NAMES = {
    "Vocals": "vocals",
    "Drums": "drums",
    "Bass": "bass",
    "Guitar": "guitar",
}


def _audio_separator_available() -> bool:
    try:
        import audio_separator  # noqa: F401

        return True
    except ImportError:
        return False


def classify_stem_filename(name: str) -> str | None:
    """Map an output filename to a standard coarse stem id."""
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    for stem_id, aliases in STEM_ALIASES.items():
        for alias in aliases:
            token = alias.replace(" ", "_")
            if token in normalized:
                return stem_id
    return None


def standardize_coarse_stems(
    raw_files: list[Path],
    standard_dir: Path,
) -> dict[str, Path]:
    """
    Copy separator outputs into a fixed layout: vocals.wav, bass.wav, drums.wav, guitar.wav.

    Unmapped files are logged and skipped. Duplicate stem ids keep the first match.
    """
    standard_dir.mkdir(parents=True, exist_ok=True)
    coarse: dict[str, Path] = {}

    for raw_path in raw_files:
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            continue

        stem_id = classify_stem_filename(raw_path.stem)
        if stem_id is None:
            log.warning("BS-RoFormer: unmapped stem output %s", raw_path.name)
            continue
        if stem_id in coarse:
            log.debug("BS-RoFormer: duplicate stem %s from %s", stem_id, raw_path.name)
            continue

        dest = standard_dir / f"{stem_id}.wav"
        if raw_path.resolve() != dest.resolve():
            shutil.copy2(raw_path, dest)
        coarse[stem_id] = dest

    return coarse


def release_accelerator_memory(device: str) -> None:
    """Drop cached tensors after separation to avoid OOM in later pipeline stages."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


class _BSRoFormerRuntime:
    """Lazy singleton: loads ViperX checkpoint into VRAM/RAM on first separation only."""

    _instances: dict[tuple[str, str, int], "_BSRoFormerRuntime"] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        segment_size: int,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.segment_size = segment_size
        self._separator: Any | None = None
        self._model_loaded = False

    @classmethod
    def get(cls, checkpoint_path: Path, device: str, segment_size: int) -> "_BSRoFormerRuntime":
        key = (str(checkpoint_path.resolve()), device, segment_size)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(checkpoint_path, device, segment_size)
            return cls._instances[key]

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def _separator_kwargs(self, output_dir: Path) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "output_format": "WAV",
            "model_file_dir": str(self.checkpoint_path.parent),
            "use_soundfile": True,
            "mdxc_params": {
                "segment_size": self.segment_size,
                "override_model_segment_size": False,
                "batch_size": 1,
                "overlap": 8,
                "pitch_shift": 0,
            },
        }
        if self.device in {"cuda", "mps"}:
            kwargs["use_autocast"] = True
        return kwargs

    def _output_names_for_checkpoint(self) -> dict[str, str]:
        name = self.checkpoint_path.name.lower()
        if "demucs" in name or "6s" in name or "htdemucs" in name:
            return DEMUCS_6S_OUTPUT_NAMES
        return VIPERX_2STEM_OUTPUT_NAMES

    def separate(self, audio: Path, output_dir: Path) -> list[Path]:
        if not _audio_separator_available():
            raise RuntimeError(
                "audio-separator is not installed. Install with: pip install audio-separator"
            )

        from audio_separator.separator import Separator

        if self._separator is None:
            log.info(
                "BS-RoFormer lazy load checkpoint=%s device=%s segment=%d",
                self.checkpoint_path.name,
                self.device,
                self.segment_size,
            )
            self._separator = Separator(**self._separator_kwargs(output_dir))
            self._separator.load_model(model_filename=self.checkpoint_path.name)
            self._model_loaded = True
        elif hasattr(self._separator, "output_dir"):
            self._separator.output_dir = str(output_dir)

        output_names = self._output_names_for_checkpoint()
        raw_outputs = self._separator.separate(str(audio), output_names)
        return [Path(p) if not isinstance(p, Path) else p for p in raw_outputs]


class BSRoFormerAdapter(BaseModelAdapter):
    """
    BS-RoFormer coarse separation via audio-separator (ViperX checkpoints).

    Design constraints:
    - Lazy initialization: weights load on first predict(), not worker startup.
    - Memory hygiene: MPS/CUDA cache cleared after each separation.
    - Standard I/O: outputs normalized to vocals/bass/drums/guitar paths.
    """

    def __init__(
        self,
        model_id: str = "bs-roformer/guitar-4stem",
        checkpoint_path: Path | None = None,
        device: str = "cpu",
        segment_size: int = 256,
    ) -> None:
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.segment_size = segment_size
        self.runtime = "local"
        self._runtime: _BSRoFormerRuntime | None = None

    def healthcheck(self) -> bool:
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return False
        return _audio_separator_available()

    async def predict(self, input_data: SeparateInput) -> SeparateOutput:
        return await asyncio.to_thread(self._separate, input_data)

    def _get_runtime(self) -> _BSRoFormerRuntime:
        if self.checkpoint_path is None:
            raise RuntimeError(f"{self.model_id}: checkpoint path is not configured")
        if self._runtime is None:
            self._runtime = _BSRoFormerRuntime.get(
                self.checkpoint_path,
                self.device,
                self.segment_size,
            )
        return self._runtime

    def _separate(self, input_data: SeparateInput) -> SeparateOutput:
        if not self.healthcheck():
            raise RuntimeError(
                f"{self.model_id}: BS-RoFormer unavailable — configure BS_ROFORMER_CHECKPOINT "
                "and install audio-separator"
            )

        output_dir = input_data.output_dir or input_data.audio.parent / "stems"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / "bs_roformer" / input_data.audio.stem / "raw"
        standard_dir = output_dir / "bs_roformer" / input_data.audio.stem / "standard"
        raw_dir.mkdir(parents=True, exist_ok=True)

        runtime = self._get_runtime()
        try:
            raw_files = runtime.separate(input_data.audio, raw_dir)
            coarse_stems = standardize_coarse_stems(raw_files, standard_dir)
        finally:
            release_accelerator_memory(self.device)

        guitar_path = coarse_stems.get("guitar")
        if guitar_path is None:
            raise FileNotFoundError(
                f"BS-RoFormer guitar stem not found after normalization under {standard_dir}. "
                f"Raw outputs: {[p.name for p in raw_files]}"
            )

        log.info(
            "BS-RoFormer separation complete stems=%s model_loaded=%s",
            list(coarse_stems.keys()),
            runtime.model_loaded,
        )

        return SeparateOutput(
            stem_path=guitar_path,
            model_id=self.model_id,
            guitar_part=input_data.guitar_part,
            coarse_stems=coarse_stems,
            isolation_method="bs_roformer_4stem",
        )

    def describe(self) -> dict[str, object]:
        base = super().describe()
        base["backend"] = "bs_roformer"
        base["checkpoint"] = str(self.checkpoint_path) if self.checkpoint_path else None
        base["device"] = self.device
        base["model_loaded"] = self._runtime.model_loaded if self._runtime else False
        base["audio_separator"] = _audio_separator_available()
        return base
