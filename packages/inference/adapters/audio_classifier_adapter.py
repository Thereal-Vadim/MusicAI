"""AST (Audio Spectrogram Transformer) guitar timbre classifier."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import TimbreClassifyInput, TimbreClassifyOutput

log = logging.getLogger("musicai.audio_classifier")

DEFAULT_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
ANALYSIS_SECONDS = 5.0
SAMPLE_RATE = 16_000
TOP_K = 10

# MIDI General MIDI programs used by AlphaTab / Guitar Pro
MIDI_ACOUSTIC_STEEL = 25
MIDI_ELECTRIC_CLEAN = 27
MIDI_OVERDRIVE = 29
MIDI_DISTORTION = 30


@dataclass(frozen=True)
class GuitarTimbre:
    type: str
    midi_program: int


TIMBRE_ACOUSTIC = GuitarTimbre("Acoustic Guitar", MIDI_ACOUSTIC_STEEL)
TIMBRE_CLEAN = GuitarTimbre("Electric Guitar (clean)", MIDI_ELECTRIC_CLEAN)
TIMBRE_OVERDRIVE = GuitarTimbre("Overdriven Guitar", MIDI_OVERDRIVE)
TIMBRE_DISTORTION = GuitarTimbre("Distortion Guitar", MIDI_DISTORTION)

# Substring matchers against AudioSet label text (lowercased), ordered by specificity.
_LABEL_RULES: tuple[tuple[tuple[str, ...], GuitarTimbre], ...] = (
    (("acoustic guitar", "steel guitar", "folk guitar"), TIMBRE_ACOUSTIC),
    (("distortion", "heavy metal", "death metal", "thrash metal", "guitar distortion"), TIMBRE_DISTORTION),
    (("overdrive", "electric guitar", "rock"), TIMBRE_OVERDRIVE),
    (("clean guitar", "jazz guitar", "guitar"), TIMBRE_CLEAN),
)

_GUITARISH_TOKENS = (
    "guitar",
    "distortion",
    "overdrive",
    "heavy metal",
    "electric guitar",
    "acoustic guitar",
    "strum",
    "pluck",
)


def map_audioset_labels_to_timbre(
    labels: list[tuple[str, float]],
    *,
    min_confidence: float = 0.05,
) -> tuple[GuitarTimbre, str, float]:
    """Map top-k AudioSet (label, score) pairs to a guitar timbre bucket.

    Prefers guitar-related labels over unrelated argmax winners (speech, music, etc.).
    Falls back to Clean Electric when nothing useful is found.
    """
    guitarish = [
        (label, score)
        for label, score in labels
        if score >= min_confidence and any(tok in label.lower() for tok in _GUITARISH_TOKENS)
    ]
    candidates = guitarish or [(label, score) for label, score in labels if score >= min_confidence]
    if not candidates:
        return TIMBRE_CLEAN, "fallback", 0.0

    best_label, best_score = candidates[0]
    for label, score in candidates:
        lower = label.lower()
        for needles, timbre in _LABEL_RULES:
            if any(n in lower for n in needles):
                return timbre, label, float(score)

    # No explicit rule hit — default clean, keep the top guitarish label for diagnostics.
    return TIMBRE_CLEAN, best_label, float(best_score)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_device(preferred: str) -> str:
    if not _torch_available():
        return "cpu"
    import torch

    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred in {"mps", "cuda"}:
        return "cpu"
    return preferred or "cpu"


class _ASTRuntime:
    """Lazy singleton — downloads/loads AST weights on first inference only."""

    _instances: dict[str, "_ASTRuntime"] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._feature_extractor: Any | None = None
        self._model_loaded = False

    @classmethod
    def get(cls, model_name: str, device: str) -> "_ASTRuntime":
        key = f"{model_name}::{device}"
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(model_name, device)
            return cls._instances[key]

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._feature_extractor is not None:
            return self._model, self._feature_extractor

        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        log.info("AST lazy load model=%s device=%s", self.model_name, self.device)
        feature_extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
        model = ASTForAudioClassification.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()
        self._feature_extractor = feature_extractor
        self._model = model
        self._model_loaded = True
        return model, feature_extractor

    def predict_file(self, audio_path: Path) -> TimbreClassifyOutput:
        import librosa
        import torch

        model, feature_extractor = self._ensure_model()
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=ANALYSIS_SECONDS, mono=True)
        if audio.size == 0:
            timbre = TIMBRE_CLEAN
            return TimbreClassifyOutput(
                type=timbre.type,
                midi_program=timbre.midi_program,
                label="empty_audio",
                confidence=0.0,
                model_id="",
            )

        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            top_scores, top_indices = torch.topk(probs, k=min(TOP_K, probs.shape[-1]))

        id2label = model.config.id2label
        ranked: list[tuple[str, float]] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            label = str(id2label[int(idx)])
            ranked.append((label, float(score)))

        timbre, matched_label, confidence = map_audioset_labels_to_timbre(ranked)
        return TimbreClassifyOutput(
            type=timbre.type,
            midi_program=timbre.midi_program,
            label=matched_label,
            confidence=confidence,
            model_id="",
            top_labels=[{"label": lab, "score": sc} for lab, sc in ranked],
        )


class AudioClassifierAdapter(BaseModelAdapter):
    """Classifies isolated guitar audio into Acoustic / Clean / Overdrive / Distortion."""

    def __init__(
        self,
        model_id: str = "audio-classifier/ast-audioset",
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.runtime = "local"
        self.model_name = model_name
        self.device = resolve_device(device)
        self._runtime: _ASTRuntime | None = None

    def _get_runtime(self) -> _ASTRuntime:
        if self._runtime is None:
            self._runtime = _ASTRuntime.get(self.model_name, self.device)
        return self._runtime

    def healthcheck(self) -> bool:
        return _torch_available() and _transformers_available()

    def describe(self) -> dict[str, object]:
        loaded = self._runtime.model_loaded if self._runtime else False
        return {
            **super().describe(),
            "backend": "ast",
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": loaded,
        }

    async def predict(self, input_data: TimbreClassifyInput | Path | str) -> TimbreClassifyOutput:
        if isinstance(input_data, TimbreClassifyInput):
            audio_path = Path(input_data.audio)
        else:
            audio_path = Path(input_data)

        if not self.healthcheck():
            timbre = TIMBRE_CLEAN
            return TimbreClassifyOutput(
                type=timbre.type,
                midi_program=timbre.midi_program,
                label="unavailable",
                confidence=0.0,
                model_id=self.model_id,
            )

        runtime = self._get_runtime()
        result = await asyncio.to_thread(runtime.predict_file, audio_path)
        result.model_id = self.model_id
        return result
