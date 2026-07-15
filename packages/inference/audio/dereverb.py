"""Spectral dereverberation for guitar stems (onset-preserving)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("musicai.dereverb")


def spectral_dereverb(
    y: np.ndarray,
    sr: int,
    *,
    strength: float = 0.65,
    decay_ms: float = 80.0,
    floor: float = 0.12,
    transient_mix: float = 0.18,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Suppress late reverberation / delay tails while keeping note attacks.

    Uses a causal decaying magnitude envelope as a late-reverb proxy, applies a
    Wiener-like gain, then lightly remixes HPSS percussive residual so onsets
    stay sharp for Basic Pitch.
    """
    import librosa

    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y

    strength = float(np.clip(strength, 0.0, 1.0))
    floor = float(np.clip(floor, 0.0, 1.0))
    transient_mix = float(np.clip(transient_mix, 0.0, 0.5))

    if strength <= 0.0:
        return y

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    hop_sec = hop_length / float(sr)
    decay_sec = max(decay_ms, 1.0) / 1000.0
    alpha = float(np.exp(-hop_sec / decay_sec))

    env = np.empty_like(mag)
    env[:, 0] = mag[:, 0]
    for t in range(1, mag.shape[1]):
        env[:, t] = np.maximum(mag[:, t], alpha * env[:, t - 1])

    late = np.clip(env - mag, 0.0, None)
    dry_power = mag * mag
    late_power = late * late
    eps = 1e-8
    gain = dry_power / (dry_power + strength * late_power + eps)
    gain = np.maximum(gain, floor)
    mag_clean = mag * gain

    cleaned = librosa.istft(
        mag_clean * np.exp(1j * phase),
        hop_length=hop_length,
        length=len(y),
    ).astype(np.float32)

    if transient_mix > 0:
        _, percussive = librosa.effects.hpss(y, margin=2.0)
        cleaned = (1.0 - transient_mix) * cleaned + transient_mix * percussive.astype(np.float32)

    peak_in = float(np.max(np.abs(y)) + eps)
    peak_out = float(np.max(np.abs(cleaned)) + eps)
    cleaned *= peak_in / peak_out
    return np.clip(cleaned, -1.0, 1.0).astype(np.float32)


def dereverb_file(
    input_path: Path,
    output_path: Path,
    *,
    strength: float = 0.65,
    decay_ms: float = 80.0,
    floor: float = 0.12,
    transient_mix: float = 0.18,
    sr: int = 44100,
) -> dict[str, float]:
    """Load audio, apply spectral dereverb, write WAV. Returns diagnostics."""
    import librosa
    import soundfile as sf

    y, file_sr = librosa.load(str(input_path), sr=sr, mono=True)
    cleaned = spectral_dereverb(
        y,
        file_sr,
        strength=strength,
        decay_ms=decay_ms,
        floor=floor,
        transient_mix=transient_mix,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), cleaned, file_sr)

    # Rough energy ratio: how much late energy was removed
    reduction = float(np.clip(1.0 - (np.linalg.norm(cleaned) / (np.linalg.norm(y) + 1e-8)), 0.0, 1.0))
    log.info(
        "Dereverb wrote %s strength=%.2f energy_reduction=%.3f",
        output_path.name,
        strength,
        reduction,
    )
    return {
        "strength": float(strength),
        "decay_ms": float(decay_ms),
        "floor": float(floor),
        "transient_mix": float(transient_mix),
        "energy_reduction": reduction,
        "duration_sec": float(len(cleaned) / file_sr),
    }
