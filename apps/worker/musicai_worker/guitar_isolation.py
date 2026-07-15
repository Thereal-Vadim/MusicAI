"""Post-process Demucs guitar stem into solo or rhythm emphasis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

log = logging.getLogger("musicai.guitar_isolation")

GuitarPart = Literal["combined", "solo", "rhythm"]

GUITAR_PART_LABELS: dict[GuitarPart, str] = {
    "combined": "Guitar",
    "solo": "Solo Guitar",
    "rhythm": "Rhythm Guitar",
}

ISOLATION_VERSION = "v2"


@dataclass(frozen=True)
class RhythmIsolationConfig:
    """Tuning knobs for rhythm guitar post-processing."""

    side_blend: float = 0.68
    mid_suppress: float = 0.42
    percussive_weight: float = 0.74
    harmonic_weight: float = 0.26
    hpss_margin: tuple[float, float] = (4.5, 1.15)
    bandpass_hz: tuple[float, float] = (120.0, 2800.0)
    high_cut_hz: float = 4200.0
    low_cut_hz: float = 90.0
    onset_boost: float = 0.55
    lead_suppress_db: float = 6.0


RHYTHM_CFG = RhythmIsolationConfig()


def _normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_val = float(np.max(np.abs(y)))
    if max_val < 1e-8:
        return y
    return y * (peak / max_val)


def _smooth_envelope(y: np.ndarray, sr: int, window_ms: float = 35.0) -> np.ndarray:
    window = max(1, int(sr * window_ms / 1000.0))
    env = np.abs(y)
    return uniform_filter1d(env, size=window, mode="nearest")


def _onset_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    strength = librosa.onset.onset_strength(y=y, sr=sr)
    if strength.size == 0:
        return np.zeros_like(y)
    positions = np.linspace(0, strength.size - 1, num=len(y))
    env = np.interp(positions, np.arange(strength.size), strength)
    env /= float(env.max()) + 1e-8
    return env


def _load_mono(path: Path, sr: int = 44100) -> tuple[np.ndarray, int]:
    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    return y, loaded_sr


def _load_stereo_if_available(path: Path | None, sr: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    y, _ = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim != 2 or y.shape[0] != 2:
        return None
    if float(np.std(y[0] - y[1])) < 1e-6:
        return None
    return y


def _apply_side_rhythm_mask(
    guitar: np.ndarray,
    stereo: np.ndarray,
    sr: int,
    cfg: RhythmIsolationConfig,
) -> np.ndarray:
    """Boost panned (L/R) content; rhythm doubles are usually off-center in rock/metal."""
    length = min(guitar.shape[0], stereo.shape[1])
    guitar = guitar[:length]
    left = stereo[0, :length]
    right = stereo[1, :length]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    side_env = _smooth_envelope(side, sr, window_ms=45.0)
    mid_env = _smooth_envelope(mid, sr, window_ms=45.0)
    side_env /= float(side_env.max()) + 1e-8
    mid_env /= float(mid_env.max()) + 1e-8

    # Prefer side (panned rhythm) over mid (vocals, bass, centered lead).
    mask = (1.0 - cfg.mid_suppress) + cfg.side_blend * side_env
    mask *= np.clip(1.15 - 0.55 * mid_env, 0.45, 1.15)
    return guitar * mask


def _suppress_thin_lead_lines(y: np.ndarray, sr: int, suppress_db: float) -> np.ndarray:
    """Attenuate sparse high-magnitude bins typical of single-note leads."""
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    mag = np.abs(stft)
    if mag.size == 0:
        return y
    ref = np.median(mag, axis=1, keepdims=True)
    ratio = mag / (ref + 1e-8)
    gain = np.where(ratio > 2.2, 10 ** (-suppress_db / 20.0), 1.0)
    cleaned = librosa.istft(stft * gain, hop_length=512, length=len(y))
    return cleaned.astype(np.float32)


def _filter_band(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    sos = butter(2, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, y)


def _low_high_shelf(y: np.ndarray, sr: int, low_cut: float, high_cut: float) -> np.ndarray:
    sos_hp = butter(2, low_cut, btype="highpass", fs=sr, output="sos")
    sos_lp = butter(2, high_cut, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos_lp, sosfilt(sos_hp, y))


def _solo_emphasis(y: np.ndarray, sr: int, stereo: np.ndarray | None = None) -> np.ndarray:
    if stereo is not None:
        length = min(y.shape[0], stereo.shape[1])
        y = y[:length]
        mid = (stereo[0, :length] + stereo[1, :length]) * 0.5
        mid_env = _smooth_envelope(mid, sr)
        mid_env /= float(mid_env.max()) + 1e-8
        y = y * (0.45 + 0.55 * mid_env)

    harmonic, percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
    mixed = 0.88 * harmonic + 0.12 * percussive
    mixed = librosa.effects.preemphasis(mixed, coef=0.97)
    env = _onset_envelope(mixed, sr)
    mixed *= np.clip(1.0 - 0.55 * env, 0.35, 1.0)
    return _normalize(mixed)


def _rhythm_emphasis(
    y: np.ndarray,
    sr: int,
    stereo: np.ndarray | None = None,
    cfg: RhythmIsolationConfig = RHYTHM_CFG,
) -> np.ndarray:
    if stereo is not None:
        y = _apply_side_rhythm_mask(y, stereo, sr, cfg)

    y = _low_high_shelf(y, sr, cfg.low_cut_hz, cfg.high_cut_hz)
    harmonic, percussive = librosa.effects.hpss(y, margin=cfg.hpss_margin)
    mixed = cfg.percussive_weight * percussive + cfg.harmonic_weight * harmonic
    mixed = _filter_band(mixed, sr, cfg.bandpass_hz[0], cfg.bandpass_hz[1])
    mixed = _suppress_thin_lead_lines(mixed, sr, cfg.lead_suppress_db)

    env = _onset_envelope(mixed, sr)
    mixed *= np.clip(0.72 + cfg.onset_boost * env, 0.72, 1.45)
    return _normalize(mixed)


def isolate_guitar_part(
    source: Path,
    part: GuitarPart,
    output_dir: Path,
    *,
    mix_path: Path | None = None,
) -> Path:
    """Return path to audio used for transcription (may equal source for combined)."""
    if part == "combined":
        return source

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"guitar_{part}_{ISOLATION_VERSION}.wav"
    legacy_path = output_dir / f"guitar_{part}.wav"

    log.info("Isolating guitar part=%s from %s mix=%s", part, source.name, mix_path)
    y, sr = _load_mono(source)
    stereo = _load_stereo_if_available(mix_path, sr)

    if part == "solo":
        processed = _solo_emphasis(y, sr, stereo)
    else:
        processed = _rhythm_emphasis(y, sr, stereo)

    sf.write(str(out_path), processed, sr)
    if legacy_path.exists():
        legacy_path.unlink(missing_ok=True)
    log.info(
        "Wrote guitar part=%s path=%s stereo=%s",
        part,
        out_path,
        stereo is not None,
    )
    return out_path
