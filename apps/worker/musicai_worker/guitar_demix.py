"""CASA guitar de-mixing: split Demucs guitar stem into solo vs rhythm voices."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

log = logging.getLogger("musicai.guitar_demix")

DEMIX_VERSION = "casa_v2"


@dataclass(frozen=True)
class CASADemixConfig:
    """Tunable CASA parameters — reduce 'muddy' soft-mask bleed and pitch hallucinations."""

    pitch_confidence_threshold: float = 0.85
    hpss_margin_harmonic: float = 4.0
    hpss_margin_percussive: float = 3.0
    hpss_n_fft: int = 1024
    hpss_win_length: int = 1024
    hpss_hop_length: int = 256
    spatial_correlation_threshold: float = 0.90
    spatial_frame_ms: float = 20.0
    mask_hard_threshold: float = 0.55
    wiener_beta: float = 2.0
    wiener_noise_floor: float = 1e-4
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    percussive_mix: float = 0.25
    solo_residual_subtract: float = 0.92


@dataclass(frozen=True)
class GuitarDemixResult:
    solo: Path
    rhythm: Path
    method: str
    diagnostics: dict[str, float]


def _normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_val = float(np.max(np.abs(y)))
    if max_val < 1e-8:
        return y
    return y * (peak / max_val)


def _upsample_frames(values: np.ndarray, length: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    positions = np.linspace(0, values.size - 1, num=length)
    return np.interp(positions, np.arange(values.size), values).astype(np.float32)


def _time_mask_to_stft(mask: np.ndarray, n_frames: int) -> np.ndarray:
    positions = np.linspace(0, len(mask) - 1, num=n_frames)
    return np.interp(positions, np.arange(len(mask)), mask).astype(np.float32)


def _hard_gate(mask: np.ndarray, threshold: float) -> np.ndarray:
    gated = np.where(mask >= threshold, mask, 0.0)
    return np.clip(gated, 0.0, 1.0).astype(np.float32)


def _wiener_apply(
    y: np.ndarray,
    mask: np.ndarray,
    cfg: CASADemixConfig,
) -> np.ndarray:
    """Wiener-filtered STFT reconstruction — preserves phase better than soft time-domain masks."""
    stft = librosa.stft(
        y,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )
    mag = np.abs(stft)
    phase = np.exp(1j * np.angle(stft))
    mask_stft = _time_mask_to_stft(mask, mag.shape[1])
    mask_2d = np.broadcast_to(mask_stft[np.newaxis, :], mag.shape)

    power = mag**2
    signal_power = power * (mask_2d**cfg.wiener_beta)
    noise_power = power * ((1.0 - mask_2d) ** cfg.wiener_beta) + cfg.wiener_noise_floor
    wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
    mag_est = wiener_gain * mag
    return librosa.istft(
        mag_est * phase,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        length=len(y),
    ).astype(np.float32)


def _load_stereo(path: Path | None, sr: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    y, _ = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim != 2 or y.shape[0] != 2:
        return None
    if float(np.std(y[0] - y[1])) < 1e-6:
        return None
    return y


def _pitch_vibrato_map(
    y: np.ndarray,
    sr: int,
    cfg: CASADemixConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (vibrato_score, confident_voiced_mask, mean_confidence) per sample."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E6"),
        sr=sr,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    probs = np.nan_to_num(voiced_probs, nan=0.0) if voiced_probs is not None else np.zeros_like(f0)
    voiced = np.asarray(voiced_flag if voiced_flag is not None else f0 > 0, dtype=np.float32)
    confident = (probs >= cfg.pitch_confidence_threshold).astype(np.float32)
    voiced = voiced * confident

    window = 7
    f0_smooth = uniform_filter1d(f0, size=window, mode="nearest")
    f0_std = uniform_filter1d(np.abs(f0 - f0_smooth), size=window, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        vibrato = np.where(f0_smooth > 1.0, f0_std / (f0_smooth + 1e-6), 0.0)
    vibrato = vibrato * confident
    vibrato = vibrato / (float(np.max(vibrato)) + 1e-8)

    vibrato_samples = _upsample_frames(vibrato, len(y))
    voiced_samples = _upsample_frames(voiced, len(y))
    return vibrato_samples, voiced_samples, float(np.mean(probs))


def _spatial_correlation_gates(
    stereo: np.ndarray,
    length: int,
    sr: int,
    cfg: CASADemixConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard mid/side gating from L/R phase-amplitude correlation (not soft L+R/L-R)."""
    left = stereo[0, :length]
    right = stereo[1, :length]
    frame = max(1, int(sr * cfg.spatial_frame_ms / 1000.0))
    solo_gate = np.zeros(length, dtype=np.float32)
    rhythm_gate = np.zeros(length, dtype=np.float32)

    for start in range(0, length, frame):
        end = min(start + frame, length)
        l = left[start:end]
        r = right[start:end]
        if float(np.std(l)) < 1e-8 or float(np.std(r)) < 1e-8:
            corr = 0.0
        else:
            corr = abs(float(np.corrcoef(l, r)[0, 1]))

        if corr >= cfg.spatial_correlation_threshold:
            solo_gate[start:end] = 1.0
        else:
            rhythm_gate[start:end] = 1.0

    return solo_gate, rhythm_gate


def _hpss_rhythm_percussive(y: np.ndarray, cfg: CASADemixConfig) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic/percussive split with tighter margins and shorter windows for pick attacks."""
    harmonic, percussive = librosa.effects.hpss(
        y,
        margin=(cfg.hpss_margin_harmonic, cfg.hpss_margin_percussive),
        n_fft=cfg.hpss_n_fft,
        win_length=cfg.hpss_win_length,
        hop_length=cfg.hpss_hop_length,
    )
    return harmonic.astype(np.float32), percussive.astype(np.float32)


def demix_guitar_stem(
    guitar_stem: Path,
    output_dir: Path,
    *,
    mix_path: Path | None = None,
    config: CASADemixConfig | None = None,
) -> GuitarDemixResult:
    """
    Harmonic + spatial CASA de-mix on the Demucs guitar stem (v2).

    Improvements over v1:
    - Wiener STFT filtering instead of soft time-domain masks
    - Harder HPSS margins + shorter analysis window for rhythm attacks
    - Hard spatial correlation gating (mid vs side)
    - pyin confidence threshold to suppress pitch hallucinations
    """
    cfg = config or CASADemixConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    solo_path = output_dir / f"solo_{DEMIX_VERSION}.wav"
    rhythm_path = output_dir / f"rhythm_{DEMIX_VERSION}.wav"

    y, sr = librosa.load(str(guitar_stem), sr=44100, mono=True)
    stereo = _load_stereo(mix_path, sr)

    vibrato, voiced, pitch_conf_mean = _pitch_vibrato_map(y, sr, cfg)
    harmonic, percussive = _hpss_rhythm_percussive(y, cfg)
    perc_env = uniform_filter1d(np.abs(percussive), size=max(1, int(0.020 * sr)), mode="nearest")
    perc_env /= float(perc_env.max()) + 1e-8

    solo_mask = np.clip(
        0.60 * vibrato * voiced + 0.20 * uniform_filter1d(np.abs(harmonic), 40),
        0,
        1,
    )
    rhythm_mask = np.clip(
        0.40 * (1.0 - vibrato) * voiced + 0.60 * perc_env,
        0,
        1,
    )

    spatial_used = False
    if stereo is not None:
        length = min(len(y), stereo.shape[1])
        y = y[:length]
        solo_mask = solo_mask[:length]
        rhythm_mask = rhythm_mask[:length]
        harmonic = harmonic[:length]
        percussive = percussive[:length]
        solo_gate, rhythm_gate = _spatial_correlation_gates(stereo, length, sr, cfg)
        solo_mask = np.clip(solo_mask * solo_gate, 0, 1)
        rhythm_mask = np.clip(rhythm_mask * rhythm_gate, 0, 1)
        spatial_used = True

    solo_mask = _hard_gate(solo_mask, cfg.mask_hard_threshold)
    rhythm_mask = _hard_gate(rhythm_mask, cfg.mask_hard_threshold)

    solo = _wiener_apply(y, solo_mask, cfg)
    solo = _normalize(solo)
    residual = y - solo * cfg.solo_residual_subtract
    rhythm = _wiener_apply(residual, rhythm_mask, cfg)
    rhythm = _normalize(rhythm + percussive * cfg.percussive_mix)

    sf.write(str(solo_path), solo, sr)
    sf.write(str(rhythm_path), rhythm, sr)

    diagnostics = {
        "vibrato_mean": float(np.mean(vibrato)),
        "pitch_confidence_mean": pitch_conf_mean,
        "percussive_mean": float(np.mean(perc_env)),
        "solo_rms": float(np.sqrt(np.mean(solo**2))),
        "rhythm_rms": float(np.sqrt(np.mean(rhythm**2))),
        "stereo_used": float(spatial_used),
        "solo_mask_active_ratio": float(np.mean(solo_mask > 0)),
        "rhythm_mask_active_ratio": float(np.mean(rhythm_mask > 0)),
    }
    log.info("Guitar demix v2 complete diagnostics=%s", diagnostics)

    return GuitarDemixResult(
        solo=solo_path,
        rhythm=rhythm_path,
        method="casa_wiener_hpss_spatial_v2",
        diagnostics=diagnostics,
    )
