"""Zero-cost demix validation via playability + polyphony heuristics."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from judge.rules import validate_simultaneous_notes


@dataclass(frozen=True)
class DemixValidationReport:
    passed: bool
    solo_max_polyphony: int
    rhythm_max_polyphony: int
    merged_max_polyphony: int
    leakage_score: float
    solo_playability_score: float
    rhythm_playability_score: float
    recommendation: str
    used_fallback: bool = False

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "passed": self.passed,
            "solo_max_polyphony": self.solo_max_polyphony,
            "rhythm_max_polyphony": self.rhythm_max_polyphony,
            "merged_max_polyphony": self.merged_max_polyphony,
            "leakage_score": round(self.leakage_score, 4),
            "solo_playability_score": round(self.solo_playability_score, 4),
            "rhythm_playability_score": round(self.rhythm_playability_score, 4),
            "recommendation": self.recommendation,
            "used_fallback": self.used_fallback,
        }


def _estimate_polyphony(y: np.ndarray, sr: int, window_ms: float = 30.0) -> int:
    """Lightweight pitch-cluster count per window (no ML)."""
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    if chroma.size == 0:
        return 0

    frames_per_window = max(1, int((window_ms / 1000.0) * sr / hop))
    max_notes = 0
    for start in range(0, chroma.shape[1], frames_per_window):
        frame = chroma[:, start : start + frames_per_window]
        if frame.size == 0:
            continue
        energy = frame.mean(axis=1)
        threshold = float(energy.max()) * 0.35
        count = int(np.sum(energy >= threshold))
        max_notes = max(max_notes, count)
    return max_notes


def _playability_score(max_polyphony: int, max_allowed: int = 4) -> float:
    if max_polyphony <= max_allowed:
        return 1.0
    return max(0.0, 1.0 - (max_polyphony - max_allowed) * 0.2)


def validate_guitar_demix(
    solo: np.ndarray,
    rhythm: np.ndarray,
    sr: int,
    *,
    max_simultaneous: int = 4,
    target_part: str = "combined",
) -> DemixValidationReport:
    solo_poly = _estimate_polyphony(solo, sr)
    rhythm_poly = _estimate_polyphony(rhythm, sr)
    merged_poly = _estimate_polyphony(solo + rhythm, sr)

    min_len = min(len(solo), len(rhythm))
    if min_len > 0:
        solo_trim = solo[:min_len]
        rhythm_trim = rhythm[:min_len]
        if float(np.std(solo_trim)) > 1e-8 and float(np.std(rhythm_trim)) > 1e-8:
            corr = abs(float(np.corrcoef(solo_trim, rhythm_trim)[0, 1]))
        else:
            corr = 1.0
    else:
        corr = 1.0
    leakage = corr

    solo_score = _playability_score(solo_poly, max_simultaneous)
    rhythm_score = _playability_score(rhythm_poly, max_simultaneous)
    merged_ok = validate_simultaneous_notes(merged_poly, max_simultaneous)

    passed = merged_ok and leakage < 0.82 and min(solo_score, rhythm_score) > 0.4

    if target_part == "solo":
        recommendation = "solo_demix" if solo_score >= rhythm_score else "rhythm_demix"
    elif target_part == "rhythm":
        recommendation = "rhythm_demix" if rhythm_score >= solo_score else "solo_demix"
    else:
        recommendation = "combined"

    return DemixValidationReport(
        passed=passed,
        solo_max_polyphony=solo_poly,
        rhythm_max_polyphony=rhythm_poly,
        merged_max_polyphony=merged_poly,
        leakage_score=leakage,
        solo_playability_score=solo_score,
        rhythm_playability_score=rhythm_score,
        recommendation=recommendation,
    )
