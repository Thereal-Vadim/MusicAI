"""Tab similarity metrics vs reference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComparisonResult:
    pitch_recall: float
    pitch_precision: float
    pitch_f1: float
    fret_match_rate: float
    string_match_rate: float
    overall_similarity: float
    matched: int
    reference_count: int
    predicted_count: int
    details: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "pitch_recall": round(self.pitch_recall, 4),
            "pitch_precision": round(self.pitch_precision, 4),
            "pitch_f1": round(self.pitch_f1, 4),
            "fret_match_rate": round(self.fret_match_rate, 4),
            "string_match_rate": round(self.string_match_rate, 4),
            "overall_similarity": round(self.overall_similarity, 4),
            "matched": self.matched,
            "reference_count": self.reference_count,
            "predicted_count": self.predicted_count,
            "details": self.details,
        }


def _load_notes(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    if "notes" in data:
        return data["notes"]
    if "tracks" in data:
        notes: list[dict[str, object]] = []
        for track in data["tracks"]:
            for measure in track.get("measures", []):
                notes.extend(measure.get("notes", []))
        return notes
    return []


def _match_notes(
    reference: list[dict[str, object]],
    predicted: list[dict[str, object]],
    window_ms: float = 180.0,
) -> ComparisonResult:
    used_pred: set[int] = set()
    matches: list[tuple[dict, dict]] = []

    for ref in reference:
        ref_start = float(ref["start_ms"])
        ref_midi = int(ref.get("pitch_midi") or 0)
        best_idx = None
        best_dist = window_ms + 1
        for i, pred in enumerate(predicted):
            if i in used_pred:
                continue
            pred_start = float(pred.get("start_ms", 0))
            pred_midi = int(pred.get("pitch_midi") or 0)
            dist = abs(pred_start - ref_start)
            if dist <= window_ms and dist < best_dist and pred_midi == ref_midi:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            used_pred.add(best_idx)
            matches.append((ref, predicted[best_idx]))

    matched = len(matches)
    ref_n = len(reference)
    pred_n = len(predicted)
    recall = matched / ref_n if ref_n else 0.0
    precision = matched / pred_n if pred_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    fret_ok = sum(1 for r, p in matches if int(r.get("fret", -1)) == int(p.get("fret", -2)))
    string_ok = sum(1 for r, p in matches if int(r.get("string", -1)) == int(p.get("string", -2)))
    fret_rate = fret_ok / matched if matched else 0.0
    string_rate = string_ok / matched if matched else 0.0

    overall = 0.5 * f1 + 0.25 * fret_rate + 0.25 * string_rate

    details = [
        {
            "ref_start_ms": r["start_ms"],
            "pred_start_ms": p.get("start_ms"),
            "pitch_midi": r.get("pitch_midi"),
            "ref_fret": r.get("fret"),
            "pred_fret": p.get("fret"),
            "ref_string": r.get("string"),
            "pred_string": p.get("string"),
        }
        for r, p in matches
    ]

    return ComparisonResult(
        pitch_recall=recall,
        pitch_precision=precision,
        pitch_f1=f1,
        fret_match_rate=fret_rate,
        string_match_rate=string_rate,
        overall_similarity=overall,
        matched=matched,
        reference_count=ref_n,
        predicted_count=pred_n,
        details=details,
    )


def compare_files(reference_path: Path, predicted_path: Path, window_ms: float = 180.0) -> ComparisonResult:
    return _match_notes(_load_notes(reference_path), _load_notes(predicted_path), window_ms=window_ms)
