"""Time-aligned tab comparison metrics vs ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tab_schema.models import TabDocument, TabNote

NoteMatchStatus = Literal["match", "pitch_miss", "timing_miss", "fret_miss", "string_miss", "unmatched_ref", "extra_pred"]


@dataclass
class NoteAlignment:
    status: NoteMatchStatus
    ref_start_ms: float | None = None
    pred_start_ms: float | None = None
    pitch_midi: int | None = None
    ref_fret: int | None = None
    pred_fret: int | None = None
    ref_string: int | None = None
    pred_string: int | None = None
    timing_delta_ms: float | None = None
    predicted_note_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "ref_start_ms": self.ref_start_ms,
            "pred_start_ms": self.pred_start_ms,
            "pitch_midi": self.pitch_midi,
            "ref_fret": self.ref_fret,
            "pred_fret": self.pred_fret,
            "ref_string": self.ref_string,
            "pred_string": self.pred_string,
            "timing_delta_ms": self.timing_delta_ms,
            "predicted_note_id": self.predicted_note_id,
        }


@dataclass
class TabAlignmentResult:
    pitch_recall: float
    pitch_precision: float
    pitch_f1: float
    pitch_accuracy: float
    fret_accuracy: float
    string_accuracy: float
    timing_accuracy: float
    overall_f1: float
    fret_match_rate: float
    string_match_rate: float
    overall_similarity: float
    matched: int
    reference_count: int
    predicted_count: int
    window_ms: float
    timing_tolerance_ms: float
    alignments: list[NoteAlignment] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "pitch_recall": round(self.pitch_recall, 4),
            "pitch_precision": round(self.pitch_precision, 4),
            "pitch_f1": round(self.pitch_f1, 4),
            "pitch_accuracy": round(self.pitch_accuracy, 4),
            "fret_accuracy": round(self.fret_accuracy, 4),
            "string_accuracy": round(self.string_accuracy, 4),
            "timing_accuracy": round(self.timing_accuracy, 4),
            "overall_f1": round(self.overall_f1, 4),
            "fret_match_rate": round(self.fret_match_rate, 4),
            "string_match_rate": round(self.string_match_rate, 4),
            "overall_similarity": round(self.overall_similarity, 4),
            "matched": self.matched,
            "reference_count": self.reference_count,
            "predicted_count": self.predicted_count,
            "window_ms": self.window_ms,
            "timing_tolerance_ms": self.timing_tolerance_ms,
            "alignments": [a.to_dict() for a in self.alignments],
        }


def _note_dict_from_tab(note: TabNote) -> dict[str, object]:
    return {
        "start_ms": note.start_ms,
        "duration_ms": note.duration_ms,
        "string": note.string,
        "fret": note.fret,
        "pitch_midi": note.pitch_midi,
        "id": note.id,
    }


def load_notes_from_path(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    if "notes" in data:
        return list(data["notes"])
    if "tracks" in data:
        notes: list[dict[str, object]] = []
        for track in data["tracks"]:
            for measure in track.get("measures", []):
                for note in measure.get("notes", []):
                    notes.append(dict(note))
        return notes
    document = TabDocument.model_validate(data)
    return [_note_dict_from_tab(n) for n in document.all_notes()]


def load_notes_from_document(document: TabDocument) -> list[dict[str, object]]:
    return [_note_dict_from_tab(n) for n in document.all_notes()]


def _pitch_midi(note: dict[str, object]) -> int:
    return int(note.get("pitch_midi") or 0)


def _normalize_predicted_timing(
    reference: list[dict[str, object]],
    predicted: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Linearly warp predicted onsets to reference span (handles tempo / grid drift)."""
    if len(reference) < 2 or len(predicted) < 2:
        return predicted

    ref_sorted = sorted(reference, key=lambda n: float(n["start_ms"]))
    pred_sorted = sorted(predicted, key=lambda n: float(n["start_ms"]))
    ref0 = float(ref_sorted[0]["start_ms"])
    pred0 = float(pred_sorted[0]["start_ms"])
    ref_span = float(ref_sorted[-1]["start_ms"]) - ref0
    pred_span = float(pred_sorted[-1]["start_ms"]) - pred0
    if pred_span <= 1e-6 or ref_span <= 1e-6:
        return predicted

    scale = ref_span / pred_span
    normalized: list[dict[str, object]] = []
    for pred in predicted:
        copy = dict(pred)
        copy["start_ms"] = ref0 + (float(pred["start_ms"]) - pred0) * scale
        normalized.append(copy)
    return normalized


def align_tab_notes(
    reference: list[dict[str, object]],
    predicted: list[dict[str, object]],
    *,
    window_ms: float = 180.0,
    timing_tolerance_ms: float = 80.0,
    normalize_timing: bool = True,
) -> TabAlignmentResult:
    """Greedy time-window alignment with per-note status for visualization."""
    if normalize_timing:
        predicted = _normalize_predicted_timing(reference, predicted)

    used_pred: set[int] = set()
    matches: list[tuple[dict[str, object], dict[str, object], float]] = []
    alignments: list[NoteAlignment] = []

    for ref in sorted(reference, key=lambda n: float(n["start_ms"])):
        ref_start = float(ref["start_ms"])
        ref_midi = _pitch_midi(ref)
        best_idx: int | None = None
        best_dist = window_ms + 1.0

        for i, pred in enumerate(predicted):
            if i in used_pred:
                continue
            pred_start = float(pred.get("start_ms", 0))
            pred_midi = _pitch_midi(pred)
            dist = abs(pred_start - ref_start)
            if dist <= window_ms and dist < best_dist and pred_midi == ref_midi:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            used_pred.add(best_idx)
            pred = predicted[best_idx]
            matches.append((ref, pred, best_dist))
        else:
            alignments.append(
                NoteAlignment(
                    status="unmatched_ref",
                    ref_start_ms=ref_start,
                    pitch_midi=ref_midi,
                    ref_fret=int(ref.get("fret", -1)),
                    ref_string=int(ref.get("string", -1)),
                )
            )

    pitch_matched = len(matches)
    ref_n = len(reference)
    pred_n = len(predicted)
    recall = pitch_matched / ref_n if ref_n else 0.0
    precision = pitch_matched / pred_n if pred_n else 0.0
    pitch_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    timing_ok = 0
    fret_ok = 0
    string_ok = 0
    full_match = 0

    for ref, pred, dist in matches:
        ref_fret = int(ref.get("fret", -1))
        pred_fret = int(pred.get("fret", -2))
        ref_string = int(ref.get("string", -1))
        pred_string = int(pred.get("string", -2))
        fret_match = ref_fret == pred_fret
        string_match = ref_string == pred_string
        timing_match = dist <= timing_tolerance_ms

        if fret_match:
            fret_ok += 1
        if string_match:
            string_ok += 1
        if timing_match:
            timing_ok += 1
        if fret_match and string_match and timing_match:
            full_match += 1
            status: NoteMatchStatus = "match"
        elif not timing_match:
            status = "timing_miss"
        elif not fret_match:
            status = "fret_miss"
        elif not string_match:
            status = "string_miss"
        else:
            status = "pitch_miss"

        alignments.append(
            NoteAlignment(
                status=status,
                ref_start_ms=float(ref["start_ms"]),
                pred_start_ms=float(pred.get("start_ms", 0)),
                pitch_midi=_pitch_midi(ref),
                ref_fret=ref_fret,
                pred_fret=pred_fret,
                ref_string=ref_string,
                pred_string=pred_string,
                timing_delta_ms=round(dist, 2),
                predicted_note_id=str(pred.get("id")) if pred.get("id") else None,
            )
        )

    for i, pred in enumerate(predicted):
        if i not in used_pred:
            alignments.append(
                NoteAlignment(
                    status="extra_pred",
                    pred_start_ms=float(pred.get("start_ms", 0)),
                    pitch_midi=_pitch_midi(pred),
                    pred_fret=int(pred.get("fret", -1)),
                    pred_string=int(pred.get("string", -1)),
                    predicted_note_id=str(pred.get("id")) if pred.get("id") else None,
                )
            )

    alignments.sort(
        key=lambda a: (
            a.ref_start_ms if a.ref_start_ms is not None else a.pred_start_ms or 0,
        )
    )

    matched = pitch_matched
    fret_rate = fret_ok / matched if matched else 0.0
    string_rate = string_ok / matched if matched else 0.0
    timing_rate = timing_ok / matched if matched else 0.0
    pitch_accuracy = full_match / ref_n if ref_n else 0.0
    fret_accuracy = fret_rate
    string_accuracy = string_rate
    timing_accuracy = timing_rate

    overall_f1 = (
        0.4 * pitch_f1 + 0.2 * fret_rate + 0.2 * string_rate + 0.2 * timing_rate
        if matched
        else 0.0
    )
    overall_similarity = 0.4 * pitch_f1 + 0.2 * fret_rate + 0.2 * string_rate + 0.2 * timing_rate

    return TabAlignmentResult(
        pitch_recall=recall,
        pitch_precision=precision,
        pitch_f1=pitch_f1,
        pitch_accuracy=pitch_accuracy,
        fret_accuracy=fret_accuracy,
        string_accuracy=string_accuracy,
        timing_accuracy=timing_accuracy,
        overall_f1=overall_f1,
        fret_match_rate=fret_rate,
        string_match_rate=string_rate,
        overall_similarity=overall_similarity,
        matched=matched,
        reference_count=ref_n,
        predicted_count=pred_n,
        window_ms=window_ms,
        timing_tolerance_ms=timing_tolerance_ms,
        alignments=alignments,
    )


def align_files(
    reference_path: Path,
    predicted_path: Path,
    *,
    window_ms: float = 180.0,
    timing_tolerance_ms: float = 80.0,
    normalize_timing: bool = True,
) -> TabAlignmentResult:
    return align_tab_notes(
        load_notes_from_path(reference_path),
        load_notes_from_path(predicted_path),
        window_ms=window_ms,
        timing_tolerance_ms=timing_tolerance_ms,
        normalize_timing=normalize_timing,
    )


def align_documents(
    reference: TabDocument,
    predicted: TabDocument,
    *,
    window_ms: float = 180.0,
    timing_tolerance_ms: float = 80.0,
    normalize_timing: bool = True,
) -> TabAlignmentResult:
    return align_tab_notes(
        load_notes_from_document(reference),
        load_notes_from_document(predicted),
        window_ms=window_ms,
        timing_tolerance_ms=timing_tolerance_ms,
        normalize_timing=normalize_timing,
    )
