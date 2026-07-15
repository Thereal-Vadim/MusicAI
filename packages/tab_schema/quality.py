"""Compute quality metrics from judged notes."""

from __future__ import annotations

from tab_schema.models import QualityMeta, TabNote

HIGH_CONFIDENCE_THRESHOLD = 0.95

BLOCKING_FLAGS = {
    "unplayable_position",
    "too_many_simultaneous_notes",
    "chord_span_exceeded",
    "temporal_violation",
}


def is_high_confidence_note(note: TabNote) -> bool:
    """Trusted output: high overall and no Songsterr mismatch."""
    if note.confidence.overall < HIGH_CONFIDENCE_THRESHOLD:
        return False
    if "reference_mismatch" in note.flags:
        return False
    if note.judge.flags and set(note.judge.flags) & BLOCKING_FLAGS:
        return False
    return True


def is_conflict_note(note: TabNote) -> bool:
    if "reference_mismatch" in note.flags:
        return True
    if note.judge.flags and set(note.judge.flags) & BLOCKING_FLAGS:
        return True
    if not note.judge.in_scale and not note.judge.snapped:
        return True
    return False


def compute_quality_metrics(notes: list[TabNote], key_confidence: float = 0.0) -> QualityMeta:
    total = len(notes)
    if total == 0:
        return QualityMeta(key_confidence=key_confidence)

    snapped = sum(1 for n in notes if n.judge.snapped)
    high = sum(1 for n in notes if is_high_confidence_note(n))
    conflicts = sum(1 for n in notes if is_conflict_note(n))
    mean_overall = sum(n.confidence.overall for n in notes) / total
    ref_mismatch = sum(1 for n in notes if "reference_mismatch" in n.flags)

    return QualityMeta(
        notes_total=total,
        snapped_count=snapped,
        high_confidence_count=high,
        conflict_count=conflicts,
        snapped_pct=round(snapped / total, 4),
        high_confidence_pct=round(high / total, 4),
        conflict_pct=round(conflicts / total, 4),
        mean_overall=round(mean_overall, 4),
        key_confidence=round(key_confidence, 4),
        reference_mismatch_count=ref_mismatch,
    )
