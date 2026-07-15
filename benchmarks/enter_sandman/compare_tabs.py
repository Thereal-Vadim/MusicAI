"""Tab similarity metrics vs reference (Enter Sandman and ground truth)."""

from __future__ import annotations

from pathlib import Path

from tab_schema.alignment import TabAlignmentResult, align_files, align_tab_notes, load_notes_from_path

# Backward-compatible alias
ComparisonResult = TabAlignmentResult


def _match_notes(
    reference: list[dict[str, object]],
    predicted: list[dict[str, object]],
    window_ms: float = 180.0,
) -> TabAlignmentResult:
    return align_tab_notes(reference, predicted, window_ms=window_ms)


def compare_files(reference_path: Path, predicted_path: Path, window_ms: float = 180.0) -> TabAlignmentResult:
    return align_files(reference_path, predicted_path, window_ms=window_ms)


__all__ = [
    "ComparisonResult",
    "compare_files",
    "align_tab_notes",
    "load_notes_from_path",
]
