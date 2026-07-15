"""Manifest of stem audio files for preview/download."""

from __future__ import annotations

import json
from pathlib import Path

from musicai_worker.guitar_isolation import GUITAR_PART_LABELS, GuitarPart

MANIFEST_NAME = "manifest.json"

COARSE_LABELS = {
    "drums": "Ударные (Demucs)",
    "bass": "Бас (Demucs)",
    "vocals": "Вокал (Demucs)",
    "guitar": "Guitar stem (Demucs)",
}


def _relative(work_dir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(work_dir.resolve()))


def write_stems_manifest(
    work_dir: Path,
    *,
    guitar_part: GuitarPart,
    demucs_stem: Path,
    transcription_stem: Path,
    input_audio: Path,
    coarse_stems: dict[str, Path] | None = None,
    solo_demix: Path | None = None,
    rhythm_demix: Path | None = None,
) -> Path:
    stems_dir = work_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stems_dir / MANIFEST_NAME

    items: list[dict[str, str]] = []
    if input_audio.exists() and input_audio.stat().st_size > 0:
        items.append(
            {
                "id": "input",
                "label": "Исходный микс",
                "relative_path": _relative(work_dir, input_audio),
            }
        )

    for stem_id, label in COARSE_LABELS.items():
        path = (coarse_stems or {}).get(stem_id)
        if path and path.exists() and path.stat().st_size > 0:
            items.append(
                {
                    "id": stem_id if stem_id != "guitar" else "demucs",
                    "label": label,
                    "relative_path": _relative(work_dir, path),
                }
            )

    if demucs_stem.exists() and not any(i["id"] == "demucs" for i in items):
        items.append(
            {
                "id": "demucs",
                "label": "Guitar stem (Demucs)",
                "relative_path": _relative(work_dir, demucs_stem),
            }
        )

    if solo_demix and solo_demix.exists() and solo_demix.stat().st_size > 0:
        items.append(
            {
                "id": "solo_demix",
                "label": "Solo Guitar (CASA demix)",
                "relative_path": _relative(work_dir, solo_demix),
            }
        )
    if rhythm_demix and rhythm_demix.exists() and rhythm_demix.stat().st_size > 0:
        items.append(
            {
                "id": "rhythm_demix",
                "label": "Rhythm Guitar (CASA demix)",
                "relative_path": _relative(work_dir, rhythm_demix),
            }
        )

    same_as_demucs = demucs_stem.resolve() == transcription_stem.resolve()
    if transcription_stem.exists() and not same_as_demucs:
        part_label = GUITAR_PART_LABELS.get(guitar_part, "Guitar")
        items.append(
            {
                "id": "isolated",
                "label": f"Транскрипция: {part_label}",
                "relative_path": _relative(work_dir, transcription_stem),
            }
        )

    payload = {
        "guitar_part": guitar_part,
        "pipeline": "demucs+casa+playability",
        "items": items,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
