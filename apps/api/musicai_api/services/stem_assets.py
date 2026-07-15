"""Resolve stem preview assets for completed or in-progress jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from musicai_api.settings import settings

StemId = Literal[
    "input",
    "demucs",
    "isolated",
    "drums",
    "bass",
    "vocals",
    "solo_demix",
    "rhythm_demix",
]

STEM_MEDIA = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".opus": "audio/ogg",
}


class StemAssetError(Exception):
    pass


def _resolve_audio_path(work_dir: Path, raw: str | Path) -> Path | None:
    p = Path(raw)
    candidates = [p, work_dir / p, work_dir / p.name, settings.musicai_data_dir.parent / p]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved.stat().st_size > 0:
            return resolved
    return None


def _glob_fallback(work_dir: Path) -> dict | None:
    items: list[dict[str, str]] = []

    source_candidates = [work_dir / "input.wav", *sorted(work_dir.glob("*.wav"))]
    seen: set[str] = set()
    for src in source_candidates:
        if not src.exists() or src.stat().st_size == 0:
            continue
        if "stems" in src.parts or src.name in seen:
            continue
        seen.add(src.name)
        rel = str(src.resolve().relative_to(work_dir.resolve()))
        items.append({"id": "input", "label": "Исходный микс", "relative_path": rel})
        break

    demucs_matches = sorted(work_dir.glob("stems/**/guitar.wav"))
    if demucs_matches:
        rel = str(demucs_matches[0].resolve().relative_to(work_dir.resolve()))
        items.append({"id": "demucs", "label": "Guitar stem (Demucs)", "relative_path": rel})

    fallback = work_dir / "stems" / "guitar_fallback.wav"
    if fallback.exists() and fallback.stat().st_size > 0 and not demucs_matches:
        rel = str(fallback.resolve().relative_to(work_dir.resolve()))
        items.append({"id": "demucs", "label": "Guitar stem (fallback)", "relative_path": rel})

    for part in ("solo", "rhythm"):
        for suffix in (f"guitar_{part}_v2.wav", f"guitar_{part}.wav"):
            part_path = work_dir / "stems" / "parts" / suffix
            if part_path.exists() and part_path.stat().st_size > 0:
                from musicai_worker.guitar_isolation import GUITAR_PART_LABELS

                rel = str(part_path.resolve().relative_to(work_dir.resolve()))
                label = f"Изолировано: {GUITAR_PART_LABELS[part]}"
                items.append({"id": "isolated", "label": label, "relative_path": rel})
                break

    if not items:
        return None
    return {"guitar_part": "combined", "items": items}


def _read_manifest(work_dir: Path) -> dict | None:
    manifest_path = work_dir / "stems" / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    legacy = work_dir / "logs" / "separate_output.json"
    if legacy.exists():
        data = json.loads(legacy.read_text(encoding="utf-8"))
        guitar_part = data.get("guitar_part", "combined")
        items: list[dict[str, str]] = []

        input_audio = work_dir / "input.wav"
        if input_audio.exists() and input_audio.stat().st_size > 0:
            items.append(
                {
                    "id": "input",
                    "label": "Исходный микс",
                    "relative_path": "input.wav",
                }
            )
        else:
            for src in sorted(work_dir.glob("*.wav")):
                if src.stat().st_size > 0 and "stems" not in src.parts:
                    rel = str(src.resolve().relative_to(work_dir.resolve()))
                    items.append({"id": "input", "label": "Исходный микс", "relative_path": rel})
                    break

        demucs_path = _resolve_audio_path(work_dir, data.get("stem_path", ""))
        if demucs_path:
            try:
                rel = str(demucs_path.relative_to(work_dir.resolve()))
            except ValueError:
                rel = str(demucs_path.relative_to(settings.musicai_data_dir.parent.resolve()))
            items.append({"id": "demucs", "label": "Guitar stem (Demucs)", "relative_path": rel})

        isolated_path = _resolve_audio_path(work_dir, data.get("transcription_stem", ""))
        if isolated_path and (not demucs_path or isolated_path.resolve() != demucs_path.resolve()):
            from musicai_worker.guitar_isolation import GUITAR_PART_LABELS

            part_label = GUITAR_PART_LABELS.get(guitar_part, "Guitar")
            try:
                rel = str(isolated_path.resolve().relative_to(work_dir.resolve()))
            except ValueError:
                rel = str(isolated_path)
            items.append({"id": "isolated", "label": f"Изолировано: {part_label}", "relative_path": rel})

        if items:
            return {"guitar_part": guitar_part, "items": items}

    return _glob_fallback(work_dir)


def list_job_stems(work_dir: Path) -> dict | None:
    manifest = _read_manifest(work_dir)
    if not manifest:
        return None

    items = []
    for item in manifest.get("items", []):
        rel = item.get("relative_path")
        if not rel:
            continue
        candidate = (work_dir / rel).resolve()
        if not candidate.exists() or not candidate.is_file():
            continue
        if not str(candidate).startswith(str(work_dir.resolve())):
            continue
        items.append(
            {
                "id": item["id"],
                "label": item.get("label", item["id"]),
                "filename": candidate.name,
            }
        )

    if not items:
        return None
    return {"guitar_part": manifest.get("guitar_part", "combined"), "items": items}


def resolve_stem_audio(work_dir: Path, stem_id: str) -> tuple[Path, str]:
    manifest = _read_manifest(work_dir)
    if not manifest:
        raise StemAssetError("Stem manifest not found")

    match = next((item for item in manifest.get("items", []) if item.get("id") == stem_id), None)
    if not match:
        raise StemAssetError(f"Stem '{stem_id}' not found")

    rel = match.get("relative_path")
    if not rel:
        raise StemAssetError(f"Stem '{stem_id}' has no path")

    candidate = (work_dir / rel).resolve()
    work_root = work_dir.resolve()
    if not str(candidate).startswith(str(work_root)):
        raise StemAssetError("Invalid stem path")
    if not candidate.exists() or not candidate.is_file():
        raise StemAssetError(f"Stem file missing: {stem_id}")

    media_type = STEM_MEDIA.get(candidate.suffix.lower(), "application/octet-stream")
    return candidate, media_type
