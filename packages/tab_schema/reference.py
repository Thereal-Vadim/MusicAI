"""Reference tab scoring vs official Songsterr excerpts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tab_schema.models import QualityMeta, TabDocument, TabNote

SONGSTERR_ENTER_SANDMAN_URL = (
    "https://www.songsterr.com/a/wsa/metallica-enter-sandman-official-tab-s3787442"
)


@dataclass(frozen=True)
class ReferenceProfile:
    id: str
    url: str
    path: Path
    title_keywords: tuple[str, ...]
    artist_keywords: tuple[str, ...]
    window_ms: float = 200.0
    mismatch_penalty: float = 0.18
    match_bonus: float = 0.05
    snap_match_bonus: float = 0.10


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


REFERENCE_PROFILES: tuple[ReferenceProfile, ...] = (
    ReferenceProfile(
        id="enter_sandman_intro",
        url=SONGSTERR_ENTER_SANDMAN_URL,
        path=_project_root() / "benchmarks" / "enter_sandman" / "reference_intro.json",
        title_keywords=("enter sandman",),
        artist_keywords=("metallica",),
    ),
)


def detect_reference_profile(document: TabDocument) -> ReferenceProfile | None:
    return resolve_reference_profile(document.meta.title, document.meta.artist)


def resolve_reference_profile(
    title: str | None, artist: str | None
) -> ReferenceProfile | None:
    title_l = (title or "").lower()
    artist_l = (artist or "").lower()
    for profile in REFERENCE_PROFILES:
        if not profile.path.exists():
            continue
        title_ok = any(kw in title_l for kw in profile.title_keywords) if title_l else False
        artist_ok = any(kw in artist_l for kw in profile.artist_keywords) if artist_l else False
        if title_ok or artist_ok:
            return profile
    return None


def load_reference_notes(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    return list(data.get("notes", []))


def _note_pitch_midi(note: TabNote | dict[str, object]) -> int:
    if isinstance(note, TabNote):
        if note.pitch_midi is not None:
            return note.pitch_midi
        from judge.rules import name_to_midi

        return name_to_midi(note.pitch)
    return int(note.get("pitch_midi") or 0)


def _note_matches_ref(note: TabNote, ref: dict[str, object]) -> bool:
    return (
        _note_pitch_midi(note) == int(ref.get("pitch_midi") or 0)
        and note.fret == int(ref.get("fret", -1))
        and note.string == int(ref.get("string", -1))
    )


def _closest_reference(
    note: TabNote, references: list[dict[str, object]], window_ms: float
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    best_dist = window_ms + 1.0
    for ref in references:
        dist = abs(float(ref["start_ms"]) - note.start_ms)
        if dist <= window_ms and dist < best_dist:
            best_dist = dist
            best = ref
    return best


def apply_reference_scoring(
    notes: list[TabNote],
    profile: ReferenceProfile,
) -> tuple[list[TabNote], dict[str, object]]:
    """Adjust confidence: bonus on match, penalty only when differing from Songsterr ref."""
    references = load_reference_notes(profile.path)
    if not references:
        return notes, {"reference_id": profile.id, "reference_url": profile.url}

    ref_matched = 0
    mismatch_count = 0
    adjusted_count = 0

    for note in notes:
        ref = _closest_reference(note, references, profile.window_ms)
        if ref is None:
            continue

        adjusted_count += 1
        note.flags = [f for f in note.flags if f not in {"reference_match", "reference_mismatch"}]

        if _note_matches_ref(note, ref):
            note.flags.append("reference_match")
            ref_matched += 1
            bonus = profile.snap_match_bonus if note.judge.snapped else profile.match_bonus
            note.confidence.overall = min(1.0, round(note.confidence.overall + bonus, 4))
        else:
            note.flags.append("reference_mismatch")
            mismatch_count += 1
            note.confidence.overall = max(
                0.0, round(note.confidence.overall - profile.mismatch_penalty, 4)
            )

    ref_total = len(references)
    summary = {
        "reference_id": profile.id,
        "reference_url": profile.url,
        "reference_notes": ref_total,
        "reference_matched": ref_matched,
        "reference_mismatch_count": mismatch_count,
        "reference_match_pct": round(ref_matched / ref_total, 4) if ref_total else 0.0,
        "notes_in_reference_window": adjusted_count,
    }
    return notes, summary


def merge_reference_into_quality(quality: QualityMeta, summary: dict[str, object]) -> QualityMeta:
    return quality.model_copy(
        update={
            "reference_url": summary.get("reference_url"),
            "reference_match_pct": summary.get("reference_match_pct"),
            "reference_mismatch_count": summary.get("reference_mismatch_count", 0),
        }
    )
