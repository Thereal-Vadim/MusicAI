"""Import Guitar Pro files into TabDocument."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import BinaryIO

from judge.rules import midi_to_name
from tab_schema.models import (
    JudgeResult,
    NoteConfidence,
    SourceMeta,
    TabDocument,
    TabMeasure,
    TabMeta,
    TabNote,
    TabTrack,
)

TICKS_PER_QUARTER = 960


def _ticks_to_ms(ticks: int, bpm: float) -> float:
    if bpm <= 0:
        bpm = 120.0
    return ticks / TICKS_PER_QUARTER * (60_000 / bpm)


def _open_string_midi(track) -> dict[int, int]:
    return {string.number: int(string.value) for string in track.strings}


def _duration_ms(beat, bpm: float) -> float:
    ticks = int(getattr(beat.duration, "time", 0) or 0)
    if ticks <= 0:
        value = int(getattr(beat.duration, "value", 4) or 4)
        ticks = TICKS_PER_QUARTER * 4 // max(value, 1)
    return _ticks_to_ms(ticks, bpm)


def gp5_to_document(
    path: Path | str | BinaryIO,
    *,
    track_index: int = 0,
    job_id: str | None = None,
    max_measures: int | None = None,
) -> TabDocument:
    """Parse a GP3–GP7 file into TabDocument (guitar track)."""
    import guitarpro

    if isinstance(path, (str, Path)):
        parsed_path = Path(path)
        song = guitarpro.parse(str(parsed_path))
        filename = parsed_path.name
    else:
        song = guitarpro.parse(path)
        filename = "import.gp5"
    if track_index >= len(song.tracks):
        raise ValueError(f"Track index {track_index} out of range ({len(song.tracks)} tracks)")

    track = song.tracks[track_index]
    bpm = float(song.tempo or 120)
    open_strings = _open_string_midi(track)
    tuning = [midi_to_name(open_strings[i]) for i in range(6, 0, -1) if i in open_strings]

    notes: list[TabNote] = []
    cursor_ms = 0.0
    measure_start_ms = 0.0
    measures: list[TabMeasure] = []

    for measure_idx, measure in enumerate(track.measures):
        if max_measures is not None and measure_idx >= max_measures:
            break
        measure_notes: list[TabNote] = []
        measure_start_ms = cursor_ms

        for voice in measure.voices:
            for beat in voice.beats:
                if not beat.notes:
                    cursor_ms += _duration_ms(beat, bpm)
                    continue

                duration_ms = _duration_ms(beat, bpm)
                beat_start = cursor_ms

                for gp_note in beat.notes:
                    string_num = int(gp_note.string)
                    fret = int(gp_note.value)
                    open_midi = open_strings.get(string_num, 40)
                    pitch_midi = open_midi + fret
                    note = TabNote(
                        id=f"gp-{uuid.uuid4().hex[:8]}",
                        pitch=midi_to_name(pitch_midi),
                        start_ms=round(beat_start, 2),
                        duration_ms=round(duration_ms, 2),
                        string=string_num,
                        fret=fret,
                        pitch_midi=pitch_midi,
                        confidence=NoteConfidence(audio=1.0, overall=1.0),
                        judge=JudgeResult(),
                    )
                    notes.append(note)
                    measure_notes.append(note)

                cursor_ms += duration_ms

        if measure_notes:
            header = song.measureHeaders[measure_idx] if measure_idx < len(song.measureHeaders) else None
            section = header.marker.title if header and header.marker else None
            measures.append(
                TabMeasure(
                    index=measure_idx,
                    start_ms=round(measure_start_ms, 2),
                    section=section,
                    notes=sorted(measure_notes, key=lambda n: n.start_ms),
                )
            )

    return TabDocument(
        job_id=job_id,
        meta=TabMeta(
            title=song.title or None,
            artist=song.artist or None,
            album=song.album or None,
            bpm=bpm,
            tuning=tuning or ["E2", "A2", "D3", "G3", "B3", "E4"],
            source=SourceMeta(type="upload", filename=filename),
        ),
        tracks=[
            TabTrack(
                name=track.name or "Guitar",
                measures=measures or [TabMeasure(index=0, start_ms=0, notes=notes)],
            )
        ],
    )


def document_to_reference_json(document: TabDocument, *, source: str | None = None) -> dict[str, object]:
    """Flatten TabDocument to ground-truth JSON (compatible with compare_tabs)."""
    notes: list[dict[str, object]] = []
    for note in document.all_notes():
        notes.append(
            {
                "start_ms": note.start_ms,
                "duration_ms": note.duration_ms,
                "string": note.string,
                "fret": note.fret,
                "pitch_midi": note.pitch_midi,
            }
        )
    notes.sort(key=lambda n: float(n["start_ms"]))
    return {
        "title": document.meta.title,
        "artist": document.meta.artist,
        "tuning": document.meta.tuning,
        "bpm": document.meta.bpm,
        "source": source,
        "notes": notes,
    }


def gp5_to_reference_json(
    path: Path | str,
    *,
    track_index: int = 0,
    source: str | None = None,
    max_measures: int | None = None,
) -> dict[str, object]:
    document = gp5_to_document(path, track_index=track_index, max_measures=max_measures)
    return document_to_reference_json(document, source=source or str(path))


def export_reference_json(path: Path | str, output_path: Path, **kwargs: object) -> Path:
    payload = gp5_to_reference_json(path, **kwargs)  # type: ignore[arg-type]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path
