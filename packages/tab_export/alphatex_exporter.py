"""Convert TabDocument to alphaTex for alphaTab rendering."""

from __future__ import annotations

from collections import defaultdict

from tab_schema.models import TabDocument, TabMeasure, TabNote

DURATION_TO_ALPHATEX: list[tuple[int, str]] = [
    (240, ":16"),
    (480, ":8"),
    (960, ":4"),
    (1920, ":2"),
    (3840, ":1"),
]


def _duration_token(duration_ms: float, bpm: float) -> str:
    quarter_ms = 60_000 / bpm if bpm > 0 else 500
    ticks = duration_ms / quarter_ms * 960
    best = min(DURATION_TO_ALPHATEX, key=lambda item: abs(item[0] - ticks))
    return best[1]


def _note_effects(note: TabNote) -> str:
    if not note.technique:
        return ""
    effects: list[str] = []
    tech = note.technique
    if tech.palm_mute:
        effects.append("pm")
    if tech.vibrato:
        effects.append("v")
    if tech.ghost:
        effects.append("g")
    if tech.tie:
        effects.append("t")
    if tech.slide == "up":
        effects.append("sl")
    elif tech.slide == "down":
        effects.append("s")
    elif tech.slide == "into_from_below":
        effects.append("sb")
    elif tech.slide == "into_from_above":
        effects.append("sa")
    if not effects:
        return ""
    return "{" + " ".join(effects) + "}"


def _format_note(note: TabNote) -> str:
    fret = note.fret
    if note.technique and note.technique.ghost:
        return f"({fret}).{note.string}{_note_effects(note)}"
    return f"{fret}.{note.string}{_note_effects(note)}"


def _format_beat(notes: list[TabNote], duration_token: str) -> str:
    if len(notes) == 1:
        return f"{duration_token} {_format_note(notes[0])}"
    chord = " ".join(_format_note(n) for n in notes)
    return f"{duration_token} ({chord})"


def _measure_bar_lines(measure: TabMeasure, bpm: float) -> list[str]:
    lines: list[str] = []
    if measure.section:
        lines.append(f'\\section "{measure.section}"')
    if measure.time_signature:
        num, den = measure.time_signature
        if (num, den) == (4, 4):
            lines.append("\\ts common")
        else:
            lines.append(f"\\ts {num} {den}")
    if measure.tempo_bpm:
        lines.append(f"\\tempo {int(round(measure.tempo_bpm))}")
    elif measure.index == 0:
        lines.append(f"\\tempo {int(round(bpm))}")

    grouped: dict[float, list[TabNote]] = defaultdict(list)
    for note in sorted(measure.notes, key=lambda n: n.start_ms):
        grouped[note.start_ms].append(note)

    beat_chunks: list[str] = []
    for start_ms in sorted(grouped.keys()):
        notes = grouped[start_ms]
        duration_ms = max(n.duration_ms for n in notes)
        token = _duration_token(duration_ms, measure.tempo_bpm or bpm)
        beat_chunks.append(_format_beat(notes, token).strip())

    if beat_chunks:
        lines.append(" ".join(beat_chunks) + " |")
    else:
        lines.append("r |")
    return lines


def document_to_alphatex(document: TabDocument) -> str:
    meta = document.meta
    title = meta.title or "Untitled"
    artist = meta.artist or "MusicAI"
    lines = [f'\\title "{title}"', f'\\artist "{artist}"', f"\\tempo {int(round(meta.bpm))}", "\\defaultSystemsLayout 4", "."]

    for track in document.tracks:
        if track.name:
            lines.append(f'\\instrument "{track.name}"')
        for measure in track.measures:
            lines.extend(_measure_bar_lines(measure, meta.bpm))

    return "\n".join(lines) + "\n"
