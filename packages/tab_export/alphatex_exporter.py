"""Convert TabDocument to alphaTex for alphaTab rendering."""

from __future__ import annotations

import re
from collections import defaultdict

from tab_schema.models import TabDocument, TabMeasure, TabNote

TICKS_PER_QUARTER = 960

# alphaTex duration token → ticks (quarter=960)
DURATION_TICKS: list[tuple[int, str]] = [
    (120, ":64"),
    (240, ":32"),
    (480, ":16"),
    (960, ":8"),
    (1920, ":4"),
    (3840, ":2"),
    (7680, ":1"),
]

DEFAULT_TIME_SIGNATURE = (4, 4)


def _escape_tex(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _normalize_time_signature(ts: tuple[int, int] | None) -> tuple[int, int]:
    if not ts:
        return DEFAULT_TIME_SIGNATURE
    num, den = ts
    if num <= 0 or den <= 0:
        return DEFAULT_TIME_SIGNATURE
    # 1/4 was used historically in demo data but breaks alphaTab bar capacity.
    if num == 1 and den == 4:
        return DEFAULT_TIME_SIGNATURE
    return num, den


def _measure_capacity_ms(bpm: float, time_sig: tuple[int, int]) -> float:
    num, den = time_sig
    quarter_ms = 60_000 / bpm if bpm > 0 else 500
    return quarter_ms * num * (4 / den)


def _duration_token(duration_ms: float, bpm: float) -> str:
    quarter_ms = 60_000 / bpm if bpm > 0 else 500
    ticks = max(1, duration_ms / quarter_ms * TICKS_PER_QUARTER)
    best = min(DURATION_TICKS, key=lambda item: abs(item[0] - ticks))
    return best[1]


def _duration_ms_for_token(token: str, bpm: float) -> float:
    quarter_ms = 60_000 / bpm if bpm > 0 else 500
    ticks = next(t for t, tok in DURATION_TICKS if tok == token)
    return ticks / TICKS_PER_QUARTER * quarter_ms


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


def _format_note(note: TabNote) -> str | None:
    if note.string < 1 or note.string > 6:
        return None
    if note.fret < 0 or note.fret > 24:
        return None
    fret = int(note.fret)
    if note.technique and note.technique.ghost:
        return f"({fret}).{note.string}{_note_effects(note)}"
    return f"{fret}.{note.string}{_note_effects(note)}"


def _format_beat(notes: list[TabNote], duration_token: str) -> str | None:
    rendered = [part for n in notes if (part := _format_note(n))]
    if not rendered:
        return None
    if len(rendered) == 1:
        return f"{duration_token} {rendered[0]}"
    return f"{duration_token} ({' '.join(rendered)})"


def _time_signature_line(time_sig: tuple[int, int]) -> str:
    num, den = time_sig
    if (num, den) == (4, 4):
        return "\\ts common"
    return f"\\ts {num} {den}"


def _build_bars(
    grouped_beats: list[tuple[float, list[TabNote]]],
    *,
    bar_capacity_ms: float,
    bpm: float,
) -> list[str]:
    """Build one or more valid 4/4 bars from beat groups, splitting on overflow."""
    if not grouped_beats:
        return [":4 r |"]

    bars: list[str] = []
    chunks: list[str] = []
    cursor_ms = 0.0

    def flush_bar(pad: bool = True) -> None:
        nonlocal chunks, cursor_ms
        if pad and cursor_ms < bar_capacity_ms - 1e-3:
            chunks.append(f"{_duration_token(bar_capacity_ms - cursor_ms, bpm)} r")
        if chunks:
            bars.append(" ".join(chunks) + " |")
        chunks = []
        cursor_ms = 0.0

    for rel_start, notes in grouped_beats:
        rel_start = max(0.0, rel_start)
        if rel_start > cursor_ms + 1e-3:
            gap = rel_start - cursor_ms
            gap_token = _duration_token(gap, bpm)
            gap_ms = _duration_ms_for_token(gap_token, bpm)
            if cursor_ms + gap_ms > bar_capacity_ms + 1e-3:
                flush_bar()
                rel_start = 0.0
                gap = rel_start - cursor_ms
                gap_token = _duration_token(gap, bpm)
                gap_ms = _duration_ms_for_token(gap_token, bpm)
            if gap_ms > 1e-3:
                chunks.append(f"{gap_token} r")
                cursor_ms += gap_ms

        duration_ms = max(n.duration_ms for n in notes)
        token = _duration_token(duration_ms, bpm)
        beat_ms = _duration_ms_for_token(token, bpm)
        beat = _format_beat(notes, token)
        if not beat:
            continue
        if cursor_ms + beat_ms > bar_capacity_ms + 1e-3:
            flush_bar()
        chunks.append(beat)
        cursor_ms += beat_ms
        if cursor_ms >= bar_capacity_ms - 1e-3:
            flush_bar(pad=False)

    if chunks:
        flush_bar()
    return bars or [":4 r |"]


def _bars_from_measure(measure: TabMeasure, bpm: float) -> list[str]:
    time_sig = _normalize_time_signature(measure.time_signature)
    capacity_ms = _measure_capacity_ms(bpm, time_sig)

    valid_notes = [n for n in measure.notes if _format_note(n)]
    if not valid_notes:
        return [":4 r |"]

    grouped: dict[float, list[TabNote]] = defaultdict(list)
    for note in sorted(valid_notes, key=lambda n: n.start_ms):
        rel = max(0.0, note.start_ms - measure.start_ms)
        grouped[round(rel, 3)].append(note)

    sorted_starts = sorted(grouped.keys())
    max_rel = max(sorted_starts) + max(n.duration_ms for n in valid_notes)
    bar_count = max(1, int(max_rel // capacity_ms) + (1 if max_rel % capacity_ms > 1e-3 else 0))

    bars: list[str] = []
    for bar_idx in range(bar_count):
        bar_start = bar_idx * capacity_ms
        bar_end = bar_start + capacity_ms
        beats = [(rel - bar_start, grouped[rel]) for rel in sorted_starts if bar_start <= rel < bar_end]
        if not beats:
            bars.append(":4 r |")
            continue
        bars.extend(_build_bars(beats, bar_capacity_ms=capacity_ms, bpm=bpm))
    return bars


def _measure_bar_lines(measure: TabMeasure, bpm: float, *, include_section: bool) -> list[str]:
    lines: list[str] = []
    time_sig = _normalize_time_signature(measure.time_signature)

    if include_section and measure.section:
        lines.append(f'\\section "{_escape_tex(measure.section)}"')
    if measure.index == 0 or measure.time_signature:
        lines.append(_time_signature_line(time_sig))

    lines.extend(_bars_from_measure(measure, measure.tempo_bpm or bpm))
    return lines


def document_to_alphatex(document: TabDocument) -> str:
    meta = document.meta
    title = _escape_tex(meta.title or "Untitled")
    artist = _escape_tex(meta.artist or "MusicAI")
    bpm = meta.bpm if meta.bpm > 0 else 120.0

    lines = [
        f'\\title "{title}"',
        f'\\artist "{artist}"',
        f"\\tempo {int(round(bpm))}",
        _time_signature_line(DEFAULT_TIME_SIGNATURE),
        ".",
    ]

    for track in document.tracks:
        track_name = _escape_tex(track.name or "Guitar")
        lines.append(f'\\track "{track_name}"')
        if track.midi_program is not None:
            lines.append(f"\\instrument {int(track.midi_program)}")
        lines.append("\\staff { tabs }")
        for measure in track.measures:
            lines.extend(
                _measure_bar_lines(
                    measure,
                    bpm,
                    include_section=measure.index == 0 or bool(measure.section),
                )
            )

    tex = "\n".join(lines) + "\n"
    # Collapse accidental blank lines inside track content.
    return re.sub(r"\n{3,}", "\n\n", tex)
