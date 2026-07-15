"""Convert TabDocument to Guitar Pro 5 binary."""

from __future__ import annotations

import copy
import io
from collections import defaultdict
from pathlib import Path

from tab_schema.models import TabDocument, TabMeasure, TabNote

TEMPLATE_PATH = Path(__file__).parent / "assets" / "template.gp5"
TICKS_PER_QUARTER = 960
TICKS_PER_MEASURE_4_4 = 3840


def _duration_value(duration_ms: float, bpm: float) -> int:
    quarter_ms = 60_000 / bpm if bpm > 0 else 500
    ratio = duration_ms / quarter_ms
    if ratio >= 0.875:
        return 4
    if ratio >= 0.4:
        return 8
    if ratio >= 0.2:
        return 16
    return 32


def _load_template() -> object:
    import guitarpro
    import urllib.request

    if TEMPLATE_PATH.exists():
        return guitarpro.parse(str(TEMPLATE_PATH))
    url = "https://raw.githubusercontent.com/Perlence/PyGuitarPro/master/tests/Demo%20v5.gp5"
    with urllib.request.urlopen(url) as stream:
        return guitarpro.parse(stream)


def _apply_measure(track, measure: TabMeasure, template_beat, bpm: float, header_index: int) -> None:
    from guitarpro.models import Duration, Marker, Note, NoteType, Beat

    if header_index < len(track.song.measureHeaders) and measure.section:
        track.song.measureHeaders[header_index].marker = Marker(title=measure.section)

    if header_index >= len(track.measures):
        return

    voice = track.measures[header_index].voices[0]
    grouped: dict[float, list[TabNote]] = defaultdict(list)
    for note in measure.notes:
        grouped[note.start_ms].append(note)

    beats = []
    for start_ms in sorted(grouped.keys()):
        notes = grouped[start_ms]
        duration_ms = max(n.duration_ms for n in notes)
        beat = copy.deepcopy(template_beat)
        beat.voice = voice
        beat.duration = Duration(value=_duration_value(duration_ms, bpm))
        beat.notes = []
        for note in sorted(notes, key=lambda n: n.string):
            beat.notes.append(
                Note(beat=beat, string=note.string, value=note.fret, type=NoteType.normal)
            )
        beats.append(beat)
    voice.beats = beats


def document_to_gp5_bytes(document: TabDocument) -> bytes:
    import guitarpro

    song = _load_template()
    song.title = document.meta.title or "Untitled"
    song.artist = document.meta.artist or "MusicAI"
    song.album = document.meta.album or ""
    song.tempo = int(round(document.meta.bpm))

    track = song.tracks[0]
    track.name = document.tracks[0].name if document.tracks else "Guitar"
    template_beat = track.measures[0].voices[0].beats[0]

    max_measures = max((len(t.measures) for t in document.tracks), default=0)
    max_measures = max(max_measures, len(song.measureHeaders))

    for track_idx, tab_track in enumerate(document.tracks):
        gp_track = song.tracks[min(track_idx, len(song.tracks) - 1)]
        if track_idx > 0 and track_idx >= len(song.tracks):
            break
        for measure in tab_track.measures:
            _apply_measure(gp_track, measure, template_beat, document.meta.bpm, measure.index)

    buffer = io.BytesIO()
    guitarpro.write(song, buffer, version=(5, 1, 0))
    return buffer.getvalue()
