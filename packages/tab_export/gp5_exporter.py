"""Convert TabDocument to Guitar Pro 5 binary."""

from __future__ import annotations

import copy
import io
from collections import defaultdict
from pathlib import Path

from tab_schema.models import TabDocument, TabMeasure, TabNote

TEMPLATE_PATH = Path(__file__).parent / "assets" / "template.gp5"
TEMPLATE_URL = (
    "https://raw.githubusercontent.com/Perlence/PyGuitarPro/master/tests/Demo%20v5.gp5"
)
TICKS_PER_QUARTER = 960
TICKS_PER_MEASURE_4_4 = 3840

_TEMPLATE: object | None = None


def _load_template() -> object:
    global _TEMPLATE
    if _TEMPLATE is not None:
        return _TEMPLATE

    import guitarpro
    import urllib.request

    if not TEMPLATE_PATH.exists():
        TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEMPLATE_URL, TEMPLATE_PATH)

    _TEMPLATE = guitarpro.parse(str(TEMPLATE_PATH))
    return _TEMPLATE


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


def _apply_measure(track, measure: TabMeasure, template_beat, bpm: float, header_index: int) -> None:
    from guitarpro.models import Duration, Marker, Note, NoteType

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


def _prepare_single_guitar_track(song, *, track_name: str, measure_count: int) -> None:
    """Keep one guitar track and drop demo template content beyond the draft."""
    while len(song.tracks) > 1:
        del song.tracks[-1]

    track = song.tracks[0]
    track.name = track_name

    measure_count = max(measure_count, 1)
    while len(track.measures) > measure_count:
        del track.measures[-1]
    while len(song.measureHeaders) > measure_count:
        del song.measureHeaders[-1]


def document_to_gp5_bytes(document: TabDocument) -> bytes:
    import guitarpro

    song = _load_template()
    song.title = document.meta.title or "Untitled"
    song.artist = document.meta.artist or "MusicAI"
    song.album = document.meta.album or ""
    song.tempo = int(round(document.meta.bpm))

    track_name = document.tracks[0].name if document.tracks else "Guitar"
    max_measure_index = max(
        (measure.index for tab_track in document.tracks for measure in tab_track.measures),
        default=0,
    )
    _prepare_single_guitar_track(song, track_name=track_name, measure_count=max_measure_index + 1)

    track = song.tracks[0]
    template_beat = track.measures[0].voices[0].beats[0]

    for tab_track in document.tracks:
        for measure in tab_track.measures:
            _apply_measure(track, measure, template_beat, document.meta.bpm, measure.index)

    buffer = io.BytesIO()
    guitarpro.write(song, buffer, version=(5, 1, 0))
    return buffer.getvalue()
