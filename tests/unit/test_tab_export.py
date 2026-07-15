"""Tests for tab export formats."""

import json
from pathlib import Path

import guitarpro

from tab_export.alphatex_exporter import document_to_alphatex
from tab_export.gp5_exporter import document_to_gp5_bytes
from tab_schema.models import SourceMeta, TabDocument, TabMeasure, TabMeta, TabNote, TabTrack
from tab_schema.models import NoteConfidence, JudgeResult, NoteTechnique


def _intro_document() -> TabDocument:
    reference = json.loads(
        (Path(__file__).resolve().parents[2] / "benchmarks" / "enter_sandman" / "reference_intro.json").read_text()
    )
    notes = []
    for ref in reference["notes"]:
        notes.append(
            TabNote(
                id=f"n-{ref['start_ms']}",
                pitch="E2",
                start_ms=ref["start_ms"],
                duration_ms=ref["duration_ms"],
                string=ref["string"],
                fret=ref["fret"],
                pitch_midi=ref["pitch_midi"],
                confidence=NoteConfidence(audio=0.9, overall=0.9),
                judge=JudgeResult(),
            )
        )
    return TabDocument(
        meta=TabMeta(
            title="Enter Sandman",
            artist="Metallica",
            bpm=123,
            source=SourceMeta(type="upload"),
        ),
        tracks=[
            TabTrack(
                name="Distortion Guitar",
                measures=[
                    TabMeasure(
                        index=0,
                        start_ms=0,
                        section="Intro",
                        time_signature=(4, 4),
                        tempo_bpm=123,
                        notes=notes,
                    )
                ],
            )
        ],
    )


def test_alphatex_contains_metadata_and_notes():
    tex = document_to_alphatex(_intro_document())
    assert '\\title "Enter Sandman"' in tex
    assert '\\artist "Metallica"' in tex
    assert "\\tempo 123" in tex
    assert '\\section "Intro"' in tex
    assert "0.6" in tex
    assert "5.6" in tex
    assert "7.6" in tex


def test_alphatex_no_invalid_time_signature():
    tex = document_to_alphatex(_intro_document())
    assert "\\ts 1 4" not in tex
    assert "\\track" in tex
    assert "\\staff { tabs }" in tex


def test_gp5_roundtrip_intro():
    import io

    data = document_to_gp5_bytes(_intro_document())
    assert len(data) > 1000
    song = guitarpro.parse(io.BytesIO(data))
    assert len(song.tracks) == 1
    assert len(song.tracks[0].measures) == 1
    assert song.title == "Enter Sandman"
    assert song.artist == "Metallica"
    assert song.tempo == 123
    beats = song.tracks[0].measures[0].voices[0].beats
    frets = [b.notes[0].value for b in beats if b.notes]
    assert frets[:4] == [0, 0, 5, 5]


def test_gp5_multi_track_midi_programs():
    import io

    doc = _intro_document()
    lead = doc.tracks[0]
    lead.name = "Distortion Guitar (Lead)"
    lead.midi_program = 30
    lead.role = "solo"

    rhythm_notes = [
        TabNote(
            id="r-0",
            pitch="E2",
            start_ms=0,
            duration_ms=400,
            string=6,
            fret=0,
            pitch_midi=40,
            confidence=NoteConfidence(audio=0.8, overall=0.8),
            judge=JudgeResult(),
        )
    ]
    doc.tracks.append(
        TabTrack(
            name="Electric Guitar (clean) (Rhythm)",
            midi_program=27,
            role="rhythm",
            measures=[
                TabMeasure(index=0, start_ms=0, notes=rhythm_notes),
            ],
        )
    )

    data = document_to_gp5_bytes(doc)
    song = guitarpro.parse(io.BytesIO(data))
    assert len(song.tracks) == 2
    assert song.tracks[0].name == "Distortion Guitar (Lead)"
    assert song.tracks[0].channel.instrument == 30
    assert song.tracks[1].name == "Electric Guitar (clean) (Rhythm)"
    assert song.tracks[1].channel.instrument == 27

    tex = document_to_alphatex(doc)
    assert '\\track "Distortion Guitar (Lead)"' in tex
    assert "\\instrument 30" in tex
    assert '\\track "Electric Guitar (clean) (Rhythm)"' in tex
    assert "\\instrument 27" in tex
