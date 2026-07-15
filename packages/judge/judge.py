"""Music Theory Judge — deterministic validation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import yaml

from judge.rules import (
    ChordContext,
    DetectedKey,
    chord_pitch_classes,
    chord_span_ok,
    detect_key_with_fallback,
    infer_chord_symbol,
    is_playable_fret,
    midi_to_name,
    midi_to_pitch_class,
    name_to_midi,
    scale_pitch_classes,
    snap_midi_to_scale,
    validate_simultaneous_notes,
    validate_temporal_order,
)
from judge.settings import JudgeSettings, judge_settings
from tab_schema.models import JudgeResult, NoteConfidence, TabNote


@dataclass
class JudgeConfig:
    snap_audio_confidence_threshold: float = 0.5
    beats_per_measure: int = 4
    min_note_duration_ms: float = 50.0
    max_simultaneous_notes: int = 4
    max_chord_span: int = 5
    use_music21: bool = True

    @classmethod
    def from_settings(cls, settings: JudgeSettings | None = None) -> "JudgeConfig":
        cfg = settings or judge_settings
        return cls(
            snap_audio_confidence_threshold=cfg.judge_snap_audio_confidence_threshold,
            beats_per_measure=cfg.judge_beats_per_measure,
            min_note_duration_ms=cfg.judge_min_note_duration_ms,
            max_simultaneous_notes=cfg.judge_max_simultaneous_notes,
            max_chord_span=cfg.judge_max_chord_span,
            use_music21=cfg.judge_use_music21,
        )

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "JudgeConfig":
        yaml_path = path or Path(__file__).parent / "config.yaml"
        if not yaml_path.exists():
            return cls.from_settings()
        data = yaml.safe_load(yaml_path.read_text()) or {}
        return cls(
            snap_audio_confidence_threshold=float(data.get("snap_audio_confidence_threshold", 0.5)),
            beats_per_measure=int(data.get("beats_per_measure", 4)),
            min_note_duration_ms=float(data.get("min_note_duration_ms", 50.0)),
            max_simultaneous_notes=int(data.get("max_simultaneous_notes", 4)),
            max_chord_span=int(data.get("max_chord_span", 5)),
            use_music21=bool(data.get("use_music21", True)),
        )


@dataclass
class JudgeStats:
    total_notes: int = 0
    snapped_notes: int = 0
    flagged_notes: int = 0
    chromatic_notes: int = 0
    playability_violations: int = 0


@dataclass
class JudgedResult:
    notes: list[TabNote]
    key: DetectedKey
    chords: list[ChordContext] = field(default_factory=list)
    stats: JudgeStats = field(default_factory=JudgeStats)

    def to_report(self) -> dict[str, object]:
        return {
            "key": {"root": self.key.root, "mode": self.key.mode, "confidence": self.key.confidence},
            "chords": [
                {"measure_index": c.measure_index, "symbol": c.symbol} for c in self.chords
            ],
            "stats": {
                "total_notes": self.stats.total_notes,
                "snapped_notes": self.stats.snapped_notes,
                "flagged_notes": self.stats.flagged_notes,
                "chromatic_notes": self.stats.chromatic_notes,
                "playability_violations": self.stats.playability_violations,
            },
        }


class MusicTheoryJudge:
    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig.from_yaml()

    def judge(self, notes: list[TabNote], bpm: float = 120.0) -> JudgedResult:
        if not notes:
            key = DetectedKey(root="C", mode="major", confidence=0.0)
            return JudgedResult(notes=[], key=key)

        ordered = sorted(notes, key=lambda n: n.start_ms)
        pitch_classes = [midi_to_pitch_class(n.pitch_midi or name_to_midi(n.pitch)) for n in ordered]
        key_seed = ordered[: min(len(ordered), 8)]
        stable_pcs = [
            midi_to_pitch_class(n.pitch_midi or name_to_midi(n.pitch))
            for n in key_seed
            if n.confidence.audio >= 0.5
        ]
        key = detect_key_with_fallback(
            stable_pcs if len(stable_pcs) >= 3 else pitch_classes,
            use_music21=self.config.use_music21,
        )
        scale = scale_pitch_classes(key.root, key.mode)

        bpm = bpm if bpm and bpm > 0 else 120.0
        ms_per_measure = (60_000 / bpm) * self.config.beats_per_measure
        measures: dict[int, list[TabNote]] = {}
        for note in notes:
            measure_idx = int(note.start_ms // ms_per_measure)
            measures.setdefault(measure_idx, []).append(note)

        chords: list[ChordContext] = []
        for measure_idx, measure_notes in sorted(measures.items()):
            pcs = {
                midi_to_pitch_class(n.pitch_midi or name_to_midi(n.pitch)) for n in measure_notes
            }
            symbol = infer_chord_symbol(pcs, key)
            chords.append(
                ChordContext(
                    measure_index=measure_idx,
                    symbol=symbol,
                    pitch_classes=chord_pitch_classes(symbol),
                )
            )

        stats = JudgeStats(total_notes=len(notes))
        judged_notes: list[TabNote] = []

        temporal_violations = set(
            validate_temporal_order(notes, min_duration_ms=self.config.min_note_duration_ms)
        )

        for note in sorted(notes, key=lambda n: n.start_ms):
            measure_idx = int(note.start_ms // ms_per_measure)
            chord = next((c for c in chords if c.measure_index == measure_idx), None)
            measure_notes = measures.get(measure_idx, [])
            judged = self._validate_note(
                note,
                key,
                scale,
                chord,
                measure_notes,
                stats,
                temporal_violations,
            )
            judged_notes.append(judged)

        return JudgedResult(notes=judged_notes, key=key, chords=chords, stats=stats)

    def _validate_note(
        self,
        note: TabNote,
        key: DetectedKey,
        scale: set[int],
        chord: ChordContext | None,
        measure_notes: list[TabNote],
        stats: JudgeStats,
        temporal_violations: set[str],
    ) -> TabNote:
        flags: list[str] = list(note.flags)
        midi = note.pitch_midi or name_to_midi(note.pitch)
        pc = midi_to_pitch_class(midi)

        in_scale = pc in scale
        in_chord = pc in chord.pitch_classes if chord and chord.pitch_classes else in_scale

        if not in_scale:
            flags.append("chromatic_note")
            stats.chromatic_notes += 1
        if chord and chord.pitch_classes and not in_chord and not in_scale:
            flags.append("out_of_harmony")

        if note.duration_ms < self.config.min_note_duration_ms:
            flags.append("short_duration")

        if note.id in temporal_violations:
            flags.append("temporal_violation")

        if not is_playable_fret(note.fret, note.string):
            flags.append("unplayable_position")
            stats.playability_violations += 1

        simultaneous = [
            n
            for n in measure_notes
            if abs(n.start_ms - note.start_ms) < 30.0
        ]
        if not validate_simultaneous_notes(
            len(simultaneous), max_notes=self.config.max_simultaneous_notes
        ):
            flags.append("too_many_simultaneous_notes")
            stats.playability_violations += 1

        frets = [n.fret for n in simultaneous]
        if not chord_span_ok(frets, max_span=self.config.max_chord_span):
            flags.append("chord_span_exceeded")
            stats.playability_violations += 1

        snapped = False
        snap_reason: str | None = None
        original_pitch = note.original_pitch

        should_snap = (
            note.confidence.audio < self.config.snap_audio_confidence_threshold
            and (not in_scale or "out_of_harmony" in flags)
        )

        if should_snap:
            snapped_midi = snap_midi_to_scale(midi, scale or {pc})
            if snapped_midi != midi:
                original_pitch = note.pitch
                note.pitch = midi_to_name(snapped_midi)
                note.pitch_midi = snapped_midi
                note.sources.theory_fret = note.fret
                snapped = True
                snap_reason = "out_of_harmony_low_audio_conf"
                flags.append("auto_corrected")
                stats.snapped_notes += 1
                in_scale = True

        judge_conf = 0.95 if in_scale and in_chord else (0.6 if in_scale else 0.35)
        note.judge = JudgeResult(
            in_scale=in_scale,
            in_chord=in_chord,
            snapped=snapped,
            snap_reason=snap_reason,
            flags=list(dict.fromkeys(flags)),
        )
        note.original_pitch = original_pitch
        note.flags = list(dict.fromkeys(flags + note.flags))
        note.confidence.judge = judge_conf
        note.confidence.overall = self._overall_confidence(note.confidence)

        if note.judge.flags:
            stats.flagged_notes += 1

        return note

    @staticmethod
    def _overall_confidence(confidence: NoteConfidence) -> float:
        weights = {"audio": 0.5, "vision": 0.25, "judge": 0.25}
        total = 0.0
        weight_sum = 0.0
        for name, weight in weights.items():
            value = getattr(confidence, name)
            if value > 0:
                total += value * weight
                weight_sum += weight
        return total / weight_sum if weight_sum else confidence.audio


def note_from_raw(
    pitch_midi: int,
    start_ms: float,
    duration_ms: float,
    string_num: int,
    fret: int,
    audio_confidence: float,
    vision_confidence: float = 0.0,
) -> TabNote:
    return TabNote(
        id=str(uuid4()),
        pitch=midi_to_name(pitch_midi),
        pitch_midi=pitch_midi,
        start_ms=start_ms,
        duration_ms=duration_ms,
        string=string_num,
        fret=fret,
        confidence=NoteConfidence(audio=audio_confidence, vision=vision_confidence),
    )
