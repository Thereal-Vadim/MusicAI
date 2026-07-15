"""Music Theory Judge — deterministic validation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from judge.rules import (
    ChordContext,
    DetectedKey,
    chord_pitch_classes,
    detect_key,
    infer_chord_symbol,
    is_playable_fret,
    midi_to_name,
    midi_to_pitch_class,
    name_to_midi,
    scale_pitch_classes,
    snap_midi_to_scale,
)
from tab_schema.models import JudgeResult, NoteConfidence, TabNote


@dataclass
class JudgeConfig:
    snap_audio_confidence_threshold: float = 0.5
    beats_per_measure: int = 4
    min_note_duration_ms: float = 50.0


@dataclass
class JudgeStats:
    total_notes: int = 0
    snapped_notes: int = 0
    flagged_notes: int = 0
    chromatic_notes: int = 0


@dataclass
class JudgedResult:
    notes: list[TabNote]
    key: DetectedKey
    chords: list[ChordContext] = field(default_factory=list)
    stats: JudgeStats = field(default_factory=JudgeStats)


class MusicTheoryJudge:
    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()

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
        key = detect_key(stable_pcs if len(stable_pcs) >= 3 else pitch_classes)
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

        for note in sorted(notes, key=lambda n: n.start_ms):
            measure_idx = int(note.start_ms // ms_per_measure)
            chord = next((c for c in chords if c.measure_index == measure_idx), None)
            judged = self._validate_note(note, key, scale, chord, stats)
            judged_notes.append(judged)

        return JudgedResult(notes=judged_notes, key=key, chords=chords, stats=stats)

    def _validate_note(
        self,
        note: TabNote,
        key: DetectedKey,
        scale: set[int],
        chord: ChordContext | None,
        stats: JudgeStats,
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

        if not is_playable_fret(note.fret, note.string):
            flags.append("unplayable_position")

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
