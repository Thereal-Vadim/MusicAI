"""Music theory validation rules."""

from __future__ import annotations

from dataclasses import dataclass

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

CHORD_TEMPLATES: dict[str, set[int]] = {
    "maj": {0, 4, 7},
    "min": {0, 3, 7},
    "dim": {0, 3, 6},
    "dom7": {0, 4, 7, 10},
}


@dataclass
class DetectedKey:
    root: str
    mode: str
    confidence: float


@dataclass
class ChordContext:
    measure_index: int
    symbol: str
    pitch_classes: set[int]


def midi_to_pitch_class(midi: int) -> int:
    return midi % 12


def pitch_class_name(pc: int) -> str:
    return PITCH_CLASS_NAMES[pc % 12]


def midi_to_name(midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{pitch_class_name(midi)}{octave}"


def name_to_midi(name: str) -> int:
    if len(name) < 2:
        raise ValueError(f"Invalid pitch name: {name}")
    if name[1] == "#":
        pc = PITCH_CLASS_NAMES.index(name[:2])
        octave = int(name[2:])
    elif name[1].isdigit():
        pc = PITCH_CLASS_NAMES.index(name[0])
        octave = int(name[1:])
    else:
        raise ValueError(f"Invalid pitch name: {name}")
    return (octave + 1) * 12 + pc


def detect_key_music21(pitch_classes: list[int]) -> DetectedKey | None:
    """Use music21 Krumhansl-Schmuckler key analysis when available."""
    if not pitch_classes:
        return None
    try:
        from music21 import note as m21note, pitch, stream

        score = stream.Stream()
        for pc in pitch_classes:
            n = m21note.Note(pitch.Pitch(midi=60 + pc))
            score.append(n)
        analyzed = score.analyze("key")
        tonic_map = {"D-": "C#", "E-": "D#", "G-": "F#", "A-": "G#", "B-": "A#"}
        root = tonic_map.get(analyzed.tonic.name, analyzed.tonic.name)
        if root not in PITCH_CLASS_NAMES and len(root) == 1:
            root = pitch_class_name(PITCH_CLASS_NAMES.index(root))
        mode = analyzed.mode or "major"
        return DetectedKey(root=root, mode=mode, confidence=0.85)
    except Exception:
        return None


def detect_key(pitch_classes: list[int]) -> DetectedKey:
    if not pitch_classes:
        return DetectedKey(root="C", mode="major", confidence=0.0)

    histogram = [0.0] * 12
    for pc in pitch_classes:
        histogram[pc] += 1.0

    total = sum(histogram) or 1.0
    histogram = [h / total for h in histogram]

    best_root = 0
    best_mode = "major"
    best_score = -1.0
    for root in range(12):
        for mode, profile in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
            profile_sum = sum(profile)
            score = sum(histogram[i] * profile[(i - root) % 12] for i in range(12)) / profile_sum
            if score > best_score:
                best_score = score
                best_root = root
                best_mode = mode

    confidence = min(best_score * 2.0, 1.0)
    return DetectedKey(root=pitch_class_name(best_root), mode=best_mode, confidence=confidence)


def detect_key_with_fallback(pitch_classes: list[int], use_music21: bool = True) -> DetectedKey:
    if use_music21:
        m21_key = detect_key_music21(pitch_classes)
        if m21_key is not None:
            return m21_key
    return detect_key(pitch_classes)


def scale_pitch_classes(root: str, mode: str) -> set[int]:
    root_pc = PITCH_CLASS_NAMES.index(root)
    if mode == "minor":
        intervals = {0, 2, 3, 5, 7, 8, 10}
    else:
        intervals = {0, 2, 4, 5, 7, 9, 11}
    return {(root_pc + i) % 12 for i in intervals}


def infer_chord_symbol(pitch_classes: set[int], key: DetectedKey) -> str:
    if not pitch_classes:
        return "N.C."

    root_candidates = sorted(pitch_classes)
    best: tuple[str, int] | None = None
    for root_pc in root_candidates:
        relative = {(pc - root_pc) % 12 for pc in pitch_classes}
        for quality, template in CHORD_TEMPLATES.items():
            overlap = len(relative & template)
            if best is None or overlap > best[1]:
                suffix = "" if quality == "maj" else ("m" if quality == "min" else quality)
                best = (f"{pitch_class_name(root_pc)}{suffix}", overlap)

    return best[0] if best else "N.C."


def chord_pitch_classes(symbol: str) -> set[int]:
    if symbol == "N.C.":
        return set()

    quality = "maj"
    root_part = symbol
    if symbol.endswith("dom7"):
        quality = "dom7"
        root_part = symbol[:-4]
    elif symbol.endswith("dim"):
        quality = "dim"
        root_part = symbol[:-3]
    elif symbol.endswith("m"):
        quality = "min"
        root_part = symbol[:-1]

    if "#" in root_part:
        root_pc = PITCH_CLASS_NAMES.index(root_part[:2])
    else:
        root_pc = PITCH_CLASS_NAMES.index(root_part[0])

    template = CHORD_TEMPLATES[quality]
    return {(root_pc + interval) % 12 for interval in template}


def snap_midi_to_pitch_classes(midi: int, allowed: set[int]) -> int:
    if not allowed:
        return midi
    pc = midi_to_pitch_class(midi)
    if pc in allowed:
        return midi

    best_midi = midi
    best_dist = 999
    for delta in range(-6, 7):
        candidate = midi + delta
        if midi_to_pitch_class(candidate) in allowed:
            dist = abs(delta)
            if dist < best_dist:
                best_dist = dist
                best_midi = candidate
    return best_midi


def snap_midi_to_scale(midi: int, scale: set[int]) -> int:
    return snap_midi_to_pitch_classes(midi, scale)


def is_playable_fret(fret: int, string_num: int) -> bool:
    return 1 <= string_num <= 6 and 0 <= fret <= 24


def chord_span_ok(frets: list[int], max_span: int = 5) -> bool:
    if len(frets) <= 1:
        return True
    return max(frets) - min(frets) <= max_span


def validate_simultaneous_notes(note_count: int, max_notes: int = 4) -> bool:
    return note_count <= max_notes


def validate_temporal_order(notes: list, min_duration_ms: float = 50.0) -> list[str]:
    """Return note ids that violate temporal sanity."""
    violations: list[str] = []
    ordered = sorted(notes, key=lambda n: n.start_ms)
    prev_start = -1.0
    for note in ordered:
        if note.start_ms < prev_start:
            violations.append(note.id)
        if note.duration_ms < min_duration_ms:
            violations.append(note.id)
        prev_start = note.start_ms
    return list(dict.fromkeys(violations))
