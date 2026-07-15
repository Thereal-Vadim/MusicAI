"""Fret assignment and fingering optimization."""

from __future__ import annotations

from dataclasses import dataclass

from judge.rules import name_to_midi
from tab_schema.models import TabNote

STANDARD_TUNING_MIDI = [64, 59, 55, 50, 45, 40]  # string 1 (high e) .. string 6 (low E)


@dataclass
class FretPosition:
    string: int
    fret: int
    midi: int


def midi_to_positions(midi: int, max_fret: int = 24) -> list[FretPosition]:
    positions: list[FretPosition] = []
    for string_idx, open_midi in enumerate(STANDARD_TUNING_MIDI, start=1):
        fret = midi - open_midi
        if 0 <= fret <= max_fret:
            positions.append(FretPosition(string=string_idx, fret=fret, midi=midi))
    return positions


def assign_fingering(notes: list[tuple[int, float, float, float]]) -> list[TabNote]:
    """Assign string/fret from (pitch_midi, start_ms, duration_ms, confidence)."""
    from judge.judge import note_from_raw

    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n[1])
    assignments: list[TabNote] = []
    hand_position = 0

    for pitch_midi, start_ms, duration_ms, confidence in sorted_notes:
        candidates = midi_to_positions(pitch_midi)
        if not candidates:
            assignments.append(
                note_from_raw(pitch_midi, start_ms, duration_ms, 1, 0, confidence)
            )
            continue

        best = min(
            candidates,
            key=lambda pos: abs(pos.fret - hand_position) + (0 if pos.fret >= hand_position else 2),
        )
        hand_position = best.fret
        assignments.append(
            note_from_raw(
                pitch_midi,
                start_ms,
                duration_ms,
                best.string,
                best.fret,
                confidence,
            )
        )

    return assignments


def optimize_sequence(notes: list[TabNote]) -> list[TabNote]:
    """Dynamic programming optimizer minimizing hand movement."""
    if len(notes) <= 1:
        return notes

    candidates_per_note: list[list[FretPosition]] = []
    for note in notes:
        midi = note.pitch_midi or name_to_midi(note.pitch)
        candidates_per_note.append(midi_to_positions(midi) or [FretPosition(note.string, note.fret, midi)])

    n = len(notes)
    dp: list[list[float]] = [[float("inf")] * len(candidates_per_note[i]) for i in range(n)]
    back: list[list[int]] = [[-1] * len(candidates_per_note[i]) for i in range(n)]

    for j, pos in enumerate(candidates_per_note[0]):
        dp[0][j] = pos.fret

    for i in range(1, n):
        for j, pos in enumerate(candidates_per_note[i]):
            for k, prev in enumerate(candidates_per_note[i - 1]):
                cost = abs(pos.fret - prev.fret) + (5 if pos.string == prev.string else 0)
                total = dp[i - 1][k] + cost
                if total < dp[i][j]:
                    dp[i][j] = total
                    back[i][j] = k

    last_idx = min(range(len(dp[-1])), key=lambda idx: dp[-1][idx])
    path: list[int] = [last_idx]
    for i in range(n - 1, 0, -1):
        last_idx = back[i][last_idx]
        path.append(last_idx)
    path.reverse()

    optimized: list[TabNote] = []
    for i, cand_idx in enumerate(path):
        pos = candidates_per_note[i][cand_idx]
        note = notes[i].model_copy(deep=True)
        note.string = pos.string
        note.fret = pos.fret
        optimized.append(note)

    return optimized
