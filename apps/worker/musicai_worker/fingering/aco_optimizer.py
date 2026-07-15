"""ACO-based fingering optimizer — biomechanical playability over fret-position graph."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from judge.rules import chord_span_ok, name_to_midi, validate_simultaneous_notes
from musicai_worker.fingering.optimizer import FretPosition, midi_to_positions
from tab_schema.models import TabNote


@dataclass(frozen=True)
class ACOConfig:
    n_ants: int = 24
    n_iterations: int = 50
    max_fret_span: int = 4
    alpha: float = 1.0
    beta: float = 2.5
    rho: float = 0.45
    q: float = 100.0
    open_string_bonus: float = 2.0
    string_skip_penalty: float = 3.5
    position_shift_penalty: float = 1.0
    large_shift_threshold: int = 5
    large_shift_extra: float = 4.0
    same_string_repeat_penalty: float = 2.0
    chord_time_tolerance_ms: float = 15.0
    impossible_penalty: float = 1e9


def _group_simultaneous(notes: list[TabNote], tolerance_ms: float) -> list[list[int]]:
    if not notes:
        return []
    groups: list[list[int]] = []
    current = [0]
    for i in range(1, len(notes)):
        if abs(notes[i].start_ms - notes[current[0]].start_ms) <= tolerance_ms:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def _hand_anchor(positions: list[FretPosition]) -> int:
    """Approximate left-hand position as the lowest fret in a chord or single note."""
    return min(p.fret for p in positions)


def _transition_cost(
    prev_positions: list[FretPosition] | None,
    next_positions: list[FretPosition],
    config: ACOConfig,
) -> float:
    """Local heuristic η — lower cost = more playable transition."""
    next_anchor = _hand_anchor(next_positions)
    open_bonus = sum(config.open_string_bonus for p in next_positions if p.fret == 0)

    if prev_positions is None:
        return max(0.0, float(next_anchor) * 0.4 - open_bonus)

    prev_anchor = _hand_anchor(prev_positions)
    prev = prev_positions[-1]
    nxt = next_positions[0]

    fret_delta = abs(next_anchor - prev_anchor)
    cost = fret_delta * config.position_shift_penalty

    if fret_delta > config.large_shift_threshold:
        cost += (fret_delta - config.large_shift_threshold) * config.large_shift_extra

    string_delta = abs(nxt.string - prev.string)
    if string_delta == 0:
        cost += config.same_string_repeat_penalty
    elif string_delta > 1:
        cost += config.string_skip_penalty * (string_delta - 1)

    if nxt.fret < prev.fret:
        cost += 1.5

    cost -= open_bonus
    return max(0.0, cost)


def _chord_cluster_cost(positions: list[FretPosition], config: ACOConfig) -> float:
    """Polyphony + anatomical constraints for simultaneous notes."""
    if len(positions) <= 1:
        return 0.0

    strings = [p.string for p in positions]
    if len(strings) != len(set(strings)):
        return config.impossible_penalty

    if not validate_simultaneous_notes(len(positions), max_notes=6):
        return config.impossible_penalty

    frets = [p.fret for p in positions]
    if not chord_span_ok(frets, max_span=config.max_fret_span):
        return config.impossible_penalty

    span = max(frets) - min(frets)
    return span * 0.5


def _build_segment_choices(
    note_indices: list[int],
    candidates_per_note: list[list[FretPosition]],
    config: ACOConfig,
) -> list[tuple[int, ...]]:
    """Joint fret assignments for a time slice (single note or chord super-node)."""
    if len(note_indices) == 1:
        idx = note_indices[0]
        return [(choice,) for choice in range(len(candidates_per_note[idx]))]

    ranges = [range(len(candidates_per_note[i])) for i in note_indices]
    valid: list[tuple[int, ...]] = []
    for combo in product(*ranges):
        positions = [candidates_per_note[note_indices[j]][combo[j]] for j in range(len(note_indices))]
        if _chord_cluster_cost(positions, config) >= config.impossible_penalty:
            continue
        valid.append(combo)

    if valid:
        return valid

    return [(0,) * len(note_indices)]


def _positions_for_choice(
    note_indices: list[int],
    choice: tuple[int, ...],
    candidates_per_note: list[list[FretPosition]],
) -> list[FretPosition]:
    return [candidates_per_note[note_indices[j]][choice[j]] for j in range(len(note_indices))]


def _evaluate_segment_path(
    segments: list[list[int]],
    segment_choices: list[list[tuple[int, ...]]],
    path: list[int],
    candidates_per_note: list[list[FretPosition]],
    config: ACOConfig,
) -> float:
    total = 0.0
    prev_positions: list[FretPosition] | None = None

    for seg_idx, note_indices in enumerate(segments):
        positions = _positions_for_choice(
            note_indices,
            segment_choices[seg_idx][path[seg_idx]],
            candidates_per_note,
        )
        total += _chord_cluster_cost(positions, config)
        total += _transition_cost(prev_positions, positions, config)
        prev_positions = positions

    return total


def _heuristic_eta(
    prev_positions: list[FretPosition] | None,
    next_positions: list[FretPosition],
    config: ACOConfig,
) -> float:
    cost = _transition_cost(prev_positions, next_positions, config)
    return 1.0 / (1.0 + cost)


def optimize_sequence_aco(notes: list[TabNote], config: ACOConfig | None = None) -> list[TabNote]:
    """
    Ant Colony Optimization on a fret-position graph.

    Each timeline segment is a column of nodes (all playable string/fret options).
    Chords are super-nodes: only jointly playable assignments survive.
    Ants minimize biomechanical cost (shifts, string skips, stretch) via pheromone search.
    """
    if len(notes) <= 1:
        return notes

    cfg = config or ACOConfig()
    candidates_per_note: list[list[FretPosition]] = []
    for note in notes:
        midi = note.pitch_midi or name_to_midi(note.pitch)
        candidates_per_note.append(
            midi_to_positions(midi) or [FretPosition(note.string, note.fret, midi)]
        )

    segments = _group_simultaneous(notes, cfg.chord_time_tolerance_ms)
    segment_choices = [_build_segment_choices(seg, candidates_per_note, cfg) for seg in segments]
    n_segments = len(segments)
    n_choices = [len(choices) for choices in segment_choices]

    pheromone = [np.ones(n_c, dtype=np.float64) for n_c in n_choices]
    best_path: list[int] | None = None
    best_cost = float("inf")
    rng = np.random.default_rng(42)

    for _ in range(cfg.n_iterations):
        iteration_best_path: list[int] | None = None
        iteration_best_cost = float("inf")

        for _ant in range(cfg.n_ants):
            path: list[int] = []
            prev_positions: list[FretPosition] | None = None

            for seg_idx, n_c in enumerate(n_choices):
                if seg_idx == 0:
                    probs = pheromone[seg_idx] ** cfg.alpha
                else:
                    heuristic = np.array(
                        [
                            _heuristic_eta(
                                prev_positions,
                                _positions_for_choice(
                                    segments[seg_idx],
                                    segment_choices[seg_idx][j],
                                    candidates_per_note,
                                ),
                                cfg,
                            )
                            for j in range(n_c)
                        ],
                        dtype=np.float64,
                    )
                    probs = (pheromone[seg_idx] ** cfg.alpha) * (heuristic ** cfg.beta)

                probs_sum = probs.sum()
                if probs_sum <= 0:
                    choice = int(rng.integers(0, n_c))
                else:
                    choice = int(rng.choice(n_c, p=probs / probs_sum))
                path.append(choice)
                prev_positions = _positions_for_choice(
                    segments[seg_idx],
                    segment_choices[seg_idx][choice],
                    candidates_per_note,
                )

            cost = _evaluate_segment_path(
                segments, segment_choices, path, candidates_per_note, cfg
            )
            if cost < iteration_best_cost:
                iteration_best_cost = cost
                iteration_best_path = path

        if iteration_best_path is None:
            break

        if iteration_best_cost < best_cost:
            best_cost = iteration_best_cost
            best_path = iteration_best_path

        if iteration_best_cost >= cfg.impossible_penalty:
            continue

        deposit = cfg.q / (1.0 + iteration_best_cost)
        for seg_idx in range(n_segments):
            pheromone[seg_idx] *= 1.0 - cfg.rho
            pheromone[seg_idx][iteration_best_path[seg_idx]] += deposit

    if best_path is None or best_cost >= cfg.impossible_penalty:
        from musicai_worker.fingering.optimizer import optimize_sequence

        return optimize_sequence(notes)

    optimized: list[TabNote] = []
    for seg_idx, note_indices in enumerate(segments):
        positions = _positions_for_choice(
            note_indices,
            segment_choices[seg_idx][best_path[seg_idx]],
            candidates_per_note,
        )
        for note_idx, pos in zip(note_indices, positions, strict=True):
            note = notes[note_idx].model_copy(deep=True)
            note.string = pos.string
            note.fret = pos.fret
            optimized.append(note)

    return optimized
