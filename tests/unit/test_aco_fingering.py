"""ACO fingering optimizer tests."""

from judge.judge import note_from_raw
from musicai_worker.fingering.aco_optimizer import (
    ACOConfig,
    _chord_cluster_cost,
    _transition_cost,
    optimize_sequence_aco,
)
from musicai_worker.fingering.optimizer import FretPosition, optimize_sequence


def test_aco_preserves_note_count():
    notes = [
        note_from_raw(64, 0, 200, 1, 0, 0.8),
        note_from_raw(67, 250, 200, 1, 3, 0.8),
        note_from_raw(69, 500, 200, 1, 5, 0.8),
        note_from_raw(72, 750, 200, 1, 8, 0.8),
    ]
    optimized = optimize_sequence_aco(notes, ACOConfig(n_ants=8, n_iterations=10))
    assert len(optimized) == 4
    assert all(1 <= n.string <= 6 for n in optimized)


def test_aco_chord_cluster_respects_fret_span():
    notes = [
        note_from_raw(52, 0, 300, 6, 0, 0.9),
        note_from_raw(57, 0, 300, 5, 2, 0.9),
        note_from_raw(60, 0, 300, 4, 5, 0.9),
    ]
    cfg = ACOConfig(n_ants=12, n_iterations=20, max_fret_span=4)
    optimized = optimize_sequence_aco(notes, cfg)
    frets = [n.fret for n in optimized]
    assert max(frets) - min(frets) <= cfg.max_fret_span + 2


def test_aco_chord_avoids_same_string_collision():
    notes = [
        note_from_raw(64, 0, 300, 1, 0, 0.9),
        note_from_raw(67, 0, 300, 1, 3, 0.9),
    ]
    optimized = optimize_sequence_aco(notes, ACOConfig(n_ants=16, n_iterations=25))
    strings = [n.string for n in optimized]
    assert len(strings) == len(set(strings))


def test_transition_cost_penalizes_string_skip():
    cfg = ACOConfig()
    adjacent = _transition_cost(
        [FretPosition(2, 5, 69)],
        [FretPosition(3, 5, 64)],
        cfg,
    )
    skip = _transition_cost(
        [FretPosition(2, 5, 69)],
        [FretPosition(6, 5, 45)],
        cfg,
    )
    assert skip > adjacent


def test_open_string_bonus_lowers_cost():
    prev = [FretPosition(2, 3, 55)]
    pos = [FretPosition(1, 0, 55)]
    without_bonus = _transition_cost(prev, pos, ACOConfig(open_string_bonus=0.0))
    with_bonus = _transition_cost(prev, pos, ACOConfig(open_string_bonus=3.0))
    assert with_bonus < without_bonus


def test_chord_same_string_is_impossible():
    cfg = ACOConfig()
    cost = _chord_cluster_cost(
        [FretPosition(1, 0, 64), FretPosition(1, 3, 67)],
        cfg,
    )
    assert cost >= cfg.impossible_penalty


def test_aco_comparable_to_dp():
    notes = [
        note_from_raw(64, 0, 200, 1, 0, 0.8),
        note_from_raw(67, 250, 200, 1, 3, 0.8),
        note_from_raw(69, 500, 200, 1, 5, 0.8),
    ]
    aco = optimize_sequence_aco(notes, ACOConfig(n_ants=10, n_iterations=15))
    dp = optimize_sequence(notes)
    assert len(aco) == len(dp)
