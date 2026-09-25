"""Tests for the opt-in `max_candidates_per_target` cap."""

import numpy as np
import pytest

from rapidmatch.config import MatchConfig
from rapidmatch.matching.greedy_match import greedy_match
from rapidmatch.scoring.scorer import score_pairs, target_moments


def test_cap_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="max_candidates_per_target"):
        MatchConfig(
            match_vars=["x"],
            treatment_col="t",
            max_candidates_per_target=0,
        )


def test_cap_defaults_to_none() -> None:
    cfg = MatchConfig(match_vars=["x"], treatment_col="t")
    assert cfg.max_candidates_per_target is None


def test_cap_bounds_pairs_per_target() -> None:
    cfg = MatchConfig(
        match_vars=["x"],
        treatment_col="t",
        max_candidates_per_target=50,
    )
    t_ids = np.array([1])
    c_ids = np.arange(1000, 6000, dtype=np.int64)
    tx = np.array([[0.0]])
    cx = np.arange(5000, dtype=np.float64).reshape(-1, 1)
    mean, std = target_moments(tx)
    t, c, s = score_pairs(t_ids, c_ids, tx, cx, ["x"], cfg, mean, std)
    assert len(t) == len(c) == len(s) == 50
    np.testing.assert_array_equal(t, np.full(50, 1))
    assert len(set(c.tolist())) == 50


def test_cap_keeps_the_closest_controls() -> None:
    cfg = MatchConfig(
        match_vars=["x"],
        treatment_col="t",
        max_candidates_per_target=10,
    )
    t_ids = np.array([1])
    c_ids = np.arange(1, 101, dtype=np.int64)
    tx = np.array([[0.0]])
    cx = np.arange(100, dtype=np.float64).reshape(-1, 1)
    mean, std = target_moments(tx)
    _, c, s = score_pairs(t_ids, c_ids, tx, cx, ["x"], cfg, mean, std)
    # Target sits at 0.0, so the 10 closest controls are x = 0..9.
    assert set(c.tolist()) == set(range(1, 11))
    assert s.max() == pytest.approx(1.0)


def test_nearest_control_is_never_pruned() -> None:
    """The single best control per target must always survive the cap."""
    cfg = MatchConfig(
        match_vars=["x"],
        treatment_col="t",
        max_candidates_per_target=5,
    )
    t_ids = np.array([1, 2, 3])
    c_ids = np.arange(100, dtype=np.int64)
    tx = np.array([[10.0], [20.0], [30.0]])
    cx = np.arange(100, dtype=np.float64).reshape(-1, 1)
    mean, std = target_moments(tx)
    _, c, s = score_pairs(t_ids, c_ids, tx, cx, ["x"], cfg, mean, std)
    assert len(s) == 15
    for row, target_value in enumerate((10.0, 20.0, 30.0)):
        block_c = c[row * 5 : (row + 1) * 5]
        block_s = s[row * 5 : (row + 1) * 5]
        best = int(np.argmax(block_s))
        assert cx[int(block_c[best]), 0] == target_value


def test_cap_below_control_count_preserves_assignments() -> None:
    """With uncontested nearest neighbours, pruning must not change matches."""
    rng = np.random.default_rng(11)
    n_t, n_c = 60, 400
    tx = rng.normal(size=(n_t, 1)) * 10.0
    cx = tx[rng.integers(0, n_t, size=n_c)] + rng.normal(size=(n_c, 1))
    t_ids = np.arange(1, n_t + 1, dtype=np.int64)
    c_ids = np.arange(10_000, 10_000 + n_c, dtype=np.int64)
    mean, std = target_moments(tx)

    full_cfg = MatchConfig(match_vars=["x"], treatment_col="t")
    capped_cfg = MatchConfig(
        match_vars=["x"], treatment_col="t", max_candidates_per_target=25
    )
    full = score_pairs(t_ids, c_ids, tx, cx, ["x"], full_cfg, mean, std)
    capped = score_pairs(t_ids, c_ids, tx, cx, ["x"], capped_cfg, mean, std)

    full_assign = greedy_match(*full, n=1)
    capped_assign = greedy_match(*capped, n=1)
    assert len(capped_assign) == len(full_assign) == n_t
    assert full_assign == capped_assign


def test_small_stratum_is_not_pruned() -> None:
    """Fewer controls than the cap must pass through with original order."""
    cfg = MatchConfig(
        match_vars=["x"],
        treatment_col="t",
        max_candidates_per_target=50,
    )
    t_ids = np.array([1, 2])
    c_ids = np.array([10, 11, 12], dtype=np.int64)
    tx = np.array([[0.0], [1.0]])
    cx = np.array([[0.0], [2.0], [1.0]])
    mean, std = target_moments(tx)
    t, c, _ = score_pairs(t_ids, c_ids, tx, cx, ["x"], cfg, mean, std)
    np.testing.assert_array_equal(t, np.array([1, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(c, np.array([10, 11, 12, 10, 11, 12]))


def test_cap_applies_to_chunked_path(monkeypatch) -> None:
    import rapidmatch.scoring.scorer as scorer_mod

    monkeypatch.setattr(scorer_mod, "_MAX_DISTANCE_CELLS", 4)
    cfg = MatchConfig(
        match_vars=["x", "y"],
        treatment_col="t",
        max_candidates_per_target=3,
    )
    t_ids = np.array([1, 2, 3])
    c_ids = np.array([10, 11, 12], dtype=np.int64)
    tx = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    cx = np.array([[0.0, 1.0], [1.0, 0.0], [3.0, 3.0]])
    mean, std = target_moments(tx)
    t, c, s = score_pairs(t_ids, c_ids, tx, cx, ["x", "y"], cfg, mean, std)
    assert len(t) == len(c) == len(s) == 9
    for row in range(3):
        assert set(c[row * 3 : (row + 1) * 3].tolist()) == {10, 11, 12}
    assert np.all(s > 0)
    assert np.all(s <= 1.0)


def test_cap_applies_to_categorical_only_stratum() -> None:
    cfg = MatchConfig(
        match_vars=["region"],
        treatment_col="t",
        max_candidates_per_target=4,
    )
    t_ids = np.array([1, 2])
    c_ids = np.arange(10, 25, dtype=np.int64)
    tx = np.empty((2, 0), dtype=np.float64)
    cx = np.empty((15, 0), dtype=np.float64)
    mean = np.array([], dtype=np.float64)
    std = np.array([], dtype=np.float64)
    t, c, s = score_pairs(t_ids, c_ids, tx, cx, [], cfg, mean, std)
    assert len(t) == len(c) == len(s) == 8
    np.testing.assert_array_equal(s, np.ones(8, dtype=np.float64))
    assert set(c.tolist()).issubset(set(c_ids.tolist()))


def test_cap_prunes_deterministically() -> None:
    cfg = MatchConfig(
        match_vars=["x"],
        treatment_col="t",
        max_candidates_per_target=7,
    )
    t_ids = np.array([1, 2])
    c_ids = np.arange(100, dtype=np.int64)
    tx = np.array([[0.0], [50.0]])
    cx = np.arange(100, dtype=np.float64).reshape(-1, 1)
    mean, std = target_moments(tx)
    first = score_pairs(t_ids, c_ids, tx, cx, ["x"], cfg, mean, std)
    second = score_pairs(t_ids, c_ids, tx, cx, ["x"], cfg, mean, std)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
