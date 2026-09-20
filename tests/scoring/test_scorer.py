import numpy as np

from rapidmatch.config import MatchConfig
from rapidmatch.scoring.scorer import score_pairs, target_moments


def test_match_strength_bounded() -> None:
    cfg = MatchConfig(match_vars=["x"], treatment_col="t")
    t_ids = np.array([1, 2])
    c_ids = np.array([10, 11])
    tx = np.array([[0.0], [1.0]])
    cx = np.array([[0.0], [2.0]])
    mean, std = target_moments(tx)
    _, _, strength = score_pairs(t_ids, c_ids, tx, cx, ["x"], cfg, mean, std)
    assert np.all(strength > 0)
    assert np.all(strength <= 1)
    assert strength[0] == np.max(strength)


def test_identical_rows_have_strength_one() -> None:
    cfg = MatchConfig(match_vars=["x", "y"], treatment_col="t")
    t_ids = np.array([1])
    c_ids = np.array([2])
    x = np.array([[3.0, 4.0]])
    mean, std = target_moments(x)
    _, _, strength = score_pairs(t_ids, c_ids, x, x, ["x", "y"], cfg, mean, std)
    assert strength[0] == 1.0


def _full_tensor_reference(
    target_ids: np.ndarray,
    control_ids: np.ndarray,
    target_x: np.ndarray,
    control_x: np.ndarray,
    numeric_vars: list[str],
    config: MatchConfig,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array([config.weight_for(v) for v in numeric_vars], dtype=np.float64)
    std = np.where(target_std == 0, 1.0, target_std)
    zt = (target_x - target_mean) / std
    zc = (control_x - target_mean) / std
    zt = zt * weights
    zc = zc * weights
    delta = zt[:, None, :] - zc[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=2))
    strength = np.exp(-dist)
    t_idx, c_idx = np.meshgrid(
        np.arange(len(target_ids)), np.arange(len(control_ids)), indexing="ij"
    )
    return (
        target_ids[t_idx.ravel()],
        control_ids[c_idx.ravel()],
        strength.ravel(),
    )


def test_repeat_tile_order_matches_meshgrid() -> None:
    cfg = MatchConfig(match_vars=["x", "y"], treatment_col="t")
    t_ids = np.array([1, 2])
    c_ids = np.array([10, 11, 12])
    tx = np.array([[0.0, 1.0], [2.0, 3.0]])
    cx = np.array([[0.0, 1.0], [4.0, 5.0], [1.0, 0.0]])
    mean, std = target_moments(tx)
    got = score_pairs(t_ids, c_ids, tx, cx, ["x", "y"], cfg, mean, std)
    ref = _full_tensor_reference(t_ids, c_ids, tx, cx, ["x", "y"], cfg, mean, std)
    for g, r in zip(got, ref):
        np.testing.assert_array_equal(g, r)


def test_chunked_score_equals_full_tensor(monkeypatch) -> None:
    import rapidmatch.scoring.scorer as scorer_mod

    monkeypatch.setattr(scorer_mod, "_MAX_DISTANCE_CELLS", 4)
    cfg = MatchConfig(match_vars=["x", "y"], treatment_col="t")
    t_ids = np.array([1, 2, 3])
    c_ids = np.array([10, 11, 12])
    tx = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    cx = np.array([[0.0, 1.0], [1.0, 0.0], [3.0, 3.0]])
    mean, std = target_moments(tx)
    got = scorer_mod.score_pairs(t_ids, c_ids, tx, cx, ["x", "y"], cfg, mean, std)
    ref = _full_tensor_reference(t_ids, c_ids, tx, cx, ["x", "y"], cfg, mean, std)
    for g, r in zip(got, ref):
        np.testing.assert_array_equal(g, r)


def test_categorical_only_not_chunked() -> None:
    cfg = MatchConfig(match_vars=["region"], treatment_col="t")
    t_ids = np.array([1, 2])
    c_ids = np.array([10, 11, 12])
    tx = np.empty((2, 0), dtype=np.float64)
    cx = np.empty((3, 0), dtype=np.float64)
    mean = np.array([], dtype=np.float64)
    std = np.array([], dtype=np.float64)
    t_out, c_out, strength = score_pairs(
        t_ids, c_ids, tx, cx, [], cfg, mean, std
    )
    np.testing.assert_array_equal(t_out, np.array([1, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(c_out, np.array([10, 11, 12, 10, 11, 12]))
    np.testing.assert_array_equal(strength, np.ones(6, dtype=np.float64))
