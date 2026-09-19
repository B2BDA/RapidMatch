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
