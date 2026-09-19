import numpy as np

from rapidmatch.balance.js_distance import js_distance
from rapidmatch.balance.ks_statistic import ks_statistic


def test_identical_samples_are_zero() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cats = np.array(["A", "B", "A", "B"])
    assert ks_statistic(x, x) == 0.0
    assert js_distance(cats, cats) == 0.0


def test_shifted_and_unbalanced_are_positive() -> None:
    left = np.array([0.0, 1.0, 2.0, 3.0])
    right = np.array([10.0, 11.0, 12.0, 13.0])
    assert ks_statistic(left, right) > 0.5
    assert js_distance(np.array(["A", "A", "A"]), np.array(["B", "B", "B"])) > 0.3
