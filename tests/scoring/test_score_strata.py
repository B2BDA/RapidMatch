import numpy as np

from rapidmatch.config import MatchConfig
from rapidmatch.scoring.score_strata import score_all_strata


def _setup() -> tuple:
    rng = np.random.default_rng(7)
    n = 200
    ids = np.arange(1, n + 1)
    treatment = np.zeros(n, dtype=np.int64)
    treatment[::3] = 1
    strata = np.array(["c" if i % 5 == 0 else "a" if i % 2 else "b" for i in range(n)])
    x = rng.normal(size=(n, 1))
    return ids, treatment, strata, x


def test_parallel_matches_serial_exactly() -> None:
    ids, treatment, strata, x = _setup()
    eligible = ["a", "b", "c"]
    cfg = MatchConfig(match_vars=["x"], treatment_col="t")
    mean, std = np.array([0.0]), np.array([1.0])
    serial = score_all_strata(
        eligible, strata, ids, treatment, x, ["x"], cfg, mean, std
    )
    parallel = score_all_strata(
        eligible, strata, ids, treatment, x, ["x"], cfg, mean, std, n_workers=2
    )
    assert len(serial) == len(parallel) == 3
    for s_arr, p_arr in zip(serial, parallel):
        np.testing.assert_array_equal(s_arr, p_arr)


def test_order_is_deterministic_across_worker_counts() -> None:
    ids, treatment, strata, x = _setup()
    eligible = ["c", "a", "b"]
    cfg = MatchConfig(match_vars=["x"], treatment_col="t")
    mean, std = np.array([0.0]), np.array([1.0])
    r1 = score_all_strata(
        eligible, strata, ids, treatment, x, ["x"], cfg, mean, std, n_workers=3
    )
    r2 = score_all_strata(
        eligible, strata, ids, treatment, x, ["x"], cfg, mean, std, n_workers=1
    )
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)


def test_empty_strata_returns_empty() -> None:
    ids, treatment, strata, x = _setup()
    cfg = MatchConfig(match_vars=["x"], treatment_col="t")
    mean, std = np.array([0.0]), np.array([1.0])
    t, c, s = score_all_strata([], strata, ids, treatment, x, ["x"], cfg, mean, std)
    assert len(t) == len(c) == len(s) == 0