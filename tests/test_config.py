import pytest

from rapidmatch.config import MatchConfig


def test_rejects_overlapping_vars() -> None:
    with pytest.raises(ValueError, match="overlap"):
        MatchConfig(
            match_vars=["income", "age"],
            monitor_vars=["age"],
            treatment_col="is_target",
        )


def test_rejects_unknown_weight_keys() -> None:
    with pytest.raises(ValueError, match="weights"):
        MatchConfig(
            match_vars=["income"],
            treatment_col="is_target",
            weights={"tenure": 2.0},
        )


def test_defaults() -> None:
    cfg = MatchConfig(match_vars=["income"], treatment_col="is_target")
    assert cfg.n == 1
    assert cfg.weight_for("income") == 1.0
    assert cfg.min_control_pool_size == 5
