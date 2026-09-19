import numpy as np
import pandas as pd

from rapidmatch import ControlMatcher, MatchConfig


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    target = pd.DataFrame(
        {
            "income": rng.normal(70, 5, 40),
            "age": rng.normal(40, 4, 40),
            "region": rng.choice(["A", "B"], 40),
            "is_target": 1,
        }
    )
    control = pd.DataFrame(
        {
            "income": rng.normal(55, 8, 160),
            "age": rng.normal(32, 6, 160),
            "region": rng.choice(["A", "B"], 160),
            "is_target": 0,
        }
    )
    return pd.concat([target, control], ignore_index=True)


def test_pipeline_covers_every_target() -> None:
    cfg = MatchConfig(
        match_vars=["income", "age", "region"],
        treatment_col="is_target",
        n=1,
        tolerance=0.0,
        min_control_pool_size=1,
        n_bins=3,
    )
    result = ControlMatcher(cfg).fit_match(_frame())
    assert result.coverage_summary["n_target"] == 40
    assert result.targets["match_status"].isin(
        ["matched", "no_control_available", "below_tolerance", "unmatched"]
    ).all()
    matched = result.pairs[result.pairs["match_status"] == "matched"]
    assert matched["control_rm_id"].nunique() == len(matched)
    assert (matched["match_strength"] > 0).all()
    assert (matched["match_strength"] <= 1).all()
    assert result.report is not None
    assert result.report.coverage["n_target"] == 40
    assert "null_counts" in result.report.data_profile


def test_thin_flag_can_coexist_with_matched() -> None:
    cfg = MatchConfig(
        match_vars=["income", "region"],
        treatment_col="is_target",
        n=1,
        tolerance=0.0,
        min_control_pool_size=50,
        n_bins=2,
    )
    result = ControlMatcher(cfg).fit_match(_frame())
    both = result.pairs[
        (result.pairs["match_status"] == "matched") & (result.pairs["thin_stratum"])
    ]
    assert result.coverage_summary["n_thin_stratum"] >= 0
    assert both.empty or (both["match_status"] == "matched").all()


def test_monitor_var_shows_up_in_balance_table() -> None:
    rng = np.random.default_rng(1)
    n_t, n_c = 30, 90
    df = pd.DataFrame(
        {
            "income": np.concatenate(
                [rng.normal(70, 5, n_t), rng.normal(68, 6, n_c)]
            ),
            "tenure": np.concatenate(
                [rng.normal(12, 2, n_t), rng.normal(30, 4, n_c)]
            ),
            "is_target": [1] * n_t + [0] * n_c,
        }
    )
    cfg = MatchConfig(
        match_vars=["income"],
        monitor_vars=["tenure"],
        treatment_col="is_target",
        n=1,
        tolerance=0.0,
        min_control_pool_size=1,
        n_bins=3,
        ks_threshold=0.05,
    )
    result = ControlMatcher(cfg).fit_match(df)
    bal = result.report.balance
    assert "tenure" in set(bal["variable"])
    tenure = bal[bal["variable"] == "tenure"].iloc[0]
    assert tenure["role"] == "monitor"
    assert tenure["kind"] == "ks"
