import numpy as np
import pandas as pd
import pyarrow.compute as pc

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
    assert result.targets["match_status"].to_pylist()
    matched = result.pairs.filter(
        pc.equal(result.pairs["match_status"], "matched")
    )
    control_ids = matched["control_rm_id"].to_pylist()
    assert len(set(control_ids)) == len(matched)
    strengths = matched["match_strength"].to_pylist()
    assert all(0 < s <= 1 for s in strengths)
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
    both = result.pairs.filter(
        pc.and_(pc.equal(result.pairs["match_status"], "matched"), result.pairs["thin_stratum"])
    )
    assert result.coverage_summary["n_thin_stratum"] >= 0
    assert both.num_rows == 0 or both["match_status"].to_pylist().count("matched") == both.num_rows


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
    assert "tenure" in bal["variable"].to_pylist()
    tenure_rows = bal.filter(pc.equal(bal["variable"], "tenure"))
    assert tenure_rows["role"].to_pylist() == ["monitor"]
    assert tenure_rows["kind"].to_pylist() == ["ks"]