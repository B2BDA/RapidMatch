from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from rapidmatch import ControlMatcher, MatchConfig


def make_demo_frame(seed: int = 42) -> pd.DataFrame:
    """Target cluster plus a lookalike control pool and a shifted distractor pool."""
    rng = np.random.default_rng(seed)
    n_target = 200
    n_like = 400
    n_other = 400
    target = pd.DataFrame(
        {
            "id": np.arange(n_target, dtype=int),
            "income": rng.normal(70000, 8000, n_target),
            "age": rng.normal(42, 6, n_target),
            "region": rng.choice(["N", "S", "E", "W"], n_target),
            "is_target": 1,
        }
    )
    like = pd.DataFrame(
        {
            "id": np.arange(n_target, n_target + n_like, dtype=int),
            "income": rng.normal(70000, 8000, n_like),
            "age": rng.normal(42, 6, n_like),
            "region": rng.choice(["N", "S", "E", "W"], n_like),
            "is_target": 0,
        }
    )
    other = pd.DataFrame(
        {
            "id": np.arange(n_target + n_like, n_target + n_like + n_other, dtype=int),
            "income": rng.normal(45000, 9000, n_other),
            "age": rng.normal(30, 7, n_other),
            "region": rng.choice(["N", "S", "E", "W"], n_other),
            "is_target": 0,
        }
    )
    return pd.concat([target, like, other], ignore_index=True)


def main() -> None:
    df = make_demo_frame()
    config = MatchConfig(
        match_vars=["income", "age", "region"],
        treatment_col="is_target",
        id_col="id",
        weights={"income": 1.5, "age": 1.0},
        n=1,
        tolerance=0.2,
        min_control_pool_size=3,
        n_bins=4,
        # Opt-in scaling knobs: serial + no bars by default.
        n_workers=2,
        duckdb_threads=4,
        progress=True,
    )
    result = ControlMatcher(config).fit_match(df)
    print("coverage")
    for key, value in result.coverage_summary.items():
        print(f"  {key}: {value}")
    print("\nstatus counts")
    statuses = result.targets["match_status"].to_pylist()
    for status in dict.fromkeys(statuses):
        print(f"  {status}: {statuses.count(status)}")
    print("\nmatched pairs (top 10)")
    pairs = result.pairs
    matched = pairs.filter(pc.equal(pairs["match_status"], "matched"))
    cols = ["target_id", "control_id", "match_strength", "match_rank"]
    for row in matched.select(cols).to_pylist()[:10]:
        print(row)


if __name__ == "__main__":
    main()