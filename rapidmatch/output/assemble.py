"""Module 10 — Output assembly.

Every target row appears in the result. Uncovered rows are not dropped;
they are labeled via `match_status`. `thin_stratum` is a separate boolean
and can co-occur with `matched`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class MatchResult:
    """What a caller inspects after `fit_match`.

    Attributes:
        pairs: One row per (target, control) match, plus one leftover row
            per unmatched target.
        targets: One row per target with rolled-up status.
        cutoff: Strength quantile that was actually applied.
        coverage_summary: Counts and percent matched.
        report: Module 13 artifacts (coverage / balance / drift log / profile).
    """

    pairs: pd.DataFrame
    targets: pd.DataFrame
    cutoff: float
    coverage_summary: dict[str, Any]
    report: Any = field(default=None)


def assemble(
    target_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    kept: list[tuple[int, int, float, int]],
    below: set[int],
    no_control_ids: set[int],
    thin_ids: set[int],
    cutoff: float,
    id_col: Optional[str] = None,
) -> MatchResult:
    target_lookup = target_frame.set_index("_rm_id")
    control_lookup = control_frame.set_index("_rm_id")
    rows = []
    matched_targets: set[int] = set()
    for t, c, strength, rank in kept:
        matched_targets.add(t)
        trow = target_lookup.loc[t]
        crow = control_lookup.loc[c]
        status = "matched"
        rows.append(
            {
                "target_id": _row_id(trow, t, id_col),
                "control_id": _row_id(crow, c, id_col),
                "target_rm_id": t,
                "control_rm_id": c,
                "stratum": trow["_stratum"],
                "match_strength": strength,
                "match_rank": rank,
                "match_status": status,
                "thin_stratum": t in thin_ids,
            }
        )

    leftover = []
    for t, trow in target_lookup.iterrows():
        t = int(t)
        if t in matched_targets:
            continue
        if t in no_control_ids:
            status = "no_control_available"
        elif t in below:
            status = "below_tolerance"
        else:
            # Eligible but lost every slot to stronger competing pairs.
            status = "unmatched"
        leftover.append(
            {
                "target_id": _row_id(trow, t, id_col),
                "control_id": None,
                "target_rm_id": t,
                "control_rm_id": None,
                "stratum": trow["_stratum"],
                "match_strength": None,
                "match_rank": None,
                "match_status": status,
                "thin_stratum": t in thin_ids,
            }
        )

    pairs = pd.DataFrame(rows + leftover)
    targets = (
        pairs.groupby("target_rm_id", as_index=False)
        .agg(
            target_id=("target_id", "first"),
            stratum=("stratum", "first"),
            n_matches=("control_rm_id", lambda s: int(s.notna().sum())),
            match_status=("match_status", _best_status),
            thin_stratum=("thin_stratum", "max"),
            best_strength=("match_strength", "max"),
        )
    )
    summary = {
        "n_target": int(len(targets)),
        "n_matched": int((targets["match_status"] == "matched").sum()),
        "n_no_control": int((targets["match_status"] == "no_control_available").sum()),
        "n_below_tolerance": int((targets["match_status"] == "below_tolerance").sum()),
        "n_thin_stratum": int(targets["thin_stratum"].sum()),
        "tolerance_cutoff": cutoff,
        "pct_matched": float((targets["match_status"] == "matched").mean())
        if len(targets)
        else 0.0,
    }
    return MatchResult(
        pairs=pairs, targets=targets, cutoff=cutoff, coverage_summary=summary
    )


def _row_id(row: pd.Series, rm_id: int, id_col: Optional[str]) -> Any:
    if id_col and id_col in row.index:
        return row[id_col]
    return rm_id


def _best_status(series: pd.Series) -> str:
    """If any pair matched, the target is matched."""
    values = set(series)
    if "matched" in values:
        return "matched"
    if "below_tolerance" in values:
        return "below_tolerance"
    if "no_control_available" in values:
        return "no_control_available"
    return series.iloc[0]
