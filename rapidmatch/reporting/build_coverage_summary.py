"""Coverage summary artifact: % matched / no control / below tolerance."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def build_coverage_summary(
    targets: pd.DataFrame,
    cutoff: float,
    n_unmatched: Optional[int] = None,
) -> dict[str, Any]:
    n = int(len(targets))
    n_matched = int((targets["match_status"] == "matched").sum()) if n else 0
    n_no = int((targets["match_status"] == "no_control_available").sum()) if n else 0
    n_below = int((targets["match_status"] == "below_tolerance").sum()) if n else 0
    if n_unmatched is None:
        n_unmatched = int((targets["match_status"] == "unmatched").sum()) if n else 0
    n_thin = int(targets["thin_stratum"].sum()) if n and "thin_stratum" in targets else 0
    return {
        "n_target": n,
        "n_matched": n_matched,
        "n_no_control": n_no,
        "n_below_tolerance": n_below,
        "n_unmatched": int(n_unmatched),
        "n_thin_stratum": n_thin,
        "pct_matched": (n_matched / n) if n else 0.0,
        "pct_no_control": (n_no / n) if n else 0.0,
        "pct_below_tolerance": (n_below / n) if n else 0.0,
        "tolerance_cutoff": cutoff,
    }
