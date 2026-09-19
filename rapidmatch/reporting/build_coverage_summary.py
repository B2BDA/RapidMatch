"""Coverage summary artifact: % matched / no control / below tolerance."""

from __future__ import annotations

from typing import Any, Optional

import pyarrow as pa
import pyarrow.compute as pc


def build_coverage_summary(
    targets: pa.Table,
    cutoff: float,
    n_unmatched: Optional[int] = None,
) -> dict[str, Any]:
    n = targets.num_rows
    if n:
        status = targets["match_status"].to_pylist()
        counts = {
            s: int(sum(1 for v in status if v == s)) for s in {"matched", "no_control_available", "below_tolerance", "unmatched"}
        }
        n_matched = counts["matched"]
        n_no = counts["no_control_available"]
        n_below = counts["below_tolerance"]
        if n_unmatched is None:
            n_unmatched = counts["unmatched"]
        if "thin_stratum" in targets.column_names:
            n_thin = int(pc.sum(targets["thin_stratum"]).as_py() or 0)
        else:
            n_thin = 0
    else:
        n_matched = n_no = n_below = 0
        if n_unmatched is None:
            n_unmatched = 0
        n_thin = 0
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