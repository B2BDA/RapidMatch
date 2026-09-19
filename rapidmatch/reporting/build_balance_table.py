"""Before/after balance table for match_vars and monitor_vars."""

from __future__ import annotations

from typing import Sequence

import pyarrow as pa

from rapidmatch.balance.checker import BalanceRow


def build_balance_table(
    before: Sequence[BalanceRow],
    after: Sequence[BalanceRow],
) -> pa.Table:
    before_map = {(r.variable, r.role): r for r in before}
    after_map = {(r.variable, r.role): r for r in after}
    keys = list(dict.fromkeys([*before_map, *after_map]))

    cols = {
        "variable": [],
        "role": [],
        "kind": [],
        "before": [],
        "after": [],
        "threshold": [],
        "flagged_before": [],
        "flagged_after": [],
    }
    for key in keys:
        b = before_map.get(key)
        a = after_map.get(key)
        src = a or b
        cols["variable"].append(src.variable)
        cols["role"].append(src.role)
        cols["kind"].append(src.kind)
        cols["before"].append(None if b is None else b.statistic)
        cols["after"].append(None if a is None else a.statistic)
        cols["threshold"].append(src.threshold)
        cols["flagged_before"].append(None if b is None else b.flagged)
        cols["flagged_after"].append(None if a is None else a.flagged)
    return (
        pa.table(cols)
        if cols["variable"]
        else pa.table(
            {
                "variable": pa.array([], type=pa.string()),
                "role": pa.array([], type=pa.string()),
                "kind": pa.array([], type=pa.string()),
                "before": pa.array([], type=pa.float64()),
                "after": pa.array([], type=pa.float64()),
                "threshold": pa.array([], type=pa.float64()),
                "flagged_before": pa.array([], type=pa.bool_()),
                "flagged_after": pa.array([], type=pa.bool_()),
            }
        )
    )