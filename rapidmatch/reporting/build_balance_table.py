"""Before/after balance table for match_vars and monitor_vars."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from rapidmatch.balance.checker import BalanceRow


def build_balance_table(
    before: Sequence[BalanceRow],
    after: Sequence[BalanceRow],
) -> pd.DataFrame:
    before_map = {(r.variable, r.role): r for r in before}
    after_map = {(r.variable, r.role): r for r in after}
    keys = list(dict.fromkeys([*before_map, *after_map]))
    rows = []
    for key in keys:
        b = before_map.get(key)
        a = after_map.get(key)
        src = a or b
        rows.append(
            {
                "variable": src.variable,
                "role": src.role,
                "kind": src.kind,
                "before": None if b is None else b.statistic,
                "after": None if a is None else a.statistic,
                "threshold": src.threshold,
                "flagged_before": None if b is None else b.flagged,
                "flagged_after": None if a is None else a.flagged,
            }
        )
    return pd.DataFrame(rows)
