"""Module 11 — Post-hoc balance validation.

JS (categorical) / KS (numeric) between matched control and the *matched*
target subset. Match_vars are a sanity check (should already be tight).
Monitor_vars are the primary check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from rapidmatch.balance.js_distance import js_distance
from rapidmatch.balance.ks_statistic import ks_statistic
from rapidmatch.ingestion.validate import is_numeric_dtype


@dataclass(frozen=True)
class BalanceRow:
    variable: str
    role: str
    kind: str
    statistic: float
    threshold: float
    flagged: bool


class BalanceChecker:
    """Compare target vs matched-control distributions for a list of vars."""

    def __init__(self, js_threshold: float = 0.10, ks_threshold: float = 0.05) -> None:
        self.js_threshold = js_threshold
        self.ks_threshold = ks_threshold

    def check(
        self,
        target: pd.DataFrame,
        control: pd.DataFrame,
        variables: Sequence[str],
        role: str,
        dtypes: dict[str, str],
    ) -> list[BalanceRow]:
        rows: list[BalanceRow] = []
        for var in variables:
            if var not in target.columns or var not in control.columns:
                continue
            numeric = is_numeric_dtype(dtypes.get(var, ""))
            if numeric:
                stat = ks_statistic(
                    target[var].to_numpy(), control[var].to_numpy()
                )
                threshold = self.ks_threshold
                kind = "ks"
            else:
                stat = js_distance(
                    target[var].astype(str).to_numpy(),
                    control[var].astype(str).to_numpy(),
                )
                threshold = self.js_threshold
                kind = "js"
            rows.append(
                BalanceRow(
                    variable=var,
                    role=role,
                    kind=kind,
                    statistic=stat,
                    threshold=threshold,
                    flagged=stat > threshold,
                )
            )
        return rows


def check_balance(
    target: pd.DataFrame,
    control: pd.DataFrame,
    match_vars: Sequence[str],
    monitor_vars: Sequence[str],
    dtypes: dict[str, str],
    js_threshold: float = 0.10,
    ks_threshold: float = 0.05,
) -> list[BalanceRow]:
    checker = BalanceChecker(js_threshold=js_threshold, ks_threshold=ks_threshold)
    return [
        *checker.check(target, control, match_vars, "match", dtypes),
        *checker.check(target, control, monitor_vars, "monitor", dtypes),
    ]
