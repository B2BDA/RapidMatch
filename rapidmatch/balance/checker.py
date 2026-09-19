"""Module 11 — Post-hoc balance validation.

JS (categorical) / KS (numeric) between matched control and the *matched*
target subset. Match_vars are a sanity check (should already be tight).
Monitor_vars are the primary check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pyarrow as pa

from rapidmatch.balance.columns import float_values, label_values
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
        target: pa.Table,
        control: pa.Table,
        variables: Sequence[str],
        role: str,
        dtypes: Mapping[str, str],
    ) -> list[BalanceRow]:
        rows: list[BalanceRow] = []
        for var in variables:
            if var not in target.column_names or var not in control.column_names:
                continue
            numeric = is_numeric_dtype(dtypes.get(var, ""))
            if numeric:
                stat = ks_statistic(
                    float_values(target, var), float_values(control, var)
                )
                threshold = self.ks_threshold
                kind = "ks"
            else:
                stat = js_distance(
                    label_values(target, var),
                    label_values(control, var),
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
    target: pa.Table,
    control: pa.Table,
    match_vars: Sequence[str],
    monitor_vars: Sequence[str],
    dtypes: Mapping[str, str],
    js_threshold: float = 0.10,
    ks_threshold: float = 0.05,
) -> list[BalanceRow]:
    checker = BalanceChecker(js_threshold=js_threshold, ks_threshold=ks_threshold)
    return [
        *checker.check(target, control, match_vars, "match", dtypes),
        *checker.check(target, control, monitor_vars, "monitor", dtypes),
    ]