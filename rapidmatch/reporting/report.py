"""Module 13 — Reporting orchestrator.

Assembles coverage, balance, and drift-log artifacts. Does not compute
balance itself; it receives already-built pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import pandas as pd

from rapidmatch.balance.checker import BalanceRow
from rapidmatch.drift.correct import TrimEvent
from rapidmatch.reporting.build_balance_table import build_balance_table
from rapidmatch.reporting.build_coverage_summary import build_coverage_summary
from rapidmatch.reporting.build_drift_log import build_drift_log


@dataclass
class Report:
    coverage: dict[str, Any]
    balance: pd.DataFrame
    drift_log: pd.DataFrame
    data_profile: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "balance": self.balance.to_dict(orient="records"),
            "drift_log": self.drift_log.to_dict(orient="records"),
            "data_profile": self.data_profile,
        }


def build_report(
    targets: pd.DataFrame,
    cutoff: float,
    before: Sequence[BalanceRow],
    after: Sequence[BalanceRow],
    events: Sequence[TrimEvent],
    data_profile: Optional[dict[str, Any]] = None,
) -> Report:
    return Report(
        coverage=build_coverage_summary(targets, cutoff),
        balance=build_balance_table(before, after),
        drift_log=build_drift_log(events),
        data_profile=data_profile,
    )
