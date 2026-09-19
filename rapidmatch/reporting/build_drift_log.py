"""Audit log of rows trimmed by Module 12."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from rapidmatch.drift.correct import TrimEvent


def build_drift_log(events: Sequence[TrimEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=[
                "variable",
                "kind",
                "group",
                "n_removed",
                "target_share",
                "control_share_before",
                "control_rm_ids",
            ]
        )
    return pd.DataFrame(
        [
            {
                "variable": e.variable,
                "kind": e.kind,
                "group": e.group,
                "n_removed": e.n_removed,
                "target_share": e.target_share,
                "control_share_before": e.control_share_before,
                "control_rm_ids": e.control_rm_ids,
            }
            for e in events
        ]
    )
