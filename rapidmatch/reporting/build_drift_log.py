"""Audit log of rows trimmed by Module 12."""

from __future__ import annotations

from typing import Sequence

import pyarrow as pa

from rapidmatch.drift.correct import TrimEvent


def build_drift_log(events: Sequence[TrimEvent]) -> pa.Table:
    cols = {
        "variable": [],
        "kind": [],
        "group": [],
        "n_removed": [],
        "target_share": [],
        "control_share_before": [],
        "control_rm_ids": [],
    }
    for e in events:
        cols["variable"].append(e.variable)
        cols["kind"].append(e.kind)
        cols["group"].append(str(e.group))
        cols["n_removed"].append(e.n_removed)
        cols["target_share"].append(e.target_share)
        cols["control_share_before"].append(e.control_share_before)
        cols["control_rm_ids"].append(e.control_rm_ids)

    if not events:
        return pa.table(
            {
                "variable": pa.array([], type=pa.string()),
                "kind": pa.array([], type=pa.string()),
                "group": pa.array([], type=pa.string()),
                "n_removed": pa.array([], type=pa.int64()),
                "target_share": pa.array([], type=pa.float64()),
                "control_share_before": pa.array([], type=pa.float64()),
                "control_rm_ids": pa.array([], type=pa.list_(pa.int64())),
            }
        )
    return pa.table(cols)