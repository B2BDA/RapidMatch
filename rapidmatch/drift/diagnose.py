"""Identify over-represented groups on a flagged monitor_var.

Numeric vars are split into quantile bins of the *target* so we can point
at a concrete bin rather than a continuous range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from rapidmatch.balance.columns import float_values, label_values


@dataclass(frozen=True)
class ExcessGroup:
    variable: str
    kind: str
    group: Any
    target_share: float
    control_share: float
    excess_rows: int


def excess_groups(
    target: pa.Table,
    control: pa.Table,
    variable: str,
    kind: str,
    n_bins: int = 4,
) -> list[ExcessGroup]:
    """Groups in control whose share exceeds the target's share."""
    if kind == "js":
        t_vals = label_values(target, variable)
        c_vals = label_values(control, variable)
        t_vc = _value_counts(t_vals)
        c_vc = _value_counts(c_vals)
        labels = sorted(set(t_vals) | set(c_vals))
        n_control = len(c_vals)
        out = []
        for lab in labels:
            ts = float(t_vc.get(lab, 0.0))
            cs = float(c_vc.get(lab, 0.0))
            excess = int(np.floor(max(0.0, (cs - ts) * n_control)))
            if excess > 0:
                out.append(ExcessGroup(variable, kind, lab, ts, cs, excess))
        return sorted(out, key=lambda g: g.excess_rows, reverse=True)

    t_num = float_values(target, variable)
    t_num = t_num[np.isfinite(t_num)]
    if len(t_num) == 0 or control.num_rows == 0:
        return []
    probs = [i / n_bins for i in range(1, n_bins)]
    edges = np.unique(np.quantile(t_num, probs))
    t_bins = np.digitize(float_values(target, variable), edges, right=True)
    c_bins = np.digitize(float_values(control, variable), edges, right=True)
    n_control = len(c_bins)
    n_obs_bins = int(c_bins.max()) + 1 if len(c_bins) else 0
    out = []
    for b in range(max(n_bins, n_obs_bins)):
        ts = float((t_bins == b).mean()) if len(t_bins) else 0.0
        cs = float((c_bins == b).mean()) if len(c_bins) else 0.0
        excess = int(np.floor(max(0.0, (cs - ts) * n_control)))
        if excess > 0:
            out.append(ExcessGroup(variable, kind, int(b), ts, cs, excess))
    return sorted(out, key=lambda g: g.excess_rows, reverse=True)


def _value_counts(values: np.ndarray) -> dict[str, float]:
    """Label -> relative frequency. NaN keys collapse under `__MISSING__`."""
    vc = pa.compute.value_counts(pa.array(values, type=pa.string()))
    pairs = vc.to_pylist()
    total = sum(int(p["counts"]) for p in pairs) or 1
    return {p["values"]: int(p["counts"]) / total for p in pairs}