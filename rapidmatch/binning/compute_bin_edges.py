"""Quantile bin edges from the TARGET group only.

Control rows later fall into those same buckets. That is what makes the
comparison fair: the grid is defined by who was treated, not by the pool.
"""

from __future__ import annotations

from typing import Sequence

import duckdb

from rapidmatch._sql import quote_ident


def compute_bin_edges(
    con: duckdb.DuckDBPyConnection,
    numeric_vars: Sequence[str],
    n_bins: int,
    source: str = "v_prepared",
) -> dict[str, list[float]]:
    """Return interior quantile cuts per numeric var.

    n_bins=4 -> cuts at 25/50/75th percentiles of the target group.
    """
    if not numeric_vars:
        return {}
    probs = [i / n_bins for i in range(1, n_bins)]
    parts = []
    for var in numeric_vars:
        ident = quote_ident(var)
        for p in probs:
            parts.append(f"quantile_cont({ident}, {p})")
    sql = f"SELECT {', '.join(parts)} FROM {source} WHERE _treatment = 1"
    row = con.execute(sql).fetchone()
    if row is None:
        raise ValueError("no target rows available to compute bin edges")
    edges: dict[str, list[float]] = {}
    width = len(probs)
    for i, var in enumerate(numeric_vars):
        chunk = row[i * width : (i + 1) * width]
        edges[var] = [float(v) if v is not None else 0.0 for v in chunk]
    return edges
