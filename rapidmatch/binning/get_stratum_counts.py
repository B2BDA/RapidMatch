"""One GROUP BY: target count and control count per stratum."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb


@dataclass(frozen=True)
class StratumCount:
    stratum: str
    n_target: int
    n_control: int


def get_stratum_counts(
    con: duckdb.DuckDBPyConnection,
    source: str = "v_stratified",
) -> list[StratumCount]:
    rows = con.execute(
        f"""
        SELECT
            _stratum,
            CAST(SUM(CASE WHEN _treatment = 1 THEN 1 ELSE 0 END) AS INTEGER) AS n_target,
            CAST(SUM(CASE WHEN _treatment = 0 THEN 1 ELSE 0 END) AS INTEGER) AS n_control
        FROM {source}
        GROUP BY 1
        """
    ).fetchall()
    return [
        StratumCount(stratum=str(s), n_target=int(nt), n_control=int(nc))
        for s, nt, nc in rows
    ]
