"""Module 2 — Data report.

Profiling is built from scratch (RapidSegment's loader has none). Lazy:
constructing `DataReport` runs nothing. `.summary()` / `.to_dict()` fire one
batched DuckDB query for nulls + target/control counts, plus an
`information_schema` type census (no data scan).
"""

from __future__ import annotations

from typing import Any, Optional

import duckdb

from rapidmatch._sql import quote_ident
from rapidmatch.ingestion.validate import column_types, is_numeric_dtype


class DataReport:
    """Lazy profile of `udl_data` (or another relation) for one treatment col."""

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        treatment_col: str,
        relation: str = "udl_data",
    ) -> None:
        self._con = con
        self._treatment_col = treatment_col
        self._relation = relation
        self._cache: Optional[dict[str, Any]] = None

    def summary(self) -> dict[str, Any]:
        """Run the batched profile query once and cache the result."""
        if self._cache is None:
            self._cache = self._compute()
        return self._cache

    def to_dict(self) -> dict[str, Any]:
        return self.summary()

    def _compute(self) -> dict[str, Any]:
        types = column_types(self._con, self._relation)
        n_numeric = sum(1 for dtype in types.values() if is_numeric_dtype(dtype))
        n_categorical = len(types) - n_numeric
        columns = list(types)
        tcol = quote_ident(self._treatment_col)
        parts = [
            "COUNT(*)",
            f"SUM(CASE WHEN {tcol} IN (1, 1.0, TRUE) THEN 1 ELSE 0 END)",
            f"SUM(CASE WHEN {tcol} IN (0, 0.0, FALSE) THEN 1 ELSE 0 END)",
        ]
        for name in columns:
            ident = quote_ident(name)
            parts.append(f"SUM(CASE WHEN {ident} IS NULL THEN 1 ELSE 0 END)")
        row = self._con.execute(
            f"SELECT {', '.join(parts)} FROM {self._relation}"
        ).fetchone()
        if row is None:
            raise ValueError(f"relation {self._relation!r} is empty")
        n_rows = int(row[0] or 0)
        n_target = int(row[1] or 0)
        n_control = int(row[2] or 0)
        null_counts = {}
        null_rates = {}
        for i, name in enumerate(columns):
            n_null = int(row[3 + i] or 0)
            null_counts[name] = n_null
            null_rates[name] = (n_null / n_rows) if n_rows else 0.0
        return {
            "n_rows": n_rows,
            "n_target": n_target,
            "n_control": n_control,
            "n_numeric": n_numeric,
            "n_categorical": n_categorical,
            "column_types": types,
            "null_counts": null_counts,
            "null_rates": null_rates,
        }
