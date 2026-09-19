"""Schema checks against `udl_data` before any matching work starts."""

from __future__ import annotations

from typing import Iterable

import duckdb

from rapidmatch._sql import quote_ident
from rapidmatch.config import MatchConfig

# DuckDB type names that we treat as numeric match_vars (bin + z-score).
NUMERIC_TOKENS = ("DOUBLE", "FLOAT", "DECIMAL", "REAL", "INT", "NUMERIC", "HUGEINT")


def column_types(con: duckdb.DuckDBPyConnection, relation: str = "udl_data") -> dict[str, str]:
    """Column name -> DuckDB type. Metadata only; no data scan."""
    rows = con.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = ?
        """,
        [relation],
    ).fetchall()
    return {name: dtype.upper() for name, dtype in rows}


def is_numeric_dtype(dtype: str) -> bool:
    up = dtype.upper()
    return any(token in up for token in NUMERIC_TOKENS)


def classify_match_vars(
    types: dict[str, str], match_vars: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Split match_vars into (numeric, categorical) using DuckDB types."""
    numeric: list[str] = []
    categorical: list[str] = []
    for var in match_vars:
        if is_numeric_dtype(types[var]):
            numeric.append(var)
        else:
            categorical.append(var)
    return numeric, categorical


def validate_schema(
    con: duckdb.DuckDBPyConnection,
    config: MatchConfig,
    treatment_values: Iterable | None = None,
) -> dict[str, str]:
    """Required columns exist, treatment is binary 0/1, both classes present.

    `treatment_values` short-circuits the DISTINCT scan: callers that already
    collected the treatment column (e.g. the bundled data profile) pass it in
    so validation adds no extra table scan.
    """
    types = column_types(con, "udl_data")
    needed = [config.treatment_col, *config.match_vars, *config.monitor_vars]
    if config.id_col:
        needed.append(config.id_col)
    missing = [c for c in needed if c not in types]
    if missing:
        raise ValueError(f"columns not found in input: {missing}")

    if treatment_values is None:
        tcol = quote_ident(config.treatment_col)
        distinct = con.execute(
            f"SELECT DISTINCT {tcol} FROM udl_data WHERE {tcol} IS NOT NULL"
        ).fetchall()
        values = {row[0] for row in distinct}
    else:
        values = {v for v in treatment_values}
    coerced = set()
    for value in values:
        if value in (0, 0.0, False):
            coerced.add(0)
        elif value in (1, 1.0, True):
            coerced.add(1)
        else:
            raise ValueError(
                f"{config.treatment_col} must be binary 0/1; saw {sorted(values)!r}"
            )
    if coerced != {0, 1}:
        raise ValueError(
            f"{config.treatment_col} must contain both 0 and 1; saw {sorted(values)!r}"
        )
    return types
