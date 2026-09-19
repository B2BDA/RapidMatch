"""Module 4 — Missing value handling.

Distance math cannot see nulls, but missingness is itself a group we want
to match on. So:

- numeric nulls become 0 (sentinel) AND an `is_missing_<var>` flag is added
  to the stratum key. Missing only matches missing.
- categorical nulls become the literal category `__MISSING__`.
- flags are NEVER used in distance scoring, only in stratification.
"""

from __future__ import annotations

from typing import Sequence

import duckdb

from rapidmatch._sql import quote_ident

MISSING_CATEGORY = "__MISSING__"


def missing_flag_name(var: str) -> str:
    return f"is_missing_{var}"


def handle_missing(
    con: duckdb.DuckDBPyConnection,
    treatment_col: str,
    numeric_vars: Sequence[str],
    categorical_vars: Sequence[str],
    source: str = "v_source",
    dest: str = "v_prepared",
) -> str:
    """Build `v_prepared` with imputed match_vars + `_treatment` 0/1."""
    tcol = quote_ident(treatment_col)
    extra: list[str] = ["_rm_id"]
    extra.append(
        f"""
        CASE
            WHEN {tcol} IN (1, 1.0, TRUE) THEN 1
            ELSE 0
        END AS _treatment
        """
    )
    for var in numeric_vars:
        ident = quote_ident(var)
        extra.append(f"({ident} IS NULL) AS {quote_ident(missing_flag_name(var))}")
        extra.append(f"COALESCE({ident}, 0) AS {ident}")
    for var in categorical_vars:
        ident = quote_ident(var)
        extra.append(
            f"COALESCE(CAST({ident} AS VARCHAR), '{MISSING_CATEGORY}') AS {ident}"
        )

    pass_through = _pass_through_columns(
        con, source, numeric_vars, categorical_vars, treatment_col
    )
    select_sql = ", ".join([*extra, *pass_through])
    con.execute(f"CREATE OR REPLACE VIEW {dest} AS SELECT {select_sql} FROM {source}")
    return dest


def _pass_through_columns(
    con: duckdb.DuckDBPyConnection,
    source: str,
    numeric_vars: Sequence[str],
    categorical_vars: Sequence[str],
    treatment_col: str,
) -> list[str]:
    """Keep non-match columns (monitor vars, ids) untouched."""
    covered = {"_rm_id", treatment_col, *numeric_vars, *categorical_vars}
    cols = con.execute(f"PRAGMA table_info({source})").fetchall()
    out = []
    for row in cols:
        name = row[1]
        if name in covered:
            continue
        out.append(quote_ident(name))
    if treatment_col not in numeric_vars and treatment_col not in categorical_vars:
        out.append(f"{quote_ident(treatment_col)} AS {quote_ident(treatment_col)}")
    return out
