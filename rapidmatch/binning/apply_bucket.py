"""Turn numeric values into bin indices using target-derived edges."""

from __future__ import annotations

from typing import Sequence

from rapidmatch._sql import quote_ident


def bin_column_name(var: str) -> str:
    return f"_bin_{var}"


def bucket_sql(var: str, edges: Sequence[float], n_bins: int) -> str:
    """CASE expression: value <= edge_i -> bin i, else last bin."""
    ident = quote_ident(var)
    if not edges:
        return f"0 AS {quote_ident(bin_column_name(var))}"
    clauses = []
    for i, edge in enumerate(edges):
        clauses.append(f"WHEN {ident} <= {edge} THEN {i}")
    clauses.append(f"ELSE {n_bins - 1}")
    return (
        "CASE "
        + " ".join(clauses)
        + f" END AS {quote_ident(bin_column_name(var))}"
    )
