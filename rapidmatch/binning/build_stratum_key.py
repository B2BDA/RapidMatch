"""Composite stratum key = bins + categories + missingness flags.

Two rows share a stratum only if they look the same on every matching
dimension *including* whether a numeric was originally missing.
"""

from __future__ import annotations

from typing import Sequence

from rapidmatch._sql import quote_ident
from rapidmatch.binning.apply_bucket import bin_column_name
from rapidmatch.missingness.handle_missing import missing_flag_name


def stratum_key_sql(
    numeric_vars: Sequence[str],
    categorical_vars: Sequence[str],
) -> str:
    parts: list[str] = []
    for var in numeric_vars:
        parts.append(f"CAST({quote_ident(bin_column_name(var))} AS VARCHAR)")
    for var in categorical_vars:
        parts.append(quote_ident(var))
    for var in numeric_vars:
        parts.append(f"CAST({quote_ident(missing_flag_name(var))} AS VARCHAR)")
    if not parts:
        return "CAST('all' AS VARCHAR) AS _stratum"
    return f"concat_ws('|', {', '.join(parts)}) AS _stratum"
