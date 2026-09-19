"""Shared column extraction helpers for Arrow-backed balance/drift checks.

These convert a PyArrow column into plain NumPy in the shape the pure-python
statistics (JS/KS) and drift logic already expect, so the numerical core is
unchanged and only the import/export surface is Arrow.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def float_values(table: pa.Table, var: str) -> np.ndarray:
    """Column as float64 NumPy; nulls become NaN (dropped by the stats)."""
    col = table[var]
    fp = pc.cast(col, pa.float64())
    return np.asarray(fp.to_numpy(zero_copy_only=False), dtype=np.float64)


def label_values(table: pa.Table, var: str) -> np.ndarray:
    """Categorical column as NumPy strings; nulls become `__MISSING__`."""
    col = table[var]
    filled = pc.fill_null(col, "__MISSING__")
    as_str = pc.cast(filled, pa.string())
    return np.asarray(as_str.to_numpy(zero_copy_only=False), dtype=object)