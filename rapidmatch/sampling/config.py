"""Configuration and validation for the downsampling entry point."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence
import math
import secrets


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _columns(values, name):
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of column names, not a string")
    values = tuple(values)
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{name} must contain non-empty column names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


@dataclass(frozen=True)
class SamplingConfig:
    sample_size: int
    label_col: str | None = None
    stratify_vars: Sequence[str] = ()
    check_vars: Sequence[str] = ()
    n_bins: int = 4
    random_state: int | None = None
    ks_threshold: float = 0.05
    js_threshold: float = 0.10
    check_by_label: bool = False
    duckdb_threads: int | None = None
    max_report_rows: int = 1000

    def __post_init__(self):
        for name, minimum in (("sample_size", 1), ("n_bins", 2), ("max_report_rows", 1)):
            object.__setattr__(self, name, positive_integer(getattr(self, name), name, minimum))
        if self.label_col is not None and (not isinstance(self.label_col, str) or not self.label_col):
            raise ValueError("label_col must be a non-empty column name or None")
        for name in ("stratify_vars", "check_vars"):
            cols = _columns(getattr(self, name), name)
            object.__setattr__(self, name, tuple(v for v in cols if v != self.label_col))
        for name in ("ks_threshold", "js_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        if self.duckdb_threads is not None:
            object.__setattr__(self, "duckdb_threads", positive_integer(self.duckdb_threads, "duckdb_threads"))
        if not isinstance(self.check_by_label, bool):
            raise ValueError("check_by_label must be boolean")
        if self.check_by_label and self.label_col is None:
            raise ValueError("check_by_label requires label_col")
        seed = secrets.randbits(63) if self.random_state is None else positive_integer(self.random_state, "random_state", 0)
        if seed >= 2**63:
            raise ValueError("random_state must be less than 2**63")
        object.__setattr__(self, "random_state", seed)

    @property
    def features(self):
        return tuple(dict.fromkeys([*self.stratify_vars, *self.check_vars]))
