"""Match configuration (Module 3).

This is the only place user knobs live. The pipeline never invents defaults
beyond what is declared here, so a reader can understand a run from the
config object alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class MatchConfig:
    """Immutable settings for one matching run.

    Attributes:
        match_vars: Columns used to stratify AND to score distance.
        treatment_col: Binary 0/1 flag. 1 = campaign target, 0 = candidate control.
        monitor_vars: Watched after matching (not used to match). JS/KS balance
            is the primary check; flagged monitor vars can be trim-corrected.
        weights: Per-variable multipliers on z-scored numeric match_vars.
            Missing keys default to 1.0. Unknown keys are an error.
        n: How many control rows each target may receive (1:1 default).
        tolerance: Keep assignments whose strength is at/above this quantile
            of accepted pair strengths. 0 keeps everyone; 0.8 keeps the top 20%.
        min_control_pool_size: Absolute floor of control candidates in a stratum.
            Below this the stratum is flagged `thin_stratum` but still matched.
        min_control_ratio: Optional extra floor: ceil(ratio * n_target_in_stratum).
        n_bins: Quantile bins computed on the target group for numeric match_vars.
        id_col: Optional business id copied into the output. Internal matching
            always uses `_rm_id` regardless.
        js_threshold: Flag a categorical var when JS distance exceeds this
            (default 0.10).
        ks_threshold: Flag a numeric var when the KS statistic exceeds this
            (default 0.05).
        n_workers: Parallel threads used to score strata, or None for serial.
            Order of results is preserved regardless of worker count.
        duckdb_threads: DuckDB execution threads, or None for DuckDB default.
        progress: Show terminal-only progress bars (requires tqdm installed
            via the `progress` extra and a TTY stderr).
    """

    match_vars: Sequence[str]
    treatment_col: str
    monitor_vars: Sequence[str] = field(default_factory=tuple)
    weights: Mapping[str, float] = field(default_factory=dict)
    n: int = 1
    tolerance: float = 0.8
    min_control_pool_size: int = 5
    min_control_ratio: Optional[float] = None
    n_bins: int = 4
    id_col: Optional[str] = None
    js_threshold: float = 0.10
    ks_threshold: float = 0.05
    n_workers: Optional[int] = None
    duckdb_threads: Optional[int] = None
    progress: bool = False

    def __post_init__(self) -> None:
        # Frozen dataclass: tuples/dicts so callers cannot mutate after construct.
        object.__setattr__(self, "match_vars", tuple(self.match_vars))
        object.__setattr__(self, "monitor_vars", tuple(self.monitor_vars))
        object.__setattr__(self, "weights", dict(self.weights))
        _validate(self)

    def weight_for(self, var: str) -> float:
        """Weight used in distance scoring. Unspecified vars count as 1."""
        return float(self.weights.get(var, 1.0))


def _validate(cfg: MatchConfig) -> None:
    """Fail fast with messages that name the bad field."""
    if not cfg.match_vars:
        raise ValueError("match_vars must be a non-empty sequence")
    if len(set(cfg.match_vars)) != len(cfg.match_vars):
        raise ValueError("match_vars contains duplicates")
    if len(set(cfg.monitor_vars)) != len(cfg.monitor_vars):
        raise ValueError("monitor_vars contains duplicates")
    overlap = set(cfg.match_vars) & set(cfg.monitor_vars)
    if overlap:
        raise ValueError(f"match_vars and monitor_vars overlap: {sorted(overlap)}")
    unknown_weights = set(cfg.weights) - set(cfg.match_vars)
    if unknown_weights:
        raise ValueError(
            f"weights keys are not in match_vars: {sorted(unknown_weights)}"
        )
    if not cfg.treatment_col:
        raise ValueError("treatment_col is required")
    if cfg.treatment_col in cfg.match_vars or cfg.treatment_col in cfg.monitor_vars:
        raise ValueError("treatment_col cannot also be a match_var or monitor_var")
    if cfg.n < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 <= cfg.tolerance <= 1.0:
        raise ValueError("tolerance must be in [0, 1]")
    if cfg.min_control_pool_size < 1:
        raise ValueError("min_control_pool_size must be >= 1")
    if cfg.min_control_ratio is not None and cfg.min_control_ratio <= 0:
        raise ValueError("min_control_ratio must be > 0 when provided")
    if cfg.n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if cfg.id_col is not None and cfg.id_col == cfg.treatment_col:
        raise ValueError("id_col cannot be treatment_col")
    if not 0.0 <= cfg.js_threshold <= 1.0:
        raise ValueError("js_threshold must be in [0, 1]")
    if not 0.0 <= cfg.ks_threshold <= 1.0:
        raise ValueError("ks_threshold must be in [0, 1]")
    if cfg.n_workers is not None and cfg.n_workers < 1:
        raise ValueError("n_workers must be >= 1 when provided")
    if cfg.duckdb_threads is not None and cfg.duckdb_threads < 1:
        raise ValueError("duckdb_threads must be >= 1 when provided")
