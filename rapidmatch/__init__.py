"""RapidMatch (RiMatch) — explainable treatment/control matching.

Public surface is deliberately small. Callers configure a match, run it,
and inspect the result. Everything else lives in one-job modules that
`ControlMatcher` wires together.

    from rapidmatch import ControlMatcher, MatchConfig

    result = ControlMatcher(MatchConfig(
        match_vars=["income", "age", "region"],
        treatment_col="is_target",
    )).fit_match(df)
"""

from rapidmatch.config import MatchConfig
from rapidmatch.output.assemble import MatchResult
from rapidmatch.pipeline import ControlMatcher
from rapidmatch.reporting.report import Report

__all__ = ["ControlMatcher", "MatchConfig", "MatchResult", "Report"]
