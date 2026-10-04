import json

import pyarrow as pa
import pytest

from rapidmatch import ControlMatcher, MatchConfig, InsufficientStratumCapacityWarning
from rapidmatch.binning.get_stratum_counts import StratumCount
from rapidmatch.coverage.capacity import build_capacity


def test_capacity_distinguishes_thin_shortfall_and_global_surplus():
    counts = [StratumCount("a", 100, 50), StratumCount("b", 2, 3),
              StratumCount("c", 5, 0), StratumCount("d", 0, 500)]
    cfg = MatchConfig(match_vars=["x"], treatment_col="t", n=2)
    report = build_capacity(counts, cfg, {})
    s = report.summary
    assert s["n_control"] == 553
    assert s["global_assignment_shortfall"] == 0
    assert s["assignment_shortfall"] == 161
    assert s["assignment_capacity_ceiling"] == 53
    assert s["fully_supplied_targets_ceiling"] == 26
    assert s["targets_with_one_control_ceiling"] == 52
    rows = {r["stratum"]: r for r in report.strata.to_pylist()}
    assert not rows["a"]["thin_stratum"]
    assert rows["b"]["thin_stratum"]
    assert rows["c"]["no_control"]
    assert s["candidate_pairs"] == 5006
    assert json.loads(rows["a"]["group_values"]) == {}


def test_assess_trials_have_measured_advice_without_scoring(monkeypatch):
    import rapidmatch.pipeline as pipeline
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not score or pull full data")
    monkeypatch.setattr(pipeline, "score_all_strata", forbidden)
    monkeypatch.setattr(ControlMatcher, "_pull", forbidden)
    data = pa.table({"x": [0., 1., 2., 3., 0., 0., 2., 2.], "t": [1]*4 + [0]*4})
    cfg = MatchConfig(match_vars=["x"], treatment_col="t", n_bins=4)
    matcher = ControlMatcher(cfg)
    report = matcher.assess(data, n_bins_candidates=[2], max_report_strata=1)
    assert report.summary["coverage_ceiling"] == .5
    trial = report.trials.to_pylist()[1]
    assert trial["coverage_ceiling"] == 1
    assert trial["candidate_pairs"] == 8
    assert report.recommendations[0]["proposed_value"] == 2
    assert report.recommendations[0]["evidence"] == "measured_preflight"
    assert report.summary["strata_truncated"]
    assert matcher.config is cfg and matcher.bin_edges == {} and matcher.cutoff is None
    assert "x" in json.loads(report.strata["group_values"][0].as_py())


def test_matching_warns_before_scoring_and_attaches_report(monkeypatch):
    import rapidmatch.pipeline as pipeline
    original = pipeline.score_all_strata
    seen = []
    monkeypatch.setattr("rapidmatch.coverage.capacity.warnings.warn", lambda *a, **k: seen.append(a))
    def score(*args, **kwargs):
        assert any(a[1] is InsufficientStratumCapacityWarning for a in seen)
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "score_all_strata", score)
    data = pa.table({"g": ["a", "a", "b", "b"], "t": [1, 1, 0, 0]})
    result = ControlMatcher(MatchConfig(match_vars=["g"], treatment_col="t")).fit_match(data)
    assert result.report.capacity.summary["assignment_shortfall"] == 2
    assert result.report.to_dict()["capacity"]["summary"]["coverage_ceiling"] == 0


def test_grouping_subset_preserves_scoring_and_missing_constraints():
    data = pa.table({"id": list(range(8)), "g": ["a"]*8,
                     "x": [0., 10., None, 20., 1., 11., None, 21.],
                     "t": [1]*4 + [0]*4})
    config = dict(match_vars=["g", "x"], treatment_col="t", id_col="id", tolerance=0)
    legacy = ControlMatcher(MatchConfig(**config)).fit_match(data)
    explicit = ControlMatcher(MatchConfig(**config, stratify_vars=["g", "x"])).fit_match(data)
    assert legacy.pairs.equals(explicit.pairs)
    coarse = ControlMatcher(MatchConfig(**config, stratify_vars=["g"])).fit_match(data)
    matched = [r for r in coarse.pairs.to_pylist() if r["match_status"] == "matched"]
    assert {(r["target_id"], r["control_id"]) for r in matched} == {(0., 4.), (1., 5.), (2., 6.), (3., 7.)}
    assert any(r["match_strength"] < 1 for r in matched)
    assert coarse.report.capacity.summary["n_target_strata"] == 2
    with pytest.raises(ValueError, match="categorical"):
        ControlMatcher(MatchConfig(**config, stratify_vars=[])).fit_match(data)


@pytest.mark.parametrize("grouping", [["other"], ["x", "x"]])
def test_invalid_grouping(grouping):
    with pytest.raises(ValueError, match="stratify_vars"):
        MatchConfig(match_vars=["x"], treatment_col="t", stratify_vars=grouping)
