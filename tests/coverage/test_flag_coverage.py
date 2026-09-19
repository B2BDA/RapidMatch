from rapidmatch.binning.get_stratum_counts import StratumCount
from rapidmatch.coverage.flag_coverage import flag_coverage


def test_thin_stratum_still_eligible() -> None:
    counts = [
        StratumCount("a", n_target=3, n_control=0),
        StratumCount("b", n_target=3, n_control=2),
        StratumCount("c", n_target=3, n_control=10),
    ]
    cov = flag_coverage(counts, min_control_pool_size=5)
    assert "a" in cov.no_control
    assert "a" not in cov.eligible
    assert "b" in cov.thin
    assert "b" in cov.eligible
    assert "c" in cov.eligible
    assert "c" not in cov.thin
