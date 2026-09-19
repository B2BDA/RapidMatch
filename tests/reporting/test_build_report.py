import pandas as pd

from rapidmatch.balance.checker import BalanceRow
from rapidmatch.drift.correct import TrimEvent
from rapidmatch.reporting.report import build_report


def test_report_assembles_three_artifacts() -> None:
    targets = pd.DataFrame(
        {
            "match_status": ["matched", "matched", "below_tolerance"],
            "thin_stratum": [False, True, False],
        }
    )
    before = [
        BalanceRow("seg", "monitor", "js", 0.2, 0.1, True),
    ]
    after = [
        BalanceRow("seg", "monitor", "js", 0.05, 0.1, False),
    ]
    events = [
        TrimEvent("seg", "js", "A", 2, 0.5, 0.8, [11, 12]),
    ]
    report = build_report(targets, cutoff=0.4, before=before, after=after, events=events)
    assert report.coverage["n_matched"] == 2
    assert report.coverage["tolerance_cutoff"] == 0.4
    assert list(report.balance["variable"]) == ["seg"]
    assert report.drift_log.iloc[0]["n_removed"] == 2
    payload = report.to_dict()
    assert "coverage" in payload and "balance" in payload and "drift_log" in payload
