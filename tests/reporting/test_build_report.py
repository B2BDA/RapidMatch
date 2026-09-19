import pyarrow as pa

from rapidmatch.balance.checker import BalanceRow
from rapidmatch.drift.correct import TrimEvent
from rapidmatch.reporting.report import build_report


def test_report_assembles_three_artifacts() -> None:
    targets = pa.table(
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
    assert report.balance["variable"].to_pylist() == ["seg"]
    assert report.drift_log["n_removed"].to_pylist() == [2]
    payload = report.to_dict()
    assert "coverage" in payload and "balance" in payload and "drift_log" in payload


def test_empty_events_gives_typed_table() -> None:
    targets = pa.table(
        {
            "match_status": ["matched"],
            "thin_stratum": [False],
        }
    )
    report = build_report(targets, cutoff=0.0, before=[], after=[], events=[])
    assert report.drift_log.num_rows == 0
    assert "n_removed" in report.drift_log.column_names